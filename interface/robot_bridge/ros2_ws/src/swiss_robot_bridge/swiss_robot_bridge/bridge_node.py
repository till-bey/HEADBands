"""Outbound authenticated UniX AI Panther ROS 2 bridge node connecting to Cloud Run."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from datetime import datetime, timezone

UTC = timezone.utc  # datetime.UTC needs Python 3.11; the ROS Humble container has 3.10
from typing import Any

import websockets

from swiss_robot_bridge.camera_manager import PantherCameraManager
from swiss_robot_bridge.controllers import ControllerSwitcher, StepStream, TimedStream
from swiss_robot_bridge.local_safety import (
    LocalSafetyRejectedError,
    LocalSafetySupervisor,
    ManualCommand,
    PantherActuationPlan,
    TwistSetpoint,
    THROWS,
    retime_from_current_pose,
    throw_points,
)
from swiss_robot_bridge.panther_topics import (
    GRIPPER_DEFAULT_VEL,
    HEAD_JOINTS,
    HEAD_MAX_VEL_RPS,
    LIFT_MAX_VEL_MPS,
    MAX_ARM_JOINT_VEL_RPS,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_CONTROLLER,
    LEFT_TRAJ_CONTROLLER,
    RIGHT_GRIPPER_CONTROLLER,
    RIGHT_GRIPPER_MAX,
    RIGHT_TRAJ_CONTROLLER,
    TOPIC_LEFT_ARM_TRAJ,
    THROW_PAUSE_S,
    PANTHER_COMMAND_TOPICS,
    PANTHER_RMW_IMPLEMENTATION,
    PANTHER_ROS_DOMAIN_ID,
    RIGHT_ARM_JOINTS,
    SRV_ULTRASONIC_SWITCH,
    TOPIC_BATTERY_STATE,
    TOPIC_CAMERA_COLOR_COMPRESSED,
    TOPIC_CAMERA_COLOR_RAW,
    TOPIC_CAMERA_DEPTH_RAW,
    TOPIC_CMD_VEL,
    TOPIC_DIAGNOSTICS,
    TOPIC_FRONT_SCAN,
    TOPIC_HEAD_COMMANDS,
    TOPIC_JOINT_STATES,
    TOPIC_LEFT_GRIPPER_CMD,
    TOPIC_LEFT_GRIPPER_STATES,
    TOPIC_LIFT_COMMANDS,
    TOPIC_LIFT_STATES,
    TOPIC_ODOM,
    TOPIC_POSE,
    TOPIC_PREEMPT_TELEOP,
    TOPIC_RIGHT_ARM_TRAJ,
    TOPIC_RIGHT_GRIPPER_CMD,
    TOPIC_RIGHT_GRIPPER_STATES,
    TOPIC_SIM_FRONT_SCAN,
    single_position,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("swiss_robot_bridge")


class ROS2HardwareAdapter:
    """Binds strictly to the real UniX AI Panther ROS 2 topics & services (`ROS_DOMAIN_ID=15`, CycloneDDS).

    Falls back cleanly to standalone hardware-simulation mode when `rclpy` is not installed.
    """

    def __init__(self, supervisor: LocalSafetySupervisor, camera_manager: PantherCameraManager) -> None:
        self._supervisor = supervisor
        self._camera = camera_manager
        self._rclpy: Any = None
        self._node: Any = None
        self.sim_mode: bool = True

        # Publishers
        self._cmd_vel_pub: Any = None
        self._preempt_pub: Any = None
        self._arm_traj_pubs: dict[str, Any] = {}  # arm trajectory controller -> its joint_trajectory publisher
        self._switcher: ControllerSwitcher | None = None
        self._head_stream = TimedStream()  # head_forward_position_controller set-points over time
        self._gripper_stream = StepStream()  # gripper commands at their times
        self._head_pub: Any = None
        self._lift_pub: Any = None
        self._left_gripper_pub: Any = None
        self._right_gripper_pub: Any = None

        # Service client
        self._ultrasonic_client: Any = None
        self._ultrasonics_enabled: bool = False

        # Message classes
        self._twist_cls: Any = None
        self._empty_cls: Any = None
        self._f64_multi_cls: Any = None
        self._joint_state_cls: Any = None
        self._joint_traj_cls: Any = None
        self._joint_traj_point_cls: Any = None
        self._gripper_cmd_cls: Any = None
        self._set_bool_cls: Any = None
        self._image_cls: Any = None
        self._compressed_image_cls: Any = None

        # Lazy camera subscriptions (created on demand, destroyed when idle)
        self._camera_subs: list[Any] = []

        # Live Panther state tracked from ROS 2 subscriptions
        self.position_xyz: dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.lift_height_m: float | None = None
        self.left_gripper_pos: float | None = None
        self.right_gripper_pos: float | None = None
        self.head_joints_rad: list[float] | None = None
        self.last_master_trajectory: list[tuple[float, tuple[float, ...], float | None]] = []
        self._arm_traj_joints: tuple[str, ...] = ()
        self._arm_traj_until: float = 0.0  # monotonic end of the arm trajectory being played
        self.last_published_topics: list[str] = list(PANTHER_COMMAND_TOPICS)
        # Every joint position seen on the state topics, for the Web UI simulation view (read-only).
        # The lift and gripper state topics have no joint names on the robot, so they are stored
        # under the URDF names body_joint / joint_lf / joint_rf.
        self.joint_positions: dict[str, float] = {}
        self.joint_positions_seq: int = 0

        os.environ.setdefault("ROS_DOMAIN_ID", str(PANTHER_ROS_DOMAIN_ID))
        os.environ.setdefault("RMW_IMPLEMENTATION", PANTHER_RMW_IMPLEMENTATION)

        try:
            import rclpy
            from diagnostic_msgs.msg import DiagnosticArray
            from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import BatteryState, CompressedImage, Image, JointState, LaserScan
            from std_msgs.msg import Empty, Float64MultiArray
            from std_srvs.srv import SetBool
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

            if not rclpy.ok():
                rclpy.init()
            self._rclpy = rclpy
            self._node = rclpy.create_node("panther_safety_bridge")
            self.sim_mode = False

            self._twist_cls = Twist
            self._empty_cls = Empty
            self._f64_multi_cls = Float64MultiArray
            self._joint_state_cls = JointState
            self._joint_traj_cls = JointTrajectory
            self._joint_traj_point_cls = JointTrajectoryPoint
            self._set_bool_cls = SetBool
            self._image_cls = Image
            self._compressed_image_cls = CompressedImage

            # Core Panther command publishers. Arms go straight to their trajectory controllers (switched
            # in on demand, as master_gui does); master_controller no longer exists on clean_up.
            self._cmd_vel_pub = self._node.create_publisher(Twist, TOPIC_CMD_VEL, 10)
            self._preempt_pub = self._node.create_publisher(Empty, TOPIC_PREEMPT_TELEOP, 10)
            self._arm_traj_pubs = {
                LEFT_TRAJ_CONTROLLER: self._node.create_publisher(JointTrajectory, TOPIC_LEFT_ARM_TRAJ, 10),
                RIGHT_TRAJ_CONTROLLER: self._node.create_publisher(JointTrajectory, TOPIC_RIGHT_ARM_TRAJ, 10),
            }
            try:
                self._switcher = ControllerSwitcher(rclpy, self._node)
            except ImportError:
                logger.warning("controller_manager_msgs missing: arm controllers will not be switched in")
            self._head_pub = self._node.create_publisher(Float64MultiArray, TOPIC_HEAD_COMMANDS, 10)
            self._lift_pub = self._node.create_publisher(JointState, TOPIC_LIFT_COMMANDS, 10)

            try:
                from unix_msgs.msg import GripperCommand

                self._gripper_cmd_cls = GripperCommand
                self._left_gripper_pub = self._node.create_publisher(
                    GripperCommand, TOPIC_LEFT_GRIPPER_CMD, 10
                )
                self._right_gripper_pub = self._node.create_publisher(
                    GripperCommand, TOPIC_RIGHT_GRIPPER_CMD, 10
                )
            except ImportError:
                logger.warning(
                    "unix_msgs not found in PYTHONPATH; gripper topics (%s, %s) disabled",
                    TOPIC_LEFT_GRIPPER_CMD,
                    TOPIC_RIGHT_GRIPPER_CMD,
                )

            self._ultrasonic_client = self._node.create_client(SetBool, SRV_ULTRASONIC_SWITCH)

            # Panther telemetry & safety subscriptions
            self._node.create_subscription(BatteryState, TOPIC_BATTERY_STATE, self._on_battery, 10)
            self._node.create_subscription(Odometry, TOPIC_ODOM, self._on_odom, 10)
            self._node.create_subscription(PoseWithCovarianceStamped, TOPIC_POSE, self._on_pose, 10)
            self._node.create_subscription(DiagnosticArray, TOPIC_DIAGNOSTICS, self._on_diagnostics, 10)
            self._node.create_subscription(JointState, TOPIC_JOINT_STATES, self._on_joint_states, 10)
            self._node.create_subscription(JointState, TOPIC_LIFT_STATES, self._on_lift_state, 10)
            self._node.create_subscription(
                JointState, TOPIC_LEFT_GRIPPER_STATES, self._on_left_gripper_state, 10
            )
            self._node.create_subscription(
                JointState, TOPIC_RIGHT_GRIPPER_STATES, self._on_right_gripper_state, 10
            )
            self._node.create_subscription(LaserScan, TOPIC_FRONT_SCAN, self._on_front_scan, 10)
            self._node.create_subscription(LaserScan, TOPIC_SIM_FRONT_SCAN, self._on_front_scan, 10)

            logger.info(
                "Initialized UniX AI Panther ROS 2 node (ROS_DOMAIN_ID=%s, RMW=%s)",
                os.getenv("ROS_DOMAIN_ID"),
                os.getenv("RMW_IMPLEMENTATION"),
            )
        except ImportError:
            self._camera.allow_sim_fallback = True
            logger.info("rclpy not installed; running Panther ROS 2 bridge in standalone hardware-sim mode")

    def spin_once(self) -> None:
        if self._rclpy is not None and self._node is not None and self._rclpy.ok():
            self._rclpy.spin_once(self._node, timeout_sec=0.0)

    # --- Lazy / On-Demand Camera ROS 2 Subscriptions ---
    def activate_camera_subscriptions(self) -> None:
        if self._node is None or self._camera_subs:
            return
        if self._image_cls is not None:
            self._camera_subs.append(
                self._node.create_subscription(
                    self._image_cls,
                    TOPIC_CAMERA_COLOR_RAW,
                    lambda msg: self._camera.ingest_ros_image(msg, TOPIC_CAMERA_COLOR_RAW),
                    5,
                )
            )
            self._camera_subs.append(
                self._node.create_subscription(
                    self._image_cls,
                    TOPIC_CAMERA_DEPTH_RAW,
                    self._camera.ingest_ros_depth_image,
                    5,
                )
            )
        if self._compressed_image_cls is not None:
            self._camera_subs.append(
                self._node.create_subscription(
                    self._compressed_image_cls,
                    TOPIC_CAMERA_COLOR_COMPRESSED,
                    lambda msg: self._camera.ingest_ros_compressed_image(msg, TOPIC_CAMERA_COLOR_COMPRESSED),
                    5,
                )
            )
        logger.info("Activated on-demand ROS 2 camera subscriptions (%s)", TOPIC_CAMERA_COLOR_RAW)

    def deactivate_camera_subscriptions(self) -> None:
        if self._node is None or not self._camera_subs:
            return
        for sub in self._camera_subs:
            try:
                self._node.destroy_subscription(sub)
            except Exception:  # noqa: BLE001
                pass
        self._camera_subs.clear()
        logger.info("Deactivated idle ROS 2 camera subscriptions")

    def count_color_camera_publishers(self) -> int:
        if self._node is None:
            return 0
        try:
            return int(self._node.count_publishers(TOPIC_CAMERA_COLOR_RAW)) + int(
                self._node.count_publishers(TOPIC_CAMERA_COLOR_COMPRESSED)
            )
        except Exception:  # noqa: BLE001
            return 0

    # --- Panther State Subscription Callbacks ---
    def _on_battery(self, msg: Any) -> None:
        pct = float(getattr(msg, "percentage", -1.0))
        if math.isfinite(pct) and pct >= 0.0:
            # ROS BatteryState convention may report 0.0..1.0 or 0.0..100.0
            normalized = pct * 100.0 if pct <= 1.05 else pct
            self._supervisor.battery_percent = round(min(max(normalized, 0.0), 100.0), 1)

    def _on_odom(self, msg: Any) -> None:
        pos = msg.pose.pose.position
        self.position_xyz = {
            "x": round(float(pos.x), 3),
            "y": round(float(pos.y), 3),
            "z": round(float(pos.z), 3),
        }
        ori = getattr(msg.pose.pose, "orientation", None)
        yaw = 0.0
        if ori is not None:
            qx = float(getattr(ori, "x", 0.0))
            qy = float(getattr(ori, "y", 0.0))
            qz = float(getattr(ori, "z", 0.0))
            qw = float(getattr(ori, "w", 1.0))
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw = math.atan2(siny_cosp, cosy_cosp)
        self._supervisor.update_odometry(float(pos.x), float(pos.y), yaw, time.monotonic())

    def _on_pose(self, msg: Any) -> None:
        pos = msg.pose.pose.position
        self.position_xyz = {
            "x": round(float(pos.x), 3),
            "y": round(float(pos.y), 3),
            "z": round(float(pos.z), 3),
        }

    def _on_diagnostics(self, msg: Any) -> None:
        """Inspect `/diagnostics` for the Panther hardware/wireless E-Stop entry."""
        for status in getattr(msg, "status", []):
            name = str(getattr(status, "name", "")).lower()
            message = str(getattr(status, "message", "")).lower()
            level = getattr(status, "level", 0)
            level = level[0] if isinstance(level, bytes) else int(level)  # rclpy gives DiagnosticStatus.level as b"\x00"
            if "estop" in name or "e-stop" in name or "emergency" in name:
                triggered = level >= 2 or any(
                    tok in message for tok in ("active", "pressed", "true", "triggered", "engaged")
                )
                for kv in getattr(status, "values", []):
                    k = str(getattr(kv, "key", "")).lower()
                    v = str(getattr(kv, "value", "")).lower()
                    if ("estop" in k or "e-stop" in k or "emergency" in k) and v in {"1", "true", "active"}:
                        triggered = True
                if triggered:
                    self._supervisor.trigger_local_estop()

    def _record_joints(self, name_to_pos: dict[str, Any]) -> None:
        for name, value in name_to_pos.items():
            try:
                pos = float(value)
            except (TypeError, ValueError):
                continue
            if name and math.isfinite(pos):
                self.joint_positions[str(name)] = round(pos, 5)
        self.joint_positions_seq += 1

    def _on_joint_states(self, msg: Any) -> None:
        names = list(getattr(msg, "name", []))
        positions = list(getattr(msg, "position", []))
        name_to_pos = dict(zip(names, positions, strict=False))
        self._record_joints(name_to_pos)
        if "body_joint" in name_to_pos and float(name_to_pos["body_joint"]) > -1.0:
            self.lift_height_m = round(float(name_to_pos["body_joint"]), 4)
        if "joint_lf" in name_to_pos and float(name_to_pos["joint_lf"]) > -1.0:
            self.left_gripper_pos = round(float(name_to_pos["joint_lf"]), 3)
        if "joint_rf" in name_to_pos and float(name_to_pos["joint_rf"]) > -1.0:
            self.right_gripper_pos = round(float(name_to_pos["joint_rf"]), 3)
        if "joint_head0" in name_to_pos and "joint_head1" in name_to_pos:
            self.head_joints_rad = [
                round(float(name_to_pos["joint_head0"]), 4),
                round(float(name_to_pos["joint_head1"]), 4),
            ]

    def _on_lift_state(self, msg: Any) -> None:
        pos = single_position(msg)
        if pos is not None:
            self.lift_height_m = round(pos, 4)
            self._record_joints({"body_joint": pos})

    def _on_left_gripper_state(self, msg: Any) -> None:
        pos = single_position(msg)
        if pos is not None:
            self.left_gripper_pos = round(pos, 3)
            self._record_joints({"joint_lf": pos})

    def _on_right_gripper_state(self, msg: Any) -> None:
        pos = single_position(msg)
        if pos is not None:
            self.right_gripper_pos = round(pos, 3)
            self._record_joints({"joint_rf": pos})

    def _on_front_scan(self, msg: Any) -> None:
        """Compute closest valid front-arc obstacle distance from `/scan` (`link_front_radar`)."""
        ranges = list(getattr(msg, "ranges", []))
        if not ranges:
            return
        r_min = max(float(getattr(msg, "range_min", 0.05)), 0.05)
        r_max = float(getattr(msg, "range_max", 15.0))
        angle_min = float(getattr(msg, "angle_min", -math.pi))
        angle_inc = float(getattr(msg, "angle_increment", 0.0))

        closest: float | None = None
        for idx, dist in enumerate(ranges):
            if not math.isfinite(dist) or dist < r_min or dist > r_max:
                continue
            angle = angle_min + idx * angle_inc if angle_inc != 0.0 else 0.0
            if abs(angle) <= math.radians(35.0):
                if closest is None or dist < closest:
                    closest = dist
        self._supervisor.front_obstacle_distance_m = round(closest, 3) if closest is not None else None

    # --- Panther Command Publishers ---
    def ensure_ultrasonics_enabled(self) -> None:
        """Keep `/movebase3_drive_controller/ultrasonic_switch` enabled (`True`) when driving."""
        if self._ultrasonics_enabled or self._ultrasonic_client is None or self._set_bool_cls is None:
            return
        if self._ultrasonic_client.service_is_ready():
            req = self._set_bool_cls.Request()
            req.data = True
            self._ultrasonic_client.call_async(req)
            self._ultrasonics_enabled = True

    def publish_twist(self, setpoint: TwistSetpoint) -> None:
        if self._cmd_vel_pub is not None and self._twist_cls is not None:
            msg = self._twist_cls()
            msg.linear.x = float(setpoint.linear_x)
            msg.linear.y = float(setpoint.linear_y)
            msg.angular.z = float(setpoint.angular_z)
            self._cmd_vel_pub.publish(msg)

    def publish_preempt_stop(self) -> None:
        self.publish_twist(TwistSetpoint(0.0, 0.0, 0.0))
        if self._preempt_pub is not None and self._empty_cls is not None:
            self._preempt_pub.publish(self._empty_cls())
        self.hold_arms_if_moving()

    def hold_arms_if_moving(self) -> None:
        """Stop a throw or gesture still playing: each arm holds where it is, head and grippers stop.

        A one-point trajectory at the measured position, 0.3 s out, replaces the goal the arm's
        trajectory controller is following.
        """
        self._head_stream.clear()
        self._gripper_stream.clear()
        if not self._arm_traj_joints or time.monotonic() > self._arm_traj_until:
            return
        self._arm_traj_until = 0.0
        if self._joint_traj_cls is None:
            return
        for controller, joints in ((LEFT_TRAJ_CONTROLLER, LEFT_ARM_JOINTS), (RIGHT_TRAJ_CONTROLLER, RIGHT_ARM_JOINTS)):
            pub = self._arm_traj_pubs.get(controller)
            if pub is None or not set(joints) <= set(self._arm_traj_joints):
                continue
            if not all(j in self.joint_positions for j in joints):
                continue
            msg = self._joint_traj_cls()
            msg.joint_names = list(joints)
            pt = self._joint_traj_point_cls()
            pt.positions = [float(self.joint_positions[j]) for j in joints]
            pt.time_from_start.sec = 0
            pt.time_from_start.nanosec = 300_000_000
            msg.points.append(pt)
            pub.publish(msg)
        logger.warning("Arm trajectory interrupted: arms held at their measured position")

    def publish_head(self, head0_rad: float, head1_rad: float) -> None:
        self.head_joints_rad = [round(float(head0_rad), 4), round(float(head1_rad), 4)]
        if self._head_pub is not None and self._f64_multi_cls is not None:
            msg = self._f64_multi_cls()
            msg.data = [float(head0_rad), float(head1_rad)]
            self._head_pub.publish(msg)

    def _current_head(self) -> tuple[float, float]:
        fallback = self.head_joints_rad or [0.0, 0.0]
        return (
            float(self.joint_positions.get(HEAD_JOINTS[0], fallback[0])),
            float(self.joint_positions.get(HEAD_JOINTS[1], fallback[1])),
        )

    def _send_head(self, pitch: float, yaw: float) -> None:
        """One command to head_forward_position_controller, in its element order."""
        if self._supervisor.head_order == "yaw_pitch":
            self.publish_head(yaw, pitch)
        else:
            self.publish_head(pitch, yaw)
        self.head_joints_rad = [round(pitch, 4), round(yaw, 4)]
        if self.sim_mode:
            self._record_joints({HEAD_JOINTS[0]: pitch, HEAD_JOINTS[1]: yaw})

    def publish_head_targets(
        self,
        targets: dict[str, float],
        fallback_positions: tuple[float, float] | None = None,
    ) -> None:
        """Move only the named head joints, smoothly: an eased ramp at <= HEAD_MAX_VEL_RPS, streamed
        to head_forward_position_controller every 0.05 s by tick_streams (it jumps to each command)."""
        del fallback_positions  # kept for the call signature; element order is handled in _send_head
        cur_p, cur_y = self._current_head()
        tgt_p = float(targets.get(HEAD_JOINTS[0], cur_p))
        tgt_y = float(targets.get(HEAD_JOINTS[1], cur_y))
        gap = max(abs(tgt_p - cur_p), abs(tgt_y - cur_y))
        duration = max(0.3, (gap / HEAD_MAX_VEL_RPS) * 1.6)  # eased profile peaks ~1.6x the mean speed
        steps = max(1, int(math.ceil(duration / 0.1)))
        now = time.monotonic()
        points = [(now, (cur_p, cur_y))]
        for k in range(1, steps + 1):
            s = k / steps
            a = 0.5 - 0.5 * math.cos(math.pi * s)
            points.append((now + s * duration, (cur_p + a * (tgt_p - cur_p), cur_y + a * (tgt_y - cur_y))))
        self._head_stream.points = points

    def publish_lift(self, z_m: float, velocity_mps: float) -> None:
        self.lift_height_m = round(float(z_m), 4)
        if self._lift_pub is not None and self._joint_state_cls is not None:
            msg = self._joint_state_cls()
            msg.name = [""]
            msg.position = [float(z_m)]
            msg.velocity = [float(velocity_mps)]
            self._lift_pub.publish(msg)

    def publish_gripper(self, side: str, position: float, velocity: float, mode: str = "parallel") -> None:
        if side == "right":
            position = min(float(position), RIGHT_GRIPPER_MAX)  # as master_gui: right gripper closes no further
            self.right_gripper_pos = round(float(position), 3)
            pub = self._right_gripper_pub
        else:
            self.left_gripper_pos = round(float(position), 3)
            pub = self._left_gripper_pub
        if self.sim_mode:
            self._record_joints({"joint_rf" if side == "right" else "joint_lf": position})

        if pub is not None and self._gripper_cmd_cls is not None:
            msg = self._gripper_cmd_cls()
            msg.mode = str(mode)
            msg.command.name = [""]
            msg.command.position = [float(position)]
            msg.command.velocity = [float(velocity)]
            pub.publish(msg)

    def publish_master_trajectory(self, plan: PantherActuationPlan) -> None:
        """Play a whole-upper-body trajectory (throw, generated gesture) on the robot's own controllers.

        Arms: each arm's JointTrajectory goes to its trajectory controller, switched in first (as
        master_gui does; master_controller is gone). Head and grippers: streamed at their times by
        tick_streams.
        """
        if not plan.master_trajectory_points:
            return
        self.last_master_trajectory = list(plan.master_trajectory_points)
        names = tuple(plan.master_trajectory_joints or (RIGHT_ARM_JOINTS + ("joint_rf",)))
        rows: list[tuple[float, dict[str, float]]] = []
        for t_s, q, grip in plan.master_trajectory_points:
            values = list(q) + ([grip] if grip is not None and len(names) == len(q) + 1 else [])
            rows.append((max(0.05, float(t_s)), dict(zip(names, (float(v) for v in values)))))

        arms = {
            c: js
            for c, js in ((LEFT_TRAJ_CONTROLLER, LEFT_ARM_JOINTS), (RIGHT_TRAJ_CONTROLLER, RIGHT_ARM_JOINTS))
            if all(j in names for j in js)
        }
        grippers = [g for g in ("joint_lf", "joint_rf") if g in names]
        needed = list(arms) + [LEFT_GRIPPER_CONTROLLER if g == "joint_lf" else RIGHT_GRIPPER_CONTROLLER for g in grippers]
        if needed and self._switcher is not None and not self._switcher.ensure_active(needed):
            logger.error("Could not switch %s in; trajectory not sent", needed)
            return

        now = time.monotonic()
        self._arm_traj_joints = tuple(j for js in arms.values() for j in js)
        self._arm_traj_until = now + rows[-1][0] + 0.5
        if self.sim_mode:
            self._record_joints({j: v for j, v in rows[-1][1].items() if j.startswith(("joint_la", "joint_ra"))})

        if self._joint_traj_cls is not None:
            for controller, joints in arms.items():
                msg = self._joint_traj_cls()
                msg.joint_names = list(joints)
                for t_s, vals in rows:
                    pt = self._joint_traj_point_cls()
                    pt.time_from_start.sec = int(t_s)
                    pt.time_from_start.nanosec = int(round((t_s - int(t_s)) * 1e9)) % 1_000_000_000
                    pt.positions = [vals[j] for j in joints]
                    msg.points.append(pt)
                self._arm_traj_pubs[controller].publish(msg)

        if any(j in names for j in HEAD_JOINTS):
            self._head_stream.points = [(now, self._current_head())] + [
                (now + t_s, (vals.get(HEAD_JOINTS[0]), vals.get(HEAD_JOINTS[1]))) for t_s, vals in rows
            ]
        events: list[tuple[float, str, float]] = []
        for g in grippers:
            last = None
            for t_s, vals in rows:
                if g in vals and vals[g] != last:
                    events.append((now + t_s, g, vals[g]))
                    last = vals[g]
        self._gripper_stream.events = sorted(events)

    def tick_streams(self, now: float | None = None) -> None:
        """Called every 0.05 s: next head set-point and any gripper command that is due."""
        now = time.monotonic() if now is None else now
        sample = self._head_stream.sample(now)
        if sample is not None:
            cur_p, cur_y = self._current_head()
            pitch = cur_p if sample[0] is None else float(sample[0])
            yaw = cur_y if sample[1] is None else float(sample[1])
            self._send_head(pitch, yaw)
        for joint, value in self._gripper_stream.due(now):
            self.publish_gripper("left" if joint == "joint_lf" else "right", value, 1.0)

    def apply_manual(self, cmd: ManualCommand) -> None:
        """Joint targets from the manual panel (already checked by LocalSafetySupervisor.check_manual).

        An arm goes there in one trajectory point timed for 0.9 x MAX_ARM_JOINT_VEL_RPS, its other joints holding;
        the head ramps smoothly; grippers and lift get one command each."""
        targets = cmd.joints
        head = {j: v for j, v in targets.items() if j in HEAD_JOINTS}
        if head:
            self.publish_head_targets(head)
        if "body_joint" in targets:
            self.publish_lift(targets["body_joint"], LIFT_MAX_VEL_MPS)
        for joint, side in (("joint_lf", "left"), ("joint_rf", "right")):
            if joint in targets:
                self.publish_gripper(side, targets[joint], GRIPPER_DEFAULT_VEL)
        arm = tuple(j for js in (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS) if any(j in targets for j in js) for j in js)
        if not arm:
            return
        if not all(j in self.joint_positions for j in arm):
            raise LocalSafetyRejectedError("Arm position unknown (no /joint_states yet); arm not moved.")
        goal = tuple(float(targets.get(j, self.joint_positions[j])) for j in arm)
        gap = max(abs(q - self.joint_positions[j]) for j, q in zip(arm, goal))
        duration = max(0.3, gap / (MAX_ARM_JOINT_VEL_RPS * 0.9))
        self.publish_master_trajectory(
            PantherActuationPlan(
                command_id="manual",
                action="manual",
                master_trajectory_joints=arm,
                master_trajectory_points=[(round(duration, 3), goal, None)],
            )
        )

    def apply_panther_plan(self, plan: PantherActuationPlan) -> None:
        """Dispatch a validated `PantherActuationPlan` across the real UniX AI Panther topics."""
        if plan.preempt_teleop:
            self.publish_preempt_stop()
            return

        if plan.enable_ultrasonics:
            self.ensure_ultrasonics_enabled()

        if plan.head_targets:
            self.publish_head_targets(plan.head_targets, plan.head_positions)
        elif plan.head_positions is not None:
            self.publish_head(plan.head_positions[0], plan.head_positions[1])

        if plan.lift_position_m is not None:
            self.publish_lift(plan.lift_position_m, plan.lift_velocity_mps)

        if plan.master_trajectory_points:
            if plan.action in ("generated_gesture", "throw_object") and plan.master_trajectory_joints:
                # The arm may be anywhere: give it time to reach the first point slowly.
                plan.master_trajectory_points = retime_from_current_pose(
                    tuple(plan.master_trajectory_joints), plan.master_trajectory_points, self.joint_positions
                )
            self.publish_master_trajectory(plan)

        traj_grippers = set(plan.master_trajectory_joints or ()) & {"joint_lf", "joint_rf"}
        gripper_joint = "joint_rf" if plan.gripper_side == "right" else "joint_lf"
        if plan.master_trajectory_points and not plan.master_trajectory_joints:
            traj_grippers.add("joint_rf")  # default throw joints include the right gripper
        if (
            plan.gripper_side is not None
            and plan.gripper_position is not None
            and gripper_joint not in traj_grippers  # the trajectory times this gripper itself (throw release)
        ):
            self.publish_gripper(
                side=plan.gripper_side,
                position=plan.gripper_position,
                velocity=plan.gripper_velocity,
                mode=plan.gripper_mode,
            )

        self.publish_twist(plan.twist)


async def run_bridge_loop() -> None:
    gateway_ws_url = os.getenv("ROBOT_GATEWAY_WS_URL", "ws://localhost:8080/ws/robot-bridge")
    shared_token = os.getenv("ROBOT_BRIDGE_SHARED_TOKEN", "")
    robot_id = os.getenv("ROBOT_ID", "unix-panther-01")
    ros_domain_id = int(os.getenv("ROS_DOMAIN_ID", str(PANTHER_ROS_DOMAIN_ID)))

    supervisor = LocalSafetySupervisor(
        max_linear_speed_mps=float(os.getenv("ROBOT_MAX_LINEAR_SPEED_MPS", "0.5")),
        max_workspace_radius_m=float(os.getenv("ROBOT_MAX_WORKSPACE_RADIUS_M", "10.0")),
        watchdog_timeout_seconds=float(os.getenv("LOCAL_WATCHDOG_TIMEOUT_SECONDS", "2.5")),
        head_order=os.getenv("PANTHER_HEAD_ORDER", "pitch_yaw"),
        head_full_range=os.getenv("PANTHER_HEAD_FULL_RANGE", "1").strip().lower() in {"1", "true", "yes", "on"},
        throw_speed=float(os.getenv("PANTHER_THROW_SPEED", "1.0")),
    )
    for name in THROWS:  # a bad PANTHER_THROW_SPEED stops the bridge here, not mid-throw
        throw = throw_points(supervisor.throw_speed, name)
        peak = max(
            abs(b[1][i] - a[1][i]) / (b[0] - a[0]) for a, b in zip(throw, throw[1:]) for i in range(len(a[1]))
        )
        release = next((t for t, _q, g in throw if g == 0.0), None)
        logger.info("Throw %r: speed x%.2f, %.2f s after the 2 s pause, gripper open at %s, peak arm joint speed "
                    "%.2f rad/s", name, supervisor.throw_speed, throw[-1][0] - 2 * THROW_PAUSE_S,
                    f"{release - 2 * THROW_PAUSE_S:.2f} s" if release is not None else "never", peak)
    camera_mgr = PantherCameraManager(
        idle_timeout_seconds=float(os.getenv("PANTHER_CAMERA_IDLE_TIMEOUT_SECONDS", "90.0")),
        startup_timeout_seconds=float(os.getenv("PANTHER_CAMERA_STARTUP_TIMEOUT_SECONDS", "4.5")),
    )
    ros_hw = ROS2HardwareAdapter(supervisor=supervisor, camera_manager=camera_mgr)

    headers: dict[str, str] = {}
    if shared_token:
        headers["Authorization"] = f"Bearer {shared_token}"

    async def _local_watchdog_and_ros_spin_loop() -> None:
        """20 Hz local control loop (so a manual jog stops within 0.25 s of its last command): spins ROS 2 callbacks, streams `/cmd_vel`, zeroes on timeout, idles camera."""
        while True:
            ros_hw.spin_once()
            now_mono = time.monotonic()
            ros_hw.tick_streams(now_mono)  # head set-points and timed gripper commands
            safe_twist = supervisor.evaluate_watchdog(now_mono)
            promoted = supervisor.pop_promoted_plan()
            if promoted is not None:
                ros_hw.apply_panther_plan(promoted)
                logger.info(
                    "Started queued Panther command %s (%s) -> twist=%s duration=%.2fs",
                    promoted.command_id,
                    promoted.action,
                    promoted.twist,
                    promoted.twist_duration_seconds,
                )
            else:
                ros_hw.publish_twist(safe_twist)
            camera_mgr.check_idle_shutdown(
                now_mono,
                deactivate_ros_camera_cb=ros_hw.deactivate_camera_subscriptions,
            )
            await asyncio.sleep(0.05)

    watchdog_task = asyncio.create_task(_local_watchdog_and_ros_spin_loop())
    try:
        while True:
            try:
                async with websockets.connect(gateway_ws_url, additional_headers=headers) as ws:
                    logger.info("Connected outbound to cloud safety gateway: %s", gateway_ws_url)
                    supervisor.record_cloud_heartbeat(time.monotonic())

                    async def _capture_and_upload_camera(request_id: str) -> None:
                        frame = await camera_mgr.capture_frame_on_demand(
                            activate_ros_camera_cb=ros_hw.activate_camera_subscriptions,
                            spin_ros_once_cb=ros_hw.spin_once,
                            ros_color_publisher_count_cb=ros_hw.count_color_camera_publishers,
                        )
                        if frame is not None:
                            await ws.send(json.dumps(frame.to_ws_payload(request_id=request_id)))
                        else:
                            await ws.send(
                                json.dumps(
                                    {
                                        "type": "camera_frame_error",
                                        "request_id": request_id,
                                        "error": "No camera frame available from /camera/color/image_raw or fallback.",
                                    }
                                )
                            )

                    async def _telemetry_sender() -> None:
                        while True:
                            state_str = (
                                "estop"
                                if supervisor.local_estop_latched
                                else ("busy" if supervisor.active_command_id else "ready")
                            )
                            cam_ready = (
                                camera_mgr.has_fresh_frame(max_age_seconds=10.0)
                                or ros_hw.count_color_camera_publishers() > 0
                                or bool(camera_mgr.camera_launch_cmd)
                                or bool(camera_mgr.snapshot_url)
                                or camera_mgr.allow_sim_fallback
                            )
                            await ws.send(
                                json.dumps(
                                    {
                                        "type": "telemetry",
                                        "payload": {
                                            "robot_id": robot_id,
                                            "state": state_str,
                                            "estop_active": supervisor.local_estop_latched,
                                            "battery_percent": supervisor.battery_percent,
                                            "position": ros_hw.position_xyz,
                                            "active_command_id": supervisor.active_command_id,
                                            "ros_domain_id": ros_domain_id,
                                            "camera_ready": cam_ready,
                                            "camera_driver_active": camera_mgr.driver_active,
                                            "camera_source": camera_mgr.latest_source,
                                            "front_obstacle_distance_m": supervisor.front_obstacle_distance_m,
                                            "lift_height_m": ros_hw.lift_height_m,
                                            "left_gripper_pos": ros_hw.left_gripper_pos,
                                            "right_gripper_pos": ros_hw.right_gripper_pos,
                                            "head_joints_rad": ros_hw.head_joints_rad,
                                            "active_topics": ros_hw.last_published_topics,
                                            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                                        },
                                    }
                                )
                            )
                            await asyncio.sleep(1.0)

                    async def _joint_state_sender() -> None:
                        """5 Hz joint positions for the Web UI simulation view; sent only when they change."""
                        sent_seq = -1
                        while True:
                            seq = ros_hw.joint_positions_seq
                            if ros_hw.joint_positions and seq != sent_seq:
                                await ws.send(
                                    json.dumps(
                                        {
                                            "type": "joint_states",
                                            "positions": dict(ros_hw.joint_positions),
                                            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                                        }
                                    )
                                )
                                sent_seq = seq
                            await asyncio.sleep(0.2)

                    sender_task = asyncio.create_task(_telemetry_sender())
                    joint_task = asyncio.create_task(_joint_state_sender())
                    try:
                        async for raw_msg in ws:
                            data = json.loads(raw_msg)
                            msg_type = data.get("type")
                            supervisor.record_cloud_heartbeat(time.monotonic())

                            if msg_type == "emergency_stop":
                                supervisor.trigger_local_estop()
                                ros_hw.publish_preempt_stop()
                                logger.warning("E-STOP latched from cloud gateway!")

                            elif msg_type == "reset_estop":
                                supervisor.reset_local_estop()

                            elif msg_type == "manual":
                                try:
                                    manual = supervisor.check_manual(data.get("command"))
                                    if manual.base is not None:
                                        twist = supervisor.drive_manual(manual.base, time.monotonic())
                                        if twist.is_nonzero():
                                            ros_hw.ensure_ultrasonics_enabled()
                                        ros_hw.publish_twist(twist)
                                    ros_hw.apply_manual(manual)
                                except LocalSafetyRejectedError as exc:
                                    logger.error("Local safety gate rejected manual command: %s", exc)

                            elif msg_type == "capture_camera":
                                req_id = str(data.get("request_id") or f"cam-{int(time.time() * 1000)}")
                                asyncio.create_task(_capture_and_upload_camera(req_id))

                            elif msg_type == "execute_command":
                                try:
                                    plan, dispatched_now = supervisor.submit_cloud_command(
                                        data.get("command", {}),
                                        monotonic_now=time.monotonic(),
                                    )
                                    if dispatched_now:
                                        ros_hw.apply_panther_plan(plan)
                                        logger.info(
                                            "Executed Panther command %s (%s) -> twist=%s duration=%.2fs head=%s lift=%s gripper=%s:%s",
                                            plan.command_id,
                                            plan.action,
                                            plan.twist,
                                            plan.twist_duration_seconds,
                                            plan.head_positions,
                                            plan.lift_position_m,
                                            plan.gripper_side,
                                            plan.gripper_position,
                                        )
                                        if plan.trigger_camera:
                                            asyncio.create_task(
                                                _capture_and_upload_camera(f"auto-{plan.command_id}")
                                            )
                                    else:
                                        logger.info(
                                            "Queued sequential Panther command %s (%s, duration=%.2fs) behind %s",
                                            plan.command_id,
                                            plan.action,
                                            plan.twist_duration_seconds,
                                            supervisor.active_command_id,
                                        )
                                except LocalSafetyRejectedError as exc:
                                    ros_hw.publish_preempt_stop()
                                    logger.error("Local safety gate rejected cloud command: %s", exc)
                    finally:
                        sender_task.cancel()
                        joint_task.cancel()
            except Exception as exc:  # noqa: BLE001
                ros_hw.publish_preempt_stop()
                logger.warning("Cloud connection unavailable (%s); Panther safely halted locally", exc)
                await asyncio.sleep(3.0)
    finally:
        watchdog_task.cancel()
        camera_mgr.stop_driver()


def main() -> None:
    asyncio.run(run_bridge_loop())


if __name__ == "__main__":
    main()
