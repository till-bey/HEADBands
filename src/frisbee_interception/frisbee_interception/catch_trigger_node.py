"""frisbee_catch_trigger: catch a throw aimed at the chest (disarmed by default).

Before: the arms are put in the catching pose computed from the URDF (`frisbee_catch_pose`), with the existing
`panther_examples cartesian_follow --joints` tool. Then, once armed, for each throw:
  1. small corrections: both hands shift together (<= max_correction_m) in the catching plane toward the predicted
     catch point, through the existing <arm>_arm_joint_trajectory_position_controller (FollowJointTrajectory),
     with the same controller switch cartesian_follow does, joint speed <= max_joint_vel;
  2. the grippers close at (predicted arrival - gripper latency), through the existing gripper interface
     (unix_msgs/GripperCommand, built like panther_examples/teach.py).

Subscribes:  /frisbee/intercept, /frisbee/disable, /joint_states, /robot_description
Publishes:   gripper_topics (unix_msgs/GripperCommand) and trajectory goals, ONLY when armed
             /frisbee/catch_event (std_msgs/String, JSON): planned / cancelled / correction / would_correct /
             would_fire (disarmed) / fired / missed
Services:    ~/enable (std_srvs/SetBool). Arming is refused until gripper_latency_s and catch_window_px are
             measured, and (corrections enabled) until the arms are in the catching pose and the trajectory
             controllers could be activated. Disarming switches back to the forward controller.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, String
from std_srvs.srv import SetBool

from frisbee_msgs.msg import FrisbeeIntercept
from frisbee_interception.core.catch_pose import CatchPoseConfig, CatchPosePlanner
from frisbee_interception.core.catch_trigger import CatchConfig, CatchScheduler
from frisbee_interception.params import declare_section

FORWARD = "dual_arm_forward_position_controller"  # same names as panther_examples/robot.py
TRAJ = {"left": "left_arm_joint_trajectory_position_controller",
        "right": "right_arm_joint_trajectory_position_controller"}


class CatchTriggerNode(Node):
    def __init__(self) -> None:
        super().__init__("frisbee_catch_trigger")
        p = self.declare_parameter
        self.cb = ReentrantCallbackGroup()
        self.cfg = CatchConfig.from_dict(declare_section(self, "catch", CatchConfig()))
        self.topics = list(p("gripper_topics", ["/left_gripper/gripper_command", "/right_gripper/gripper_command"]).value)
        # right gripper: master_gui never closes it beyond 0.67 (panther_examples/robot.py RIGHT_GRIPPER_MAX)
        self.positions = [float(x) for x in p("close_positions", [1.0, 0.67]).value]
        self.velocity = float(p("close_velocity", 1.0).value)
        self.effort = float(p("close_effort", 0.5).value)
        self.max_data_age = float(p("max_data_age_s", 0.25).value)
        # corrections
        self.corr_enabled = bool(p("corrections.enabled", True).value)
        self.pose_file = p("corrections.pose_file", "").value
        self.max_joint_vel = min(float(p("corrections.max_joint_vel", 0.5).value), 0.5)  # MAX_JOINT_VEL cap
        self.min_resend = float(p("corrections.min_resend_m", 0.015).value)
        self.pose_tol = float(p("corrections.pose_tolerance_rad", 0.05).value)
        self.pose_cfg = CatchPoseConfig.from_dict(declare_section(self, "catch_pose", CatchPoseConfig()))
        if bool(p("enabled", False).value):
            self.get_logger().warn("'enabled: true' in the parameters is ignored: arm with ~/enable after checking the pose")
        if len(self.positions) != len(self.topics):
            raise ValueError("close_positions needs one value per gripper topic")

        self.enabled = False
        self.sched = CatchScheduler(self.cfg)
        self.planner: CatchPosePlanner | None = None
        self.planner_status = "waiting for /robot_description"
        self.joints: dict[str, float] = {}
        self.last_offset = None
        self.switched = None  # (activated, deactivated) to undo on disarm
        self.actions = {}
        self.pubs = []
        try:
            from unix_msgs.msg import GripperCommand  # noqa: PLC0415

            self.GripperCommand = GripperCommand
            self.pubs = [self.create_publisher(GripperCommand, t, 10) for t in self.topics]
        except ImportError:
            self.GripperCommand = None
            self.get_logger().warn("unix_msgs not found: dry run only (catch events are logged, grippers never move)")
        self.event_pub = self.create_publisher(String, "/frisbee/catch_event", 10)
        self.create_subscription(FrisbeeIntercept, "/frisbee/intercept", self._on_intercept, 10, callback_group=self.cb)
        self.create_subscription(Empty, "/frisbee/disable", self._on_disable, 10, callback_group=self.cb)
        self.create_subscription(JointState, "/joint_states", self._on_joints, 10, callback_group=self.cb)
        if self.corr_enabled:
            self.create_subscription(String, "/robot_description", self._on_urdf,
                                     QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL), callback_group=self.cb)
        self.create_service(SetBool, "~/enable", self._on_enable, callback_group=self.cb)
        self.create_timer(0.005, self._tick, callback_group=self.cb)  # 5 ms resolution for the closing time
        problems = self.cfg.calibration_problems()
        self.get_logger().info(
            "frisbee_catch_trigger: DISARMED (dry run)" + (f"; cannot arm yet: {', '.join(problems)}" if problems else "")
        )

    # ------------------------------------------------------------------ catching pose
    def _on_urdf(self, msg: String) -> None:
        if self.planner is not None or self.planner_status.startswith("solving"):
            return
        self.planner_status = "solving"
        threading.Thread(target=self._build_planner, args=(msg.data,), daemon=True).start()

    def _build_planner(self, urdf: str) -> None:
        try:
            from frisbee_interception.catch_pose_cli import urdf_hash  # noqa: PLC0415

            pl = CatchPosePlanner(urdf, self.pose_cfg)
            data = json.loads(Path(self.pose_file).read_text()) if self.pose_file and Path(self.pose_file).is_file() else None
            if data and data.get("urdf_hash") == urdf_hash(urdf) and data.get("config", {}).get("center") == self.pose_cfg.center:
                pl.nominal = {a: np.array(q) for a, q in data["nominal"].items()}
                pl.rot = {a: np.array(r) for a, r in data["rotation"].items()}
                src = f"loaded {self.pose_file}"
            else:
                pl.solve_nominal()
                src = "solved from the URDF" + (" (pose_file missing or for another URDF/center)" if self.pose_file else "")
            self.planner = pl
            self.planner_status = "ready"
            self.get_logger().info(f"catching pose {src}: hands at "
                                   f"{ {a: pl.chains[a].fk(q)[:3, 3].round(3).tolist() for a, q in pl.nominal.items()} }")
        except Exception as e:  # noqa: BLE001
            self.planner_status = f"failed: {e}"
            self.get_logger().error(f"catching pose: {e}")

    def _on_joints(self, msg: JointState) -> None:
        self.joints.update(zip(msg.name, msg.position))

    def _current(self) -> dict[str, np.ndarray] | None:
        if self.planner is None:
            return None
        try:
            return {a: np.array([self.joints[n] for n in names]) for a, names in self.planner.joint_names.items()}
        except KeyError:
            return None

    # ------------------------------------------------------------------ arming
    def _call(self, cli, req, timeout=5.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"{cli.srv_name} not available")
        ev, box = threading.Event(), {}
        fut = cli.call_async(req)
        fut.add_done_callback(lambda f: (box.setdefault("r", f.result()), ev.set()))
        if not ev.wait(timeout):
            raise RuntimeError(f"{cli.srv_name} timed out")
        return box["r"]

    def _switch(self, on, off) -> None:
        from controller_manager_msgs.srv import SwitchController  # noqa: PLC0415

        if not (on or off):
            return
        req = SwitchController.Request(activate_controllers=on, deactivate_controllers=off,
                                       strictness=SwitchController.Request.STRICT, activate_asap=True)
        req.timeout.sec = 5
        res = self._call(self.create_client(SwitchController, "/controller_manager/switch_controller",
                                            callback_group=self.cb), req)
        if not (res and res.ok):
            raise RuntimeError(f"switch_controller failed (activate {on}, deactivate {off})")

    def _prepare_corrections(self) -> None:
        """Arms in the catching pose? Then activate both trajectory controllers (like cartesian_follow)."""
        from control_msgs.action import FollowJointTrajectory  # noqa: PLC0415
        from controller_manager_msgs.srv import ListControllers  # noqa: PLC0415
        from rclpy.action import ActionClient  # noqa: PLC0415

        if self.planner is None:
            raise RuntimeError(f"catching pose not ready ({self.planner_status})")
        q = self._current()
        if q is None:
            raise RuntimeError("no /joint_states for the arm joints")
        off = {a: float(np.abs(q[a] - self.planner.nominal[a]).max()) for a in q}
        if max(off.values()) > self.pose_tol:
            raise RuntimeError("arms are not in the catching pose (max joint error "
                               + ", ".join(f"{a} {v:.2f} rad" for a, v in off.items())
                               + "): move there with cartesian_follow --joints catch_pose_<arm>.csv")
        res = self._call(self.create_client(ListControllers, "/controller_manager/list_controllers",
                                            callback_group=self.cb), ListControllers.Request())
        states = {c.name: c.state for c in res.controller}
        missing = [c for c in TRAJ.values() if c not in states]
        if missing:
            raise RuntimeError(f"not loaded: {missing}")
        on = [c for c in TRAJ.values() if states[c] != "active"]
        off_c = [FORWARD] if states.get(FORWARD) == "active" else []
        self._switch(on, off_c)
        self.switched = (on, off_c)
        for arm, ctrl in TRAJ.items():
            if arm not in self.actions:
                self.actions[arm] = ActionClient(self, FollowJointTrajectory, f"/{ctrl}/follow_joint_trajectory",
                                                 callback_group=self.cb)
            if not self.actions[arm].wait_for_server(timeout_sec=3.0):
                raise RuntimeError(f"{ctrl} action server not found")

    def _restore_controllers(self) -> None:
        if self.switched:
            on, off = self.switched
            try:
                self._switch(off, on)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"could not restore controllers: {e}")
            self.switched = None

    def _on_enable(self, req, res):
        problems = self.cfg.calibration_problems()
        if req.data:
            if problems:
                res.success, res.message = False, "refused: " + ", ".join(problems)
                return res
            if self.GripperCommand is None:
                res.success, res.message = False, "refused: unix_msgs (gripper interface) not available"
                return res
            if self.corr_enabled:
                try:
                    self._prepare_corrections()
                except Exception as e:  # noqa: BLE001
                    self._restore_controllers()
                    res.success, res.message = False, f"refused: {e}"
                    return res
            self.enabled = True
        else:
            self.enabled = False
            self._restore_controllers()
        self.sched.plan = None
        self.last_offset = None
        res.success = True
        res.message = f"catch {'ARMED' if self.enabled else 'disarmed'}" + (
            " (grippers + hand corrections)" if self.enabled and self.corr_enabled else "")
        self.get_logger().warn(res.message)
        return res

    def _on_disable(self, _msg) -> None:
        if self.enabled:
            self.get_logger().warn("catch DISARMED by /frisbee/disable")
        self.enabled = False
        self.sched.plan = None
        self.last_offset = None
        self._restore_controllers()

    # ------------------------------------------------------------------ per throw
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _event(self, kind: str, **kw) -> None:
        self.event_pub.publish(String(data=json.dumps({"event": kind, "armed": self.enabled, **kw})))

    def _on_intercept(self, msg: FrisbeeIntercept) -> None:
        now = self._now()
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        valid = msg.valid and now - stamp <= self.max_data_age and not math.isnan(msg.pixel_u)
        it = Time.from_msg(msg.intercept_time).nanoseconds * 1e-9 if valid else float("nan")
        pixel = (msg.pixel_u, msg.pixel_v) if valid else None
        had = self.sched.plan
        plan = self.sched.update(now, bool(valid), it, pixel, msg.confidence)
        if plan is not None and plan is not had:
            self._event("planned", close_in_s=round(plan.close_at - now, 3), arrival_in_s=round(plan.intercept_time - now, 3),
                        pixel=[round(plan.pixel[0], 1), round(plan.pixel[1], 1)], confidence=plan.confidence)
            has_point = msg.mode == "metric" or any(abs(v) > 1e-9 for v in (msg.catch_point.x, msg.catch_point.y, msg.catch_point.z))
            if self.corr_enabled and has_point and msg.header.frame_id == self.pose_cfg.base:
                self._correct(msg, plan.intercept_time - now)
        elif had is not None and plan is None:
            self._event("cancelled", reason=self.sched.last_reason)

    def _correct(self, msg: FrisbeeIntercept, arrival_in: float) -> None:
        """Shift both hands toward the predicted catch point (in the catching plane)."""
        if self.planner is None:
            return
        c = self.pose_cfg.center
        dy, dz = msg.catch_point.y - c[1], msg.catch_point.z - c[2]
        if self.last_offset is not None and math.hypot(dy - self.last_offset[0], dz - self.last_offset[1]) < self.min_resend:
            return
        q_now = self._current()
        sol = self.planner.correction(dy, dz, q_now)
        if sol is None:
            self._event("correction_failed", offset=[round(dy, 3), round(dz, 3)])
            return
        targets, (cy, cz) = sol
        base = q_now or self.planner.nominal
        dq = max(float(np.abs(targets[a] - base[a]).max()) for a in targets)
        # cubic with zero end speeds peaks at 1.5x its mean speed; 90% of the cap (same rule as cartesian_follow)
        duration = max(0.05, 1.5 * dq / (0.9 * self.max_joint_vel))
        info = dict(offset=[round(cy, 3), round(cz, 3)], clamped=bool((cy, cz) != (dy, dz)), max_joint_change=round(dq, 3),
                    duration_s=round(duration, 3), arrival_in_s=round(arrival_in, 3),
                    in_time=bool(duration <= arrival_in))
        self.last_offset = (dy, dz)
        if not self.enabled:
            self._event("would_correct", **info)
            return
        from builtin_interfaces.msg import Duration  # noqa: PLC0415
        from control_msgs.action import FollowJointTrajectory  # noqa: PLC0415
        from trajectory_msgs.msg import JointTrajectoryPoint  # noqa: PLC0415

        for arm, q in targets.items():
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = self.planner.joint_names[arm]
            goal.trajectory.points.append(JointTrajectoryPoint(
                positions=q.tolist(), velocities=[0.0] * len(q),
                time_from_start=Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))))
            self.actions[arm].send_goal_async(goal)  # a new goal replaces the previous trajectory
        self._event("correction", **info)

    def _tick(self) -> None:
        now = self._now()
        had = self.sched.plan
        fired = self.sched.due(now)
        if fired is None:
            if had is not None and self.sched.plan is None:
                self._event("missed", reason=self.sched.last_reason)
            return
        self.last_offset = None
        info = dict(late_s=round(now - fired.close_at, 3), arrival_in_s=round(fired.intercept_time - now, 3),
                    pixel=[round(fired.pixel[0], 1), round(fired.pixel[1], 1)])
        if not self.enabled:
            self._event("would_fire", **info)
            return
        for pub, pos in zip(self.pubs, self.positions):
            cmd = self.GripperCommand()
            cmd.mode = "parallel"
            cmd.command.name = [""]
            cmd.command.position = [pos]
            cmd.command.velocity = [self.velocity]
            cmd.command.effort = [self.effort]
            pub.publish(cmd)
        self._event("fired", **info)
        self.get_logger().warn(f"CATCH: grippers closing ({json.dumps(info)})")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CatchTriggerNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.enabled = False
        node._restore_controllers()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
