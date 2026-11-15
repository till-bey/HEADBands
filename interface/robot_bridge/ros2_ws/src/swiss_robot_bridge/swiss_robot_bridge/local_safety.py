"""Cloud-independent local ROS 2 safety supervisor, watchdog, E-Stop, and UniX AI Panther actuation planner."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from swiss_robot_bridge.panther_topics import (
    ARM_JOINT_LIMITS,
    GESTURE_MAX_DURATION_S,
    MAX_ARM_JOINT_VEL_RPS,
    GRIPPER_CLOSED_POS,
    GRIPPER_DEFAULT_VEL,
    GRIPPER_MODE_PARALLEL,
    GRIPPER_OPEN_POS,
    HEAD_PITCH_MAX_RAD,
    HEAD_PITCH_MIN_RAD,
    HEAD_YAW_MAX_RAD,
    HEAD_YAW_MIN_RAD,
    LIFT_MAX_M,
    LIFT_MAX_VEL_MPS,
    LIFT_MIN_M,
    RETARGETED_FRISBEE_THROW_WAYPOINTS,
    RIGHT_ARM_JOINTS,
)

ALLOWED_LOCAL_ACTIONS = frozenset(
    {
        "move_to",
        "rotate",
        "look_at",
        "inspect_object",
        "pick_object",
        "place_object",
        "throw_object",
        "generated_gesture",
        "return_home",
        "stop",
    }
)

MOTION_ACTIONS_REQUIRING_HUMAN_CONFIRMATION = frozenset(
    {"move_to", "rotate", "pick_object", "place_object", "throw_object", "generated_gesture", "return_home"}
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _shortest_signed_angle_diff(current_rad: float, start_rad: float) -> float:
    diff = current_rad - start_rad
    return (diff + math.pi) % (2.0 * math.pi) - math.pi


class LocalSafetyRejectedError(ValueError):
    """Raised on the robot computer when any incoming command violates local safety bounds."""


def check_gesture_locally(
    gesture: LocalGestureTrajectory,
) -> tuple[tuple[str, ...], list[tuple[float, tuple[float, ...], float | None]]]:
    """Re-check a cloud-planned gesture against local limits; return (joints, points) for playback."""
    names = tuple(gesture.joint_names)
    if not names or len(set(names)) != len(names) or any(n not in ARM_JOINT_LIMITS for n in names):
        raise LocalSafetyRejectedError(f"Gesture joints not allowed locally: {list(names)}")
    if not gesture.points or len(gesture.points) > 600:
        raise LocalSafetyRejectedError("Gesture must have 1 to 600 points.")
    out: list[tuple[float, tuple[float, ...], float | None]] = []
    prev_t, prev_q = 0.0, None
    for i, pt in enumerate(gesture.points):
        if len(pt.positions) != len(names) or not math.isfinite(pt.t) or pt.t <= prev_t:
            raise LocalSafetyRejectedError(f"Gesture point {i} malformed or not increasing in time.")
        q = tuple(float(v) for v in pt.positions)
        for name, v in zip(names, q):
            lo, hi = ARM_JOINT_LIMITS[name]
            if not math.isfinite(v) or v < lo - 1e-3 or v > hi + 1e-3:
                raise LocalSafetyRejectedError(f"Gesture point {i}: {name}={v:.3f} outside [{lo}, {hi}].")
        if prev_q is not None:
            speed = max(abs(a - b) for a, b in zip(q, prev_q)) / (pt.t - prev_t)
            if speed > MAX_ARM_JOINT_VEL_RPS + 1e-6:
                raise LocalSafetyRejectedError(
                    f"Gesture joint speed {speed:.2f} rad/s exceeds {MAX_ARM_JOINT_VEL_RPS} rad/s locally."
                )
        out.append((round(pt.t, 4), q, None))
        prev_t, prev_q = pt.t, q
    if prev_t > GESTURE_MAX_DURATION_S:
        raise LocalSafetyRejectedError(f"Gesture lasts {prev_t:.1f} s, over {GESTURE_MAX_DURATION_S:.0f} s.")
    return names, out


def retime_from_current_pose(
    names: tuple[str, ...],
    points: list[tuple[float, tuple[float, ...], float | None]],
    current: dict[str, float],
    max_vel: float = MAX_ARM_JOINT_VEL_RPS * 0.9,
) -> list[tuple[float, tuple[float, ...], float | None]]:
    """Delay the trajectory if the arm's actual pose is too far from the first point for max_vel."""
    if not points or not all(n in current for n in names):
        return points
    gap = max(abs(points[0][1][i] - float(current[n])) for i, n in enumerate(names))
    needed = gap / max_vel
    shift = max(0.0, needed - points[0][0])
    if shift <= 1e-3:
        return points
    return [(round(t + shift, 4), q, g) for t, q, g in points]


class LocalTarget3D(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    z: float = 0.0


class LocalGesturePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    positions: list[float]


class LocalGestureTrajectory(BaseModel):
    """Planned arm trajectory from the cloud planner. Only joint_names and points are used here;
    descriptive fields (name, keyframes, report) are ignored and everything is re-checked locally."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    joint_names: list[str]
    points: list[LocalGesturePoint]


class IncomingCloudCommand(BaseModel):
    """Strict local schema — rejects unknown keys, raw ROS topics, or code payloads."""

    model_config = ConfigDict(extra="forbid")

    gesture: LocalGestureTrajectory | None = None

    command_id: str
    action: str
    target: LocalTarget3D | None = None
    yaw_rad: float | None = None
    object_label: str | None = None
    speed: float = 0.0
    requires_confirmation: bool = True
    confirmed_by_human: bool = False
    confirmed_at: datetime | None = None
    created_at: datetime | None = None
    expires_at: datetime
    rationale: str = ""


@dataclass
class TwistSetpoint:
    """Safe bounded ROS 2 velocity setpoint (`geometry_msgs/Twist` on `/cmd_vel`)."""

    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0

    def is_nonzero(self) -> bool:
        return abs(self.linear_x) > 1e-5 or abs(self.linear_y) > 1e-5 or abs(self.angular_z) > 1e-5


@dataclass
class PantherActuationPlan:
    """Validated multi-controller actuation plan mapped to real UniX AI Panther ROS 2 topics."""

    command_id: str
    action: str
    twist: TwistSetpoint = field(default_factory=TwistSetpoint)
    twist_duration_seconds: float = 0.0
    target_distance_m: float = 0.0
    target_yaw_rad: float | None = None
    head_positions: tuple[float, float] | None = None  # /head_forward_position_controller/commands
    lift_position_m: float | None = None  # /lifting_controller/commands [0.0, 0.80]
    lift_velocity_mps: float = LIFT_MAX_VEL_MPS
    gripper_side: str | None = None  # "left" or "right"
    gripper_mode: str = GRIPPER_MODE_PARALLEL
    gripper_position: float | None = None  # 0.0 open .. 1.0 closed
    gripper_velocity: float = GRIPPER_DEFAULT_VEL
    master_trajectory_joints: tuple[str, ...] | None = None  # /master_controller/joint_trajectory
    master_trajectory_points: list[tuple[float, tuple[float, ...], float | None]] = field(
        default_factory=list
    )
    switch_arm_trajectory_controllers: bool = False
    trajectory_duration_seconds: float = 0.0  # arm trajectory length; holds queued steps until it ends
    trigger_camera: bool = False
    preempt_teleop: bool = False
    enable_ultrasonics: bool = False


@dataclass
class LocalSafetySupervisor:
    """Enforces local safety independently of cloud connectivity or LLM availability."""

    max_linear_speed_mps: float = 0.5
    max_angular_speed_rps: float = 1.0
    max_workspace_radius_m: float = 10.0
    watchdog_timeout_seconds: float = 2.0
    min_battery_percent: float = 10.0
    min_front_obstacle_distance_m: float = 0.35
    head_order: str = "pitch_yaw"  # URDF default: joint_head0=pitch, joint_head1=yaw

    local_estop_latched: bool = False
    local_fault_active: bool = False
    battery_percent: float = 95.0
    front_obstacle_distance_m: float | None = None
    last_cloud_heartbeat_monotonic: float | None = None
    active_command_id: str | None = None
    motion_deadline_monotonic: float | None = None
    current_twist: TwistSetpoint = field(default_factory=TwistSetpoint)
    pending_plans: list[PantherActuationPlan] = field(default_factory=list)
    odom_pose_xy_yaw: tuple[float, float, float] | None = None
    last_odom_monotonic: float | None = None

    _nominal_twist: TwistSetpoint = field(default_factory=TwistSetpoint)
    _motion_start_odom: tuple[float, float, float] | None = None
    _active_target_distance_m: float = 0.0
    _active_target_yaw_rad: float | None = None
    _promoted_plan: PantherActuationPlan | None = None

    def _clear_active_motion(self, *, clear_queue: bool = False) -> TwistSetpoint:
        self.active_command_id = None
        self.motion_deadline_monotonic = None
        self._nominal_twist = TwistSetpoint(0.0, 0.0, 0.0)
        self._motion_start_odom = None
        self._active_target_distance_m = 0.0
        self._active_target_yaw_rad = None
        self.current_twist = TwistSetpoint(0.0, 0.0, 0.0)
        if clear_queue:
            self.pending_plans.clear()
            self._promoted_plan = None
        return self.current_twist

    def update_odometry(self, x: float, y: float, yaw_rad: float, monotonic_now: float) -> None:
        if all(math.isfinite(v) for v in (x, y, yaw_rad)):
            self.odom_pose_xy_yaw = (float(x), float(y), float(yaw_rad))
            self.last_odom_monotonic = monotonic_now

    def _has_fresh_odom(self, monotonic_now: float | None) -> bool:
        if monotonic_now is None or self.odom_pose_xy_yaw is None or self.last_odom_monotonic is None:
            return False
        return (monotonic_now - self.last_odom_monotonic) <= 1.0

    def _activate_plan(self, plan: PantherActuationPlan, monotonic_now: float | None) -> None:
        self.active_command_id = plan.command_id
        self._nominal_twist = TwistSetpoint(
            plan.twist.linear_x,
            plan.twist.linear_y,
            plan.twist.angular_z,
        )
        self.current_twist = TwistSetpoint(
            plan.twist.linear_x,
            plan.twist.linear_y,
            plan.twist.angular_z,
        )
        self._active_target_distance_m = plan.target_distance_m
        self._active_target_yaw_rad = plan.target_yaw_rad
        if monotonic_now is not None and plan.twist_duration_seconds > 0.0 and plan.twist.is_nonzero():
            if self._has_fresh_odom(monotonic_now):
                self._motion_start_odom = self.odom_pose_xy_yaw
                # Allow 35% extra wall-clock budget for hardware ramp-up/deceleration while /odom closes the loop
                self.motion_deadline_monotonic = monotonic_now + plan.twist_duration_seconds * 1.35
            else:
                self._motion_start_odom = None
                self.motion_deadline_monotonic = monotonic_now + plan.twist_duration_seconds
        elif monotonic_now is not None and plan.trajectory_duration_seconds > 0.0:
            # Arm-only trajectory: no wheel motion, but the next queued step waits for it to finish.
            self._motion_start_odom = None
            self.motion_deadline_monotonic = monotonic_now + plan.trajectory_duration_seconds + 0.5
        else:
            self._motion_start_odom = None
            self.motion_deadline_monotonic = None

    def pop_promoted_plan(self) -> PantherActuationPlan | None:
        """Return and clear a queued step that was promoted into execution on step completion."""
        plan = self._promoted_plan
        self._promoted_plan = None
        return plan

    def trigger_local_estop(self) -> TwistSetpoint:
        """Rule 9: Immediate local E-Stop — zero network dependency."""
        self.local_estop_latched = True
        return self._clear_active_motion(clear_queue=True)

    def reset_local_estop(self) -> None:
        """Reset local E-Stop only when local fault is also clear."""
        if self.local_fault_active:
            raise LocalSafetyRejectedError("Cannot reset E-Stop while local hardware fault is active.")
        self.local_estop_latched = False

    def record_cloud_heartbeat(self, monotonic_now: float) -> None:
        self.last_cloud_heartbeat_monotonic = monotonic_now

    def _promote_next_queued_plan_or_stop(self, monotonic_now: float) -> TwistSetpoint:
        while self.pending_plans:
            next_plan = self.pending_plans.pop(0)
            if (
                next_plan.twist.linear_x > 0.0
                and self.front_obstacle_distance_m is not None
                and self.front_obstacle_distance_m < self.min_front_obstacle_distance_m
            ):
                return self._clear_active_motion(clear_queue=True)
            self._activate_plan(next_plan, monotonic_now)
            self._promoted_plan = next_plan
            return self.current_twist
        return self._clear_active_motion(clear_queue=False)

    def evaluate_watchdog(self, monotonic_now: float) -> TwistSetpoint:
        """Rule 8: Zero `/cmd_vel` immediately if E-Stop/fault/obstacle triggers, motion finishes, or cloud times out."""
        if self.local_estop_latched or self.local_fault_active:
            return self._clear_active_motion(clear_queue=True)

        if self.last_cloud_heartbeat_monotonic is None:
            return self._clear_active_motion(clear_queue=True)

        if (monotonic_now - self.last_cloud_heartbeat_monotonic) > self.watchdog_timeout_seconds:
            return self._clear_active_motion(clear_queue=True)

        # Front LiDAR (/scan) reflex stop when moving forward into a close obstacle
        if (
            self.current_twist.linear_x > 0.0
            and self.front_obstacle_distance_m is not None
            and self.front_obstacle_distance_m < self.min_front_obstacle_distance_m
        ):
            return self._clear_active_motion(clear_queue=True)

        # Closed-loop /odom distance & yaw precision tracking when live ROS 2 /odom is streaming
        if (
            self._motion_start_odom is not None
            and self._has_fresh_odom(monotonic_now)
            and self.odom_pose_xy_yaw is not None
        ):
            sx, sy, syaw = self._motion_start_odom
            cx, cy, cyaw = self.odom_pose_xy_yaw
            if self._active_target_distance_m > 0.0:
                traveled = math.hypot(cx - sx, cy - sy)
                remaining = self._active_target_distance_m - traveled
                if remaining <= 0.01:
                    return self._promote_next_queued_plan_or_stop(monotonic_now)
                if remaining < 0.20:
                    scale = max(0.25, remaining / 0.20)
                    self.current_twist = TwistSetpoint(
                        round(self._nominal_twist.linear_x * scale, 4),
                        round(self._nominal_twist.linear_y * scale, 4),
                        0.0,
                    )
            elif self._active_target_yaw_rad is not None and abs(self._active_target_yaw_rad) > 1e-4:
                turned = abs(_shortest_signed_angle_diff(cyaw, syaw))
                remaining_yaw = abs(self._active_target_yaw_rad) - turned
                if remaining_yaw <= 0.015:
                    return self._promote_next_queued_plan_or_stop(monotonic_now)
                if remaining_yaw < 0.18:
                    scale = max(0.25, remaining_yaw / 0.18)
                    self.current_twist = TwistSetpoint(
                        0.0,
                        0.0,
                        round(self._nominal_twist.angular_z * scale, 4),
                    )

        # Auto-zero latched /cmd_vel once timed distance/rotation duration completes (or promote next queued step)
        if self.motion_deadline_monotonic is not None and monotonic_now >= self.motion_deadline_monotonic:
            return self._promote_next_queued_plan_or_stop(monotonic_now)

        return self.current_twist

    def _compute_head_joints(self, cmd: IncomingCloudCommand) -> tuple[float, float]:
        """Compute bounded Panther head joint angles `(joint_head0, joint_head1)`."""
        yaw = 0.0
        pitch = 0.20  # gentle downward look toward workspace/floor by default
        if cmd.yaw_rad is not None and math.isfinite(cmd.yaw_rad):
            yaw = _clamp(cmd.yaw_rad, HEAD_YAW_MIN_RAD, HEAD_YAW_MAX_RAD)
        if cmd.target is not None:
            horiz = max(math.hypot(cmd.target.x, cmd.target.y), 0.25)
            if abs(cmd.target.x) > 1e-3 or abs(cmd.target.y) > 1e-3:
                yaw = _clamp(math.atan2(cmd.target.y, max(cmd.target.x, 0.15)), HEAD_YAW_MIN_RAD, HEAD_YAW_MAX_RAD)
            # Panther camera is ~1.1m above floor; positive pitch tilts head downward
            approx_cam_height_m = 1.05
            pitch = _clamp(
                math.atan2(approx_cam_height_m - cmd.target.z, horiz) * 0.45,
                HEAD_PITCH_MIN_RAD,
                HEAD_PITCH_MAX_RAD,
            )
        pitch = round(_clamp(pitch, HEAD_PITCH_MIN_RAD, HEAD_PITCH_MAX_RAD), 4)
        yaw = round(_clamp(yaw, HEAD_YAW_MIN_RAD, HEAD_YAW_MAX_RAD), 4)
        if self.head_order == "yaw_pitch":
            return (yaw, pitch)
        return (pitch, yaw)

    @staticmethod
    def _select_gripper_side(cmd: IncomingCloudCommand) -> str:
        combined = f"{cmd.object_label or ''} {cmd.rationale or ''}".lower()
        if any(tok in combined for tok in ("right", "rechts", "rächts", "droite", "derecha", "prav", "右")):
            return "right"
        if cmd.target is not None and cmd.target.y < -0.08:
            return "right"
        return "left"

    def validate_and_compute_panther_plan(
        self,
        raw_payload: dict[str, Any],
        now_utc: datetime | None = None,
        monotonic_now: float | None = None,
    ) -> PantherActuationPlan:
        """Re-validate every parameter locally and build a bounded UniX AI Panther actuation plan."""
        if "ros_topic" in raw_payload or "code" in raw_payload or "shell" in raw_payload:
            raise LocalSafetyRejectedError("Direct ROS topic or code execution rejected locally.")

        try:
            cmd = IncomingCloudCommand.model_validate(raw_payload)
        except ValidationError as exc:
            raise LocalSafetyRejectedError(f"Malformed command schema: {exc}") from exc

        if cmd.action not in ALLOWED_LOCAL_ACTIONS:
            raise LocalSafetyRejectedError(f"Action '{cmd.action}' not in local allowlist.")

        if cmd.action == "stop":
            self._clear_active_motion(clear_queue=True)
            return PantherActuationPlan(
                command_id=cmd.command_id,
                action="stop",
                twist=self.current_twist,
                preempt_teleop=True,
            )

        if self.local_estop_latched:
            raise LocalSafetyRejectedError("Local E-Stop is active; motion rejected.")
        if self.local_fault_active:
            raise LocalSafetyRejectedError("Local robot fault active; motion rejected.")
        if self.battery_percent < self.min_battery_percent:
            raise LocalSafetyRejectedError("Battery below safe local threshold.")

        ref_now = now_utc or datetime.now(tz=UTC)
        expires = cmd.expires_at if cmd.expires_at.tzinfo else cmd.expires_at.replace(tzinfo=UTC)
        if ref_now >= expires:
            raise LocalSafetyRejectedError(f"Command '{cmd.command_id}' expired before local execution.")

        if cmd.action in MOTION_ACTIONS_REQUIRING_HUMAN_CONFIRMATION and not cmd.confirmed_by_human:
            raise LocalSafetyRejectedError(
                f"Motion action '{cmd.action}' lacks mandatory confirmed_by_human=True."
            )

        if not math.isfinite(cmd.speed) or cmd.speed < 0.0 or cmd.speed > self.max_linear_speed_mps:
            raise LocalSafetyRejectedError(
                f"Speed {cmd.speed} m/s outside local limit [0, {self.max_linear_speed_mps}]."
            )

        if cmd.target is not None:
            if not all(math.isfinite(v) for v in (cmd.target.x, cmd.target.y, cmd.target.z)):
                raise LocalSafetyRejectedError("Non-finite target coordinates rejected.")
            radius = math.hypot(cmd.target.x, cmd.target.y)
            if radius > self.max_workspace_radius_m or cmd.target.z < -0.05 or cmd.target.z > 2.0:
                raise LocalSafetyRejectedError("Target outside local safe workspace envelope.")

        plan = PantherActuationPlan(command_id=cmd.command_id, action=cmd.action)
        twist = TwistSetpoint(0.0, 0.0, 0.0)

        if cmd.action in {"move_to", "return_home"} and cmd.target is not None:
            if (
                cmd.target.x > 0.0
                and self.front_obstacle_distance_m is not None
                and self.front_obstacle_distance_m < self.min_front_obstacle_distance_m
            ):
                raise LocalSafetyRejectedError(
                    f"Front LiDAR obstacle at {self.front_obstacle_distance_m:.2f}m "
                    f"(< {self.min_front_obstacle_distance_m:.2f}m); forward motion blocked."
                )
            dist = math.hypot(cmd.target.x, cmd.target.y)
            if dist > 1e-3:
                bounded_speed = min(cmd.speed, self.max_linear_speed_mps)
                if bounded_speed > 1e-4:
                    twist.linear_x = round((cmd.target.x / dist) * bounded_speed, 4)
                    twist.linear_y = round((cmd.target.y / dist) * bounded_speed, 4)
                    plan.twist_duration_seconds = round(dist / bounded_speed, 3)
                    plan.target_distance_m = round(dist, 4)
                    plan.enable_ultrasonics = True
            if cmd.target.z > 0.01:
                plan.lift_position_m = round(_clamp(cmd.target.z, LIFT_MIN_M, LIFT_MAX_M), 4)
                plan.lift_velocity_mps = LIFT_MAX_VEL_MPS

        elif cmd.action == "rotate":
            if cmd.yaw_rad is None or not math.isfinite(cmd.yaw_rad) or abs(cmd.yaw_rad) > math.pi:
                raise LocalSafetyRejectedError("yaw_rad out of local [-pi, pi] bounds.")
            if abs(cmd.yaw_rad) > 1e-4:
                sign = 1.0 if cmd.yaw_rad >= 0 else -1.0
                ang_speed = min(0.4, self.max_angular_speed_rps)
                twist.angular_z = round(sign * ang_speed, 4)
                plan.twist_duration_seconds = round(abs(cmd.yaw_rad) / ang_speed, 3)
                plan.target_yaw_rad = round(cmd.yaw_rad, 5)
                plan.enable_ultrasonics = True

        elif cmd.action in {"look_at", "inspect_object"}:
            plan.head_positions = self._compute_head_joints(cmd)
            plan.trigger_camera = True

        elif cmd.action == "pick_object":
            plan.gripper_side = self._select_gripper_side(cmd)
            plan.gripper_position = GRIPPER_CLOSED_POS
            plan.gripper_velocity = GRIPPER_DEFAULT_VEL
            plan.head_positions = self._compute_head_joints(cmd)
            plan.trigger_camera = True
            if cmd.target is not None and cmd.target.z > 0.01:
                plan.lift_position_m = round(_clamp(cmd.target.z, LIFT_MIN_M, LIFT_MAX_M), 4)

        elif cmd.action == "place_object":
            plan.gripper_side = self._select_gripper_side(cmd)
            plan.gripper_position = GRIPPER_OPEN_POS
            plan.gripper_velocity = GRIPPER_DEFAULT_VEL
            plan.head_positions = self._compute_head_joints(cmd)
            plan.trigger_camera = True
            if cmd.target is not None and cmd.target.z > 0.01:
                plan.lift_position_m = round(_clamp(cmd.target.z, LIFT_MIN_M, LIFT_MAX_M), 4)

        elif cmd.action == "throw_object":
            plan.gripper_side = "right"
            plan.gripper_position = GRIPPER_OPEN_POS
            plan.gripper_velocity = 1.0
            plan.head_positions = self._compute_head_joints(cmd)
            plan.master_trajectory_joints = RIGHT_ARM_JOINTS + ("joint_rf",)
            plan.master_trajectory_points = [
                (t_s, q_tuple, grip_val) for t_s, q_tuple, grip_val in RETARGETED_FRISBEE_THROW_WAYPOINTS
            ]
            plan.switch_arm_trajectory_controllers = True
            plan.trigger_camera = True
            if cmd.target is not None and cmd.target.z > 0.01:
                plan.lift_position_m = round(_clamp(cmd.target.z, LIFT_MIN_M, LIFT_MAX_M), 4)

        elif cmd.action == "generated_gesture":
            if cmd.gesture is None:
                raise LocalSafetyRejectedError("generated_gesture arrived without a trajectory.")
            plan.master_trajectory_joints, plan.master_trajectory_points = check_gesture_locally(cmd.gesture)
            plan.trajectory_duration_seconds = plan.master_trajectory_points[-1][0]
            plan.switch_arm_trajectory_controllers = True

        plan.twist = twist
        self._activate_plan(plan, monotonic_now)
        return plan

    def submit_cloud_command(
        self,
        raw_payload: dict[str, Any],
        now_utc: datetime | None = None,
        monotonic_now: float | None = None,
    ) -> tuple[PantherActuationPlan, bool]:
        """Validate a confirmed cloud command; execute immediately or queue sequentially if base motion is busy.

        Returns `(plan, dispatched_immediately)`. `stop` always preempts and flushes the queue immediately.
        """
        action = str(raw_payload.get("action", ""))
        if (
            action != "stop"
            and self.active_command_id is not None
            and self.motion_deadline_monotonic is not None
            and monotonic_now is not None
            and monotonic_now < self.motion_deadline_monotonic
        ):
            # Snapshot current active motion state so we can validate + build `plan` without interrupting Step 1
            saved_cmd_id = self.active_command_id
            saved_deadline = self.motion_deadline_monotonic
            saved_twist = TwistSetpoint(
                self.current_twist.linear_x,
                self.current_twist.linear_y,
                self.current_twist.angular_z,
            )
            saved_nominal = TwistSetpoint(
                self._nominal_twist.linear_x,
                self._nominal_twist.linear_y,
                self._nominal_twist.angular_z,
            )
            saved_start_odom = self._motion_start_odom
            saved_dist = self._active_target_distance_m
            saved_yaw = self._active_target_yaw_rad

            plan = self.validate_and_compute_panther_plan(
                raw_payload=raw_payload,
                now_utc=now_utc,
                monotonic_now=None,
            )
            # Restore active Step 1 motion and enqueue Step 2 sequentially
            self.active_command_id = saved_cmd_id
            self.motion_deadline_monotonic = saved_deadline
            self.current_twist = saved_twist
            self._nominal_twist = saved_nominal
            self._motion_start_odom = saved_start_odom
            self._active_target_distance_m = saved_dist
            self._active_target_yaw_rad = saved_yaw
            self.pending_plans.append(plan)
            return plan, False

        plan = self.validate_and_compute_panther_plan(
            raw_payload=raw_payload,
            now_utc=now_utc,
            monotonic_now=monotonic_now,
        )
        return plan, True

    def validate_and_compute_setpoint(
        self,
        raw_payload: dict[str, Any],
        now_utc: datetime | None = None,
    ) -> TwistSetpoint:
        """Backward-compatible helper returning the `/cmd_vel` TwistSetpoint."""
        plan = self.validate_and_compute_panther_plan(raw_payload=raw_payload, now_utc=now_utc)
        return plan.twist
