"""Deterministic Robot Safety Validator enforcing all 10 mandatory safety rules."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import ValidationError

from app.core.config import Settings
from app.robotics.action_schema import (
    DANGEROUS_ACTIONS_REQUIRING_CONFIRMATION,
    AllowedActionType,
    GestureTrajectory,
    RobotActionProposal,
    RobotOperationalState,
    RobotStatus,
    ValidatedRobotCommand,
)
from app.robotics.gesture_planner import MAX_DURATION_S as GESTURE_MAX_DURATION_S
from app.robotics.gesture_planner import joint_limits as gesture_joint_limits

# Hard cap re-checked here and on the robot bridge; the planner targets 0.45 rad/s.
GESTURE_MAX_JOINT_VEL_RPS = 0.5

# Rule 1 & 2: Patterns indicative of code injection or direct ROS topic injection
_FORBIDDEN_CODE_OR_TOPIC_RE = re.compile(
    r"(;|&&|\|\||`|\$\(|__import__|eval\s*\(|exec\s*\(|subprocess|os\.system"
    r"|/cmd_vel|/preempt_teleop|/master_controller/joint_trajectory|/dual_arm_forward_position_controller"
    r"|/head_forward_position_controller|/lifting_controller|/cartesian_follow/planned_path"
    r"|/left_gripper|/right_gripper|/joint_trajectory|/camera/color/image_raw|/can0/|/can1/|ros2\s+topic)",
    re.IGNORECASE,
)


class SafetyValidationError(ValueError):
    """Raised whenever a proposed robot action violates any mandatory safety rule."""


class ActionSafetyValidator:
    """Validates high-level action proposals before queuing or sending to the ROS 2 bridge."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def parse_and_validate_proposal(
        self,
        raw_proposal: dict[str, Any] | RobotActionProposal,
        robot_status: RobotStatus,
        require_active_bridge_watchdog: bool = False,
        gesture: GestureTrajectory | None = None,
    ) -> ValidatedRobotCommand:
        """Validate proposal structure, bounds, robot safety state, watchdog, and confirmation flag."""
        # Rule 2 & 3: Reject unknown fields (such as raw 'ros_topic' or 'script') & unknown action names
        if isinstance(raw_proposal, dict):
            if "ros_topic" in raw_proposal or "code" in raw_proposal or "shell" in raw_proposal:
                raise SafetyValidationError("Direct ROS topic or code execution fields are strictly forbidden.")
            try:
                proposal = RobotActionProposal.model_validate(raw_proposal)
            except ValidationError as exc:
                raise SafetyValidationError(f"Invalid robot action schema: {exc}") from exc
        else:
            proposal = raw_proposal

        # Rule 1: Never allow code-like payloads inside string fields
        for text_field in (proposal.object_label or "", proposal.rationale or ""):
            if _FORBIDDEN_CODE_OR_TOPIC_RE.search(text_field):
                raise SafetyValidationError("Forbidden code or ROS topic pattern detected in proposal fields.")

        #STOP action is always permitted unless malformed, even to halt motion
        if proposal.action != AllowedActionType.STOP:
            self.assert_robot_state_safe(
                robot_status=robot_status,
                require_active_watchdog=require_active_bridge_watchdog,
            )

        # Rule 4 & 5: Validate finite coordinates, workspace envelope, and speed limits
        if proposal.action == AllowedActionType.MOVE_TO and proposal.target is None:
            raise SafetyValidationError("Action 'move_to' requires an explicit 3D target.")

        if proposal.action == AllowedActionType.ROTATE and proposal.yaw_rad is None:
            raise SafetyValidationError("Action 'rotate' requires 'yaw_rad'.")

        if proposal.action in {
            AllowedActionType.PICK_OBJECT,
            AllowedActionType.INSPECT_OBJECT,
            AllowedActionType.THROW_OBJECT,
        }:
            if not proposal.object_label and proposal.target is None:
                raise SafetyValidationError(f"Action '{proposal.action}' requires 'object_label' or 'target'.")

        if proposal.action == AllowedActionType.GENERATED_GESTURE:
            if not proposal.object_label:
                raise SafetyValidationError("Action 'generated_gesture' requires a gesture description in 'object_label'.")
            if gesture is None:
                raise SafetyValidationError("Action 'generated_gesture' has no planned trajectory.")
            self.assert_gesture_trajectory_safe(gesture)
        elif gesture is not None:
            raise SafetyValidationError("A gesture trajectory is only allowed on 'generated_gesture'.")

        if proposal.target is not None:
            coords = (proposal.target.x, proposal.target.y, proposal.target.z)
            if not all(math.isfinite(c) for c in coords):
                raise SafetyValidationError("Target coordinates must be finite numbers.")
            if proposal.target.z < -0.05 or proposal.target.z > 2.0:
                raise SafetyValidationError(
                    f"Target z={proposal.target.z:.2f}m outside safe vertical envelope [-0.05m, 2.00m]."
                )
            radial_xy = math.hypot(proposal.target.x, proposal.target.y)
            if radial_xy > self._settings.robot_max_workspace_radius_m:
                raise SafetyValidationError(
                    f"Target radial distance {radial_xy:.2f}m exceeds safe workspace radius "
                    f"{self._settings.robot_max_workspace_radius_m:.2f}m."
                )

        validated_yaw = proposal.yaw_rad
        if validated_yaw is not None:
            if not math.isfinite(validated_yaw) or abs(validated_yaw) > math.pi + 1e-3:
                raise SafetyValidationError("yaw_rad must be finite and within [-pi, +pi] radians.")
            validated_yaw = max(-math.pi, min(math.pi, validated_yaw))

        if not math.isfinite(proposal.speed) or proposal.speed < 0.0:
            raise SafetyValidationError("Speed must be a non-negative finite float.")
        if proposal.speed > self._settings.robot_max_linear_speed_mps:
            raise SafetyValidationError(
                f"Requested speed {proposal.speed:.2f} m/s exceeds limit "
                f"{self._settings.robot_max_linear_speed_mps:.2f} m/s."
            )

        # Rule 6: Enforce human confirmation on all physical motion / manipulation actions
        must_confirm = (
            proposal.requires_confirmation
            or (
                self._settings.robot_require_confirmation
                and proposal.action in DANGEROUS_ACTIONS_REQUIRING_CONFIRMATION
            )
        )
        if proposal.action == AllowedActionType.STOP:
            must_confirm = False

        now = datetime.now(tz=UTC)
        expires_at = now + timedelta(seconds=self._settings.robot_command_ttl_seconds)

        return ValidatedRobotCommand(
            command_id=proposal.command_id,
            action=proposal.action,
            target=proposal.target,
            yaw_rad=validated_yaw,
            object_label=proposal.object_label,
            speed=0.0 if proposal.action == AllowedActionType.STOP else proposal.speed,
            requires_confirmation=must_confirm,
            confirmed_by_human=not must_confirm,
            created_at=now,
            expires_at=expires_at,
            rationale=proposal.rationale,
            gesture=gesture,
        )

    def assert_gesture_trajectory_safe(self, gesture: GestureTrajectory) -> None:
        """Independent re-check of a planned gesture: joints, limits, timing and joint speed."""
        limits = gesture_joint_limits()
        names = gesture.joint_names
        if not names or len(set(names)) != len(names):
            raise SafetyValidationError("Gesture joint names are empty or repeated.")
        unknown = [n for n in names if n not in limits]
        if unknown:
            raise SafetyValidationError(f"Gesture names joints outside the arms: {unknown}.")
        if not gesture.points:
            raise SafetyValidationError("Gesture trajectory has no points.")
        if len(gesture.start_positions) not in (0, len(names)):
            raise SafetyValidationError("Gesture start pose does not match its joints.")
        prev_t = 0.0
        prev_q = gesture.start_positions or None
        peak = 0.0
        for i, pt in enumerate(gesture.points):
            if len(pt.positions) != len(names):
                raise SafetyValidationError(f"Gesture point {i} has {len(pt.positions)} values for {len(names)} joints.")
            if not math.isfinite(pt.t) or pt.t <= prev_t:
                raise SafetyValidationError(f"Gesture point {i} time must be finite and increasing.")
            for name, v in zip(names, pt.positions, strict=True):
                lo, hi = limits[name]
                if not math.isfinite(v) or v < lo - 1e-3 or v > hi + 1e-3:
                    raise SafetyValidationError(f"Gesture point {i}: {name}={v:.3f} rad outside [{lo:.2f}, {hi:.2f}].")
            if prev_q is not None:
                dt = pt.t - prev_t
                peak = max(peak, max(abs(a - b) for a, b in zip(pt.positions, prev_q, strict=True)) / dt)
            prev_t, prev_q = pt.t, pt.positions
        if prev_t > GESTURE_MAX_DURATION_S:
            raise SafetyValidationError(f"Gesture lasts {prev_t:.1f} s, over {GESTURE_MAX_DURATION_S:.0f} s.")
        if peak > GESTURE_MAX_JOINT_VEL_RPS:
            raise SafetyValidationError(
                f"Gesture joint speed {peak:.2f} rad/s exceeds {GESTURE_MAX_JOINT_VEL_RPS:.2f} rad/s."
            )

    def assert_robot_state_safe(
        self,
        robot_status: RobotStatus,
        require_active_watchdog: bool = False,
    ) -> None:
        """Rule 8, 9, 10: Reject commands if E-Stop is latched, robot is faulted/unsafe, or watchdog expired."""
        if robot_status.estop_active or robot_status.state == RobotOperationalState.ESTOP:
            raise SafetyValidationError("Robot Emergency Stop (E-STOP) is active; command rejected.")

        unsafe_states = {
            RobotOperationalState.FAULT,
            RobotOperationalState.LOW_BATTERY,
            RobotOperationalState.UNSAFE,
            RobotOperationalState.DISCONNECTED,
        }
        if robot_status.state in unsafe_states:
            raise SafetyValidationError(f"Robot is in unsafe state '{robot_status.state}'; command rejected.")

        if robot_status.battery_percent < 10.0:
            raise SafetyValidationError(
                f"Robot battery critically low ({robot_status.battery_percent:.1f}%); command rejected."
            )

        if require_active_watchdog:
            if not robot_status.bridge_connected or robot_status.last_heartbeat_utc is None:
                raise SafetyValidationError("ROS 2 bridge is disconnected; watchdog rejected command.")
            age_s = (datetime.now(tz=UTC) - robot_status.last_heartbeat_utc).total_seconds()
            if age_s > self._settings.robot_watchdog_timeout_seconds:
                raise SafetyValidationError(
                    f"ROS 2 bridge heartbeat stale ({age_s:.1f}s > {self._settings.robot_watchdog_timeout_seconds:.1f}s)."
                )

    def assert_not_expired(self, command: ValidatedRobotCommand) -> None:
        """Rule 7: Reject commands past their expiration TTL."""
        now = datetime.now(tz=UTC)
        if now >= command.expires_at:
            raise SafetyValidationError(
                f"Command '{command.command_id}' expired at {command.expires_at.isoformat()}."
            )
