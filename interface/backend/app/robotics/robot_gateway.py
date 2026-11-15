"""Robot Safety Gateway managing confirmations, TTLs, E-Stop, on-demand camera capture, and Panther ROS 2 WS."""

from __future__ import annotations

import asyncio
import math
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import WebSocket

from app.core.config import Settings
from app.core.logging import get_logger
from app.infrastructure.pubsub import TelemetryPubSubPublisher
from app.robotics.action_schema import (
    AllowedActionType,
    CameraFrameSnapshot,
    GestureTrajectory,
    JointStatesSnapshot,
    ManualControlRequest,
    RobotActionProposal,
    RobotOperationalState,
    RobotStatus,
    RobotTelemetryEvent,
    ValidatedRobotCommand,
)
from app.robotics.action_validator import ActionSafetyValidator, SafetyValidationError
from app.robotics.gesture_planner import GesturePlanner
from app.robotics.motion_preview import build_preview

logger = get_logger(__name__)


class RobotSafetyGateway:
    """Coordinates cloud-side validation, human confirmation, E-Stop, on-demand camera, and ROS 2 bridge."""

    def __init__(
        self,
        settings: Settings,
        validator: ActionSafetyValidator,
        telemetry_publisher: TelemetryPubSubPublisher,
        gesture_planner: GesturePlanner | None = None,
    ) -> None:
        self._settings = settings
        self._validator = validator
        self._pubsub = telemetry_publisher
        self._gesture_planner = gesture_planner or GesturePlanner(settings)
        self._status = RobotStatus()
        self._pending_commands: dict[str, ValidatedRobotCommand] = {}
        self._dispatched_history: list[ValidatedRobotCommand] = []
        self._bridge_ws: WebSocket | None = None
        self._latest_camera_frame: CameraFrameSnapshot | None = None
        self._latest_camera_monotonic: float | None = None
        self._pending_camera_futures: dict[str, asyncio.Future[CameraFrameSnapshot | None]] = {}
        self._latest_joint_states: JointStatesSnapshot | None = None
        self._latest_joint_monotonic: float | None = None

    def ingest_joint_states(self, raw_payload: dict[str, Any]) -> JointStatesSnapshot:
        """Store the bridge's newest joint positions (display only; never fed back to the robot)."""
        snapshot = JointStatesSnapshot.model_validate(raw_payload)
        snapshot.positions = {
            str(name)[:64]: float(value)
            for name, value in snapshot.positions.items()
            if name and math.isfinite(float(value))
        }
        self._latest_joint_states = snapshot
        self._latest_joint_monotonic = time.monotonic()
        return snapshot

    def get_latest_joint_states(self) -> tuple[JointStatesSnapshot | None, float | None]:
        """Newest joint snapshot and its age in seconds, or (None, None)."""
        if self._latest_joint_states is None or self._latest_joint_monotonic is None:
            return None, None
        age = time.monotonic() - self._latest_joint_monotonic
        return self._latest_joint_states.model_copy(deep=True), age

    def get_status(self) -> RobotStatus:
        """Return live robot status including watchdog freshness check."""
        if self._status.bridge_connected and self._status.last_heartbeat_utc is not None:
            age_s = (datetime.now(tz=UTC) - self._status.last_heartbeat_utc).total_seconds()
            if age_s > self._settings.robot_watchdog_timeout_seconds:
                self._status.bridge_connected = False
                if not self._status.estop_active:
                    self._status.state = RobotOperationalState.DISCONNECTED
        return self._status.model_copy(deep=True)

    def get_latest_camera_frame(self) -> CameraFrameSnapshot | None:
        return self._latest_camera_frame.model_copy(deep=True) if self._latest_camera_frame else None

    def list_pending_confirmations(self) -> list[ValidatedRobotCommand]:
        now = datetime.now(tz=UTC)
        expired_ids = [cid for cid, cmd in self._pending_commands.items() if now >= cmd.expires_at]
        for cid in expired_ids:
            self._pending_commands.pop(cid, None)
        return list(self._pending_commands.values())

    async def attach_bridge(self, websocket: WebSocket) -> None:
        self._bridge_ws = websocket
        self._status.bridge_connected = True
        self._status.last_heartbeat_utc = datetime.now(tz=UTC)
        if not self._status.estop_active:
            self._status.state = RobotOperationalState.READY
        logger.info("Authenticated UniX AI Panther ROS 2 bridge connected")

    async def detach_bridge(self, websocket: WebSocket) -> None:
        if self._bridge_ws is websocket:
            self._bridge_ws = None
            self._status.bridge_connected = False
            self._status.camera_driver_active = False
            if not self._status.estop_active:
                self._status.state = RobotOperationalState.DISCONNECTED
            for fut in self._pending_camera_futures.values():
                if not fut.done():
                    fut.set_result(None)
            self._pending_camera_futures.clear()
            logger.warning("ROS 2 bridge disconnected")

    async def ingest_telemetry(self, telemetry: RobotTelemetryEvent) -> RobotStatus:
        """Update robot state & Panther telemetry and forward asynchronously to Cloud Pub/Sub."""
        self._status.robot_id = telemetry.robot_id
        self._status.estop_active = telemetry.estop_active or (telemetry.state == RobotOperationalState.ESTOP)
        self._status.state = RobotOperationalState.ESTOP if self._status.estop_active else telemetry.state
        self._status.battery_percent = telemetry.battery_percent
        self._status.position = telemetry.position
        self._status.active_command_id = telemetry.active_command_id
        self._status.ros_domain_id = telemetry.ros_domain_id
        self._status.camera_ready = telemetry.camera_ready or (self._latest_camera_frame is not None)
        self._status.camera_driver_active = telemetry.camera_driver_active
        if telemetry.camera_source:
            self._status.camera_source = telemetry.camera_source
        self._status.front_obstacle_distance_m = telemetry.front_obstacle_distance_m
        self._status.lift_height_m = telemetry.lift_height_m
        self._status.left_gripper_pos = telemetry.left_gripper_pos
        self._status.right_gripper_pos = telemetry.right_gripper_pos
        self._status.head_joints_rad = telemetry.head_joints_rad
        if telemetry.active_topics:
            self._status.active_topics = telemetry.active_topics
        self._status.bridge_connected = True
        self._status.last_heartbeat_utc = datetime.now(tz=UTC)

        await self._pubsub.publish_event("robot_telemetry", telemetry.model_dump(mode="json"))
        return self.get_status()

    def ingest_camera_frame(self, raw_payload: dict[str, Any]) -> CameraFrameSnapshot:
        """Store incoming camera snapshot from ROS 2 bridge and wake any awaiting ER 2 caller."""
        snapshot = CameraFrameSnapshot.model_validate(raw_payload)
        self._latest_camera_frame = snapshot
        self._latest_camera_monotonic = time.monotonic()
        self._status.camera_ready = True
        self._status.camera_source = snapshot.source
        self._status.camera_driver_active = snapshot.driver_triggered or self._status.camera_driver_active
        self._status.bridge_connected = True
        self._status.last_heartbeat_utc = datetime.now(tz=UTC)

        fut = self._pending_camera_futures.pop(snapshot.request_id, None)
        if fut is not None and not fut.done():
            fut.set_result(snapshot)
        return snapshot

    def ingest_camera_error(self, raw_payload: dict[str, Any]) -> None:
        req_id = str(raw_payload.get("request_id") or "")
        fut = self._pending_camera_futures.pop(req_id, None)
        if fut is not None and not fut.done():
            fut.set_result(None)

    async def request_camera_frame(
        self,
        reason: str = "visual_inspection",
        timeout_seconds: float = 5.5,
        max_age_seconds: float = 1.5,
    ) -> CameraFrameSnapshot | None:
        """Trigger on-demand camera capture over `/ws/robot-bridge` when visual context is needed."""
        now_mono = time.monotonic()
        if (
            max_age_seconds > 0.0
            and self._latest_camera_frame is not None
            and self._latest_camera_monotonic is not None
            and (now_mono - self._latest_camera_monotonic) <= max_age_seconds
        ):
            return self._latest_camera_frame.model_copy(deep=True)

        if self._bridge_ws is None:
            return None

        req_id = f"cam-{uuid4().hex[:12]}"
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[CameraFrameSnapshot | None] = loop.create_future()
        self._pending_camera_futures[req_id] = fut
        try:
            await self._bridge_ws.send_json(
                {
                    "type": "capture_camera",
                    "request_id": req_id,
                    "reason": reason[:200],
                }
            )
            result = await asyncio.wait_for(fut, timeout=timeout_seconds)
            return result.model_copy(deep=True) if result is not None else None
        except (TimeoutError, Exception) as exc:
            logger.warning(
                "On-demand robot camera capture timed out or failed",
                extra={"request_id": req_id, "error": str(exc)},
            )
            return None
        finally:
            self._pending_camera_futures.pop(req_id, None)

    async def submit_proposal(
        self,
        proposal: RobotActionProposal | dict[str, Any],
    ) -> tuple[ValidatedRobotCommand, bool]:
        """Validate a proposal; either queue for human confirmation or dispatch if safe/non-motion."""
        gesture = None
        action = proposal.get("action") if isinstance(proposal, dict) else proposal.action
        if action == AllowedActionType.GENERATED_GESTURE or action == AllowedActionType.GENERATED_GESTURE.value:
            if isinstance(proposal, dict):
                proposal.pop("gesture", None)  # a caller-supplied trajectory is never trusted
                description = str(proposal.get("object_label") or proposal.get("rationale") or "").strip()
            else:
                description = (proposal.object_label or proposal.rationale or "").strip()
            gesture = await self.plan_gesture(description)
        validated = self._validator.parse_and_validate_proposal(
            proposal,
            robot_status=self.get_status(),
            require_active_bridge_watchdog=False,
            gesture=gesture,
        )
        try:
            validated.preview = build_preview(validated)
        except (OSError, ValueError) as exc:  # a preview is display only: never block the command over it
            logger.warning("No simulation preview for command", extra={"action": validated.action, "error": str(exc)})

        if validated.requires_confirmation and not validated.confirmed_by_human:
            self._pending_commands[validated.command_id] = validated
            logger.info(
                "Robot command queued awaiting mandatory human confirmation",
                extra={"command_id": validated.command_id, "action": validated.action},
            )
            return validated, False

        dispatched = await self._dispatch_to_bridge(validated)
        return validated, dispatched

    async def send_manual(self, req: ManualControlRequest) -> None:
        """Check a manual panel command and send it straight to the bridge, which checks it again."""
        self._validator.assert_manual_safe(req, self.get_status())
        if self._bridge_ws is None:
            # Cloud Run may run several instances; the bridge is connected to only one of them.
            raise SafetyValidationError("The robot bridge is not connected to this server instance.")
        await self._bridge_ws.send_json({"type": "manual", "command": req.model_dump(mode="json")})

    async def plan_gesture(self, description: str) -> GestureTrajectory:
        """Plan a language-described gesture from the robot's current pose; raise if it is refused."""
        if not description:
            raise SafetyValidationError("Gesture needs a description, e.g. 'salute with the right hand'.")
        snapshot, age = self.get_latest_joint_states()
        current = snapshot.positions if snapshot is not None and age is not None and age < 5.0 else None
        result = await self._gesture_planner.plan(description[:200], current)
        if not result.ok or result.trajectory is None:
            raise SafetyValidationError("Gesture refused by the planner: " + " ".join(result.reasons))
        return result.trajectory

    async def confirm_and_dispatch(self, command_id: str) -> tuple[ValidatedRobotCommand, bool]:
        """Approve a pending human-confirmation command if unexpired and robot remains safe."""
        cmd = self._pending_commands.pop(command_id, None)
        if cmd is None:
            raise SafetyValidationError(f"Unknown or already processed command_id '{command_id}'.")

        self._validator.assert_not_expired(cmd)
        self._validator.assert_robot_state_safe(
            robot_status=self.get_status(),
            require_active_watchdog=False,
        )
        if cmd.gesture is not None:
            self._validator.assert_gesture_trajectory_safe(cmd.gesture)

        cmd.confirmed_by_human = True
        cmd.confirmed_at = datetime.now(tz=UTC)
        dispatched = await self._dispatch_to_bridge(cmd)
        return cmd, dispatched

    def reject_command(self, command_id: str) -> bool:
        """Cancel and discard a pending command."""
        return self._pending_commands.pop(command_id, None) is not None

    async def trigger_emergency_stop(self, reason: str = "Operator E-Stop") -> RobotStatus:
        """Rule 9: Immediately latch E-Stop, purge pending commands, and notify bridge if connected."""
        self._status.estop_active = True
        self._status.state = RobotOperationalState.ESTOP
        self._status.active_command_id = None
        self._pending_commands.clear()

        stop_cmd = ValidatedRobotCommand(
            command_id="estop-immediate",
            action=AllowedActionType.STOP,
            speed=0.0,
            requires_confirmation=False,
            confirmed_by_human=True,
            expires_at=datetime.now(tz=UTC),
            rationale=reason,
        )
        if self._bridge_ws is not None:
            try:
                await self._bridge_ws.send_json(
                    {"type": "emergency_stop", "reason": reason, "command": stop_cmd.model_dump(mode="json")}
                )
            except Exception as exc:
                logger.error(
                    "Failed sending E-Stop over bridge WS (local watchdog will halt robot)",
                    extra={"error": str(exc)},
                )

        await self._pubsub.publish_event("robot_estop", {"reason": reason})
        logger.warning("Robot Emergency Stop latched", extra={"reason": reason})
        return self.get_status()

    async def reset_emergency_stop(self) -> RobotStatus:
        """Clear cloud-side E-Stop latch only after explicit operator reset."""
        self._status.estop_active = False
        self._status.state = RobotOperationalState.READY
        if self._bridge_ws is not None:
            try:
                await self._bridge_ws.send_json({"type": "reset_estop"})
            except Exception:
                pass
        return self.get_status()

    async def _dispatch_to_bridge(self, command: ValidatedRobotCommand) -> bool:
        self._dispatched_history.append(command)
        if len(self._dispatched_history) > 100:
            del self._dispatched_history[:-100]

        if self._bridge_ws is None:
            logger.info(
                "Validated command recorded (no live ROS 2 bridge currently connected)",
                extra={"command_id": command.command_id, "action": command.action},
            )
            return False

        # Re-verify live watchdog right before wire transmission
        if command.action != AllowedActionType.STOP:
            self._validator.assert_robot_state_safe(self.get_status(), require_active_watchdog=True)

        await self._bridge_ws.send_json(
            {
                "type": "execute_command",
                "command": command.model_dump(mode="json", exclude={"preview"}),  # the bridge forbids unknown keys
            }
        )
        self._status.active_command_id = command.command_id
        return True
