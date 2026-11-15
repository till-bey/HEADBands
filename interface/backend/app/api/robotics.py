"""Robotics ER 2 visual analysis, on-demand camera trigger, confirmations, E-Stop, and Panther ROS 2 WS."""

from __future__ import annotations

import base64
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, Field, ValidationError

from app.container import ServiceContainer, get_container
from app.core.logging import get_logger
from app.core.security import require_api_auth, verify_robot_bridge_ws_auth
from app.robotics.action_schema import AllowedActionType, RobotActionProposal, RobotTelemetryEvent
from app.robotics.action_validator import SafetyValidationError

logger = get_logger(__name__)

router = APIRouter(tags=["robotics"])


class EStopRequest(BaseModel):
    reason: str = Field(default="Manual operator emergency stop", max_length=300)


class CameraCaptureRequest(BaseModel):
    reason: str = Field(default="Operator on-demand camera capture", max_length=200)
    max_age_seconds: float = Field(default=0.0, ge=0.0, le=30.0)


class GestureRequest(BaseModel):
    description: str = Field(min_length=2, max_length=200)


@router.post("/api/v1/robotics/gesture", dependencies=[Depends(require_api_auth)])
async def plan_generated_gesture(
    body: GestureRequest,
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> dict[str, Any]:
    """Plan a gesture from a description and queue it for human confirmation (never dispatched directly)."""
    if not container.settings.robotics_enabled:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Robotics integration is disabled.")
    description = body.description.strip()
    try:
        cmd, dispatched = await container.robot_gateway.submit_proposal(
            RobotActionProposal(
                action=AllowedActionType.GENERATED_GESTURE,
                object_label=description[:120],
                speed=0.2,
                requires_confirmation=True,
                rationale=f"Generated arm gesture: {description[:200]}",
            )
        )
    except SafetyValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return {
        "planned": True,
        "dispatched": dispatched,
        "awaiting_confirmation": cmd.requires_confirmation and not cmd.confirmed_by_human,
        "command": cmd.model_dump(mode="json"),
    }


@router.get("/api/v1/robotics/status", dependencies=[Depends(require_api_auth)])
async def get_robotics_status(
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> dict[str, Any]:
    latest_cam = container.robot_gateway.get_latest_camera_frame()
    return {
        "robotics_enabled": container.settings.robotics_enabled,
        "status": container.robot_gateway.get_status().model_dump(mode="json"),
        "latest_camera_available": latest_cam is not None,
        "latest_camera_timestamp_utc": latest_cam.timestamp_utc.isoformat() if latest_cam else None,
        "pending_confirmations": [
            cmd.model_dump(mode="json") for cmd in container.robot_gateway.list_pending_confirmations()
        ],
    }


@router.post("/api/v1/robotics/camera/capture", dependencies=[Depends(require_api_auth)])
async def capture_robot_camera(
    container: Annotated[ServiceContainer, Depends(get_container)],
    body: CameraCaptureRequest | None = None,
) -> dict[str, Any]:
    """Trigger `/camera/color/image_raw` (auto-launching Orbbec Gemini 2 if needed) on the Panther ROS 2 bridge."""
    if not container.settings.robotics_enabled:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Robotics integration is disabled.")

    req = body or CameraCaptureRequest()
    frame = await container.robot_gateway.request_camera_frame(
        reason=req.reason,
        max_age_seconds=req.max_age_seconds,
    )
    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No camera frame received; ensure UniX AI Panther ROS 2 bridge is connected on /ws/robot-bridge.",
        )
    return {
        "captured": True,
        "frame": frame.model_dump(mode="json"),
    }


@router.get("/api/v1/robotics/camera/latest", dependencies=[Depends(require_api_auth)])
async def get_latest_robot_camera(
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> dict[str, Any]:
    """Return the most recently captured Panther camera snapshot without re-triggering hardware."""
    frame = container.robot_gateway.get_latest_camera_frame()
    if frame is None:
        return {"available": False, "frame": None}
    return {"available": True, "frame": frame.model_dump(mode="json")}


@router.get("/api/v1/robotics/joints", dependencies=[Depends(require_api_auth)])
async def get_latest_joint_states(
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> dict[str, Any]:
    """Newest Panther joint positions streamed by the ROS 2 bridge, for the Web UI simulation view."""
    snapshot, age = container.robot_gateway.get_latest_joint_states()
    if snapshot is None:
        return {"available": False, "positions": {}, "age_seconds": None, "timestamp_utc": None}
    return {
        "available": True,
        "positions": snapshot.positions,
        "age_seconds": round(age, 3) if age is not None else None,
        "timestamp_utc": snapshot.timestamp_utc.isoformat(),
    }


@router.post("/api/v1/robotics/analyze", dependencies=[Depends(require_api_auth)])
async def analyze_workspace_with_er2(
    container: Annotated[ServiceContainer, Depends(get_container)],
    instruction: Annotated[str, Form(min_length=1, max_length=2000)],
    image: Annotated[UploadFile | None, File()] = None,
) -> dict[str, Any]:
    """Run Gemini Robotics ER 2 on uploaded image OR auto-triggered Panther `/camera/color/image_raw` frame."""
    if not container.settings.robotics_enabled:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Robotics integration is disabled.")

    img_bytes: bytes | None = None
    mime_type = "image/jpeg"
    camera_triggered = False
    captured_frame_json: dict[str, Any] | None = None

    if image is not None:
        img_bytes = await image.read()
        if len(img_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image exceeds 10 MB.",
            )
        mime_type = image.content_type or "image/jpeg"
    else:
        # Automatically trigger the Panther ROS 2 camera when no manual image file was uploaded
        camera_triggered = True
        cam_snap = await container.robot_gateway.request_camera_frame(
            reason=instruction,
            max_age_seconds=1.0,
        )
        if cam_snap is not None:
            try:
                img_bytes = base64.b64decode(cam_snap.jpeg_base64)
                mime_type = cam_snap.mime_type
                captured_frame_json = cam_snap.model_dump(mode="json")
            except Exception:
                img_bytes = None

    robot_status = container.robot_gateway.get_status()
    er2_result = await container.robotics_er2.analyze_and_propose(
        instruction=instruction,
        robot_status=robot_status,
        image_bytes=img_bytes,
        image_mime_type=mime_type,
    )

    validated_commands: list[dict[str, Any]] = []
    rejected_proposals: list[dict[str, Any]] = []

    for prop in er2_result.proposals:
        try:
            cmd, dispatched = await container.robot_gateway.submit_proposal(prop)
            validated_commands.append(
                {
                    "command": cmd.model_dump(mode="json"),
                    "dispatched": dispatched,
                    "awaiting_confirmation": cmd.requires_confirmation and not cmd.confirmed_by_human,
                }
            )
        except SafetyValidationError as exc:
            rejected_proposals.append(
                {
                    "proposal": prop.model_dump(mode="json"),
                    "rejected_reason": str(exc),
                }
            )

    return {
        "reasoning": er2_result.model_dump(mode="json"),
        "camera_triggered": camera_triggered,
        "camera_captured": captured_frame_json is not None,
        "camera_frame": captured_frame_json,
        "validated_commands": validated_commands,
        "rejected_proposals": rejected_proposals,
    }


@router.post("/api/v1/robotics/propose", dependencies=[Depends(require_api_auth)])
async def submit_robot_proposal(
    proposal: dict[str, Any],
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> dict[str, Any]:
    """Validate and queue/dispatch an explicit application-level action proposal."""
    if not container.settings.robotics_enabled:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Robotics integration is disabled.")
    try:
        cmd, dispatched = await container.robot_gateway.submit_proposal(proposal)
        return {
            "validated": True,
            "dispatched": dispatched,
            "awaiting_confirmation": cmd.requires_confirmation and not cmd.confirmed_by_human,
            "command": cmd.model_dump(mode="json"),
        }
    except SafetyValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/api/v1/robotics/confirm/{command_id}", dependencies=[Depends(require_api_auth)])
async def confirm_robot_command(
    command_id: str,
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> dict[str, Any]:
    """Human confirmation approval gate for physical robot actions (Safety Rule 6)."""
    try:
        cmd, dispatched = await container.robot_gateway.confirm_and_dispatch(command_id)
        return {
            "confirmed": True,
            "dispatched_to_bridge": dispatched,
            "command": cmd.model_dump(mode="json"),
        }
    except SafetyValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/api/v1/robotics/reject/{command_id}", dependencies=[Depends(require_api_auth)])
async def reject_robot_command(
    command_id: str,
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> dict[str, Any]:
    removed = container.robot_gateway.reject_command(command_id)
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Command not found in pending queue.")
    return {"rejected": True, "command_id": command_id}


@router.post("/api/v1/robotics/estop", dependencies=[Depends(require_api_auth)])
async def trigger_estop(
    body: EStopRequest,
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> dict[str, Any]:
    """Trigger immediate emergency stop; purges pending commands and notifies ROS 2 bridge."""
    robot_status = await container.robot_gateway.trigger_emergency_stop(reason=body.reason)
    return {"estop_active": True, "status": robot_status.model_dump(mode="json")}


@router.post("/api/v1/robotics/estop/reset", dependencies=[Depends(require_api_auth)])
async def reset_estop(
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> dict[str, Any]:
    robot_status = await container.robot_gateway.reset_emergency_stop()
    return {"estop_active": robot_status.estop_active, "status": robot_status.model_dump(mode="json")}


@router.websocket("/ws/robot-bridge")
async def robot_bridge_websocket(websocket: WebSocket) -> None:
    """Persistent outbound authenticated channel from the local UniX AI Panther ROS 2 bridge to Cloud Run."""
    container: ServiceContainer = websocket.app.state.container
    if not verify_robot_bridge_ws_auth(websocket, container.settings):
        await websocket.close(code=4401, reason="Unauthorized robot bridge")
        return

    await websocket.accept()
    await container.robot_gateway.attach_bridge(websocket)
    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")
            if msg_type in {"telemetry", "heartbeat"}:
                telemetry = RobotTelemetryEvent.model_validate(msg.get("payload", {}))
                updated = await container.robot_gateway.ingest_telemetry(telemetry)
                await websocket.send_json(
                    {"type": "heartbeat_ack", "estop_active": updated.estop_active}
                )
            elif msg_type == "camera_frame":
                container.robot_gateway.ingest_camera_frame(msg)
            elif msg_type == "camera_frame_error":
                container.robot_gateway.ingest_camera_error(msg)
            elif msg_type == "joint_states":
                try:
                    container.robot_gateway.ingest_joint_states(msg)
                except (ValidationError, TypeError, ValueError) as exc:
                    # Display-only data: a bad frame must not drop the bridge connection.
                    logger.warning("Ignored malformed joint_states frame", extra={"error": str(exc)})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("Robot bridge WS closed with error", extra={"error": str(exc)})
    finally:
        await container.robot_gateway.detach_bridge(websocket)
