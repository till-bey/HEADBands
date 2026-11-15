"""Structured function-calling tools for Gemini Flash -> Gemini Robotics ER 2 delegation, camera trigger & safety gateway."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from app.core.config import SWISS_DIALECT_LABELS
from app.robotics.action_schema import AllowedActionType, RobotActionProposal, Target3D
from app.robotics.action_validator import SafetyValidationError
from app.robotics.intent_parser import should_trigger_robot_camera
from app.robotics.robot_gateway import RobotSafetyGateway

if TYPE_CHECKING:
    from app.robotics.er2_client import RoboticsReasoningProvider


def get_tool_declarations() -> list[dict[str, Any]]:
    """Return JSON-schema tool specifications for Gemini Flash (System 1 voice + translator)."""
    return [
        {
            "name": "plan_robotics_task_with_er2",
            "description": (
                "Delegate ANY physical robot action, movement, multi-step navigation, rotation, return-home, "
                "stop, head look-at, camera scene inspection ('what do you see?', 'where is the frisbee?', "
                "'is the path clear?'), pick/gripper grasp, or place/gripper release task on the UniX AI Panther "
                "robot to Gemini Robotics ER 2 (even if the user omits the word 'robot', e.g. 'go forward 5m', "
                "'move forward 5 m and turn around', 'look around and find the frisbee', 'geh 5 Meter vorwärts "
                "und dreh dich um', 'avance de 5 mètres et fais demi-tour', 'gang 5 Meter füre und dräi dich ume', "
                "'avanza 5 metros y date la vuelta', 'jdi 5 metrů dopředu a otoč se', '向前走五米然后掉头'). "
                "Translate the user's complete instruction faithfully into clear English preserving all steps, "
                "distances, angles, directions, and target objects in exact order, set `needs_camera=true` whenever "
                "visual context from `/camera/color/image_raw` is needed, and supply a brief spoken acknowledgment "
                "in the user's active language."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "english_instruction": {
                        "type": "string",
                        "description": (
                            "Complete, unambiguous English translation of the robotics or visual inspection task "
                            "preserving every step, distance in meters, rotation angle, direction, and target object."
                        ),
                    },
                    "quick_spoken_ack": {
                        "type": "string",
                        "description": (
                            "Very short (2 to 6 words) natural spoken acknowledgment in the user's active language "
                            "or Swiss German dialect, e.g. 'Yeah sure, on it!', 'Sure, checking the camera!', "
                            "'Bien sûr, je regarde !', 'Ja klar, mache ich!', 'Ja sicher, machi grad!', "
                            "'¡Claro, enseguida!', 'Jasně, hned se podívám!', '好的，没问题！'."
                        ),
                    },
                    "needs_camera": {
                        "type": "boolean",
                        "description": (
                            "Set true ONLY when visual perception from the robot's RGB-D camera is needed (e.g. "
                            "asking what the robot sees, checking obstacles, finding/locating a frisbee/bottle/object, "
                            "inspecting, looking at a target, picking, or placing). Set false for blind metric motions "
                            "such as 'move forward 5 meters and turn around'."
                        ),
                    },
                },
                "required": ["english_instruction", "quick_spoken_ack"],
            },
        },
        {
            "name": "emergency_stop_robot",
            "description": (
                "Immediately trigger hardware/cloud Emergency Stop (E-Stop) and cancel pending commands "
                "when the user requests an emergency stop (e.g. 'Emergency stop!', 'Not-Aus!', "
                "'Arrêt d'urgence !', '¡Parada de emergencia!', 'Nouzové zastavení!', '急停！')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Reason for emergency stop."},
                },
            },
        },
        {
            "name": "get_robot_status",
            "description": "Read current Panther robot state, battery level, E-Stop status, ROS 2 domain, and pose.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "get_robot_capabilities",
            "description": "Discover allowed Panther robot actions, ROS 2 topics, workspace radius, and safety limits.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "switch_voice_dialect",
            "description": (
                "Switch the assistant's Swiss German TTS voice dialect (ag, be, bs, gr, lu, sg, vs, zh) "
                "ONLY when the user explicitly asks to change or switch the spoken dialect. "
                "Do NOT call this tool if the user merely mentions a Swiss place name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dialect_code": {
                        "type": "string",
                        "enum": list(SWISS_DIALECT_LABELS.keys()),
                    },
                },
                "required": ["dialect_code"],
            },
        },
    ]


class AssistantToolExecutor:
    """Executes structured LLM tool calls, triggering the Panther camera on demand for Gemini Robotics ER 2."""

    def __init__(
        self,
        robot_gateway: RobotSafetyGateway,
        max_speed: float,
        max_radius: float,
        robotics_er2: RoboticsReasoningProvider | None = None,
    ) -> None:
        self._gateway = robot_gateway
        self._max_speed = max_speed
        self._max_radius = max_radius
        self._er2 = robotics_er2

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Return (tool_result_dict, list_of_validated_robot_command_dicts)."""
        if tool_name == "get_robot_capabilities":
            status = self._gateway.get_status()
            return (
                {
                    "allowed_actions": [a.value for a in AllowedActionType],
                    "max_linear_speed_mps": self._max_speed,
                    "max_workspace_radius_m": self._max_radius,
                    "ros_domain_id": status.ros_domain_id,
                    "active_topics": status.active_topics,
                    "human_confirmation_enforced": True,
                },
                [],
            )

        if tool_name == "get_robot_status":
            status = self._gateway.get_status()
            return status.model_dump(mode="json"), []

        if tool_name == "emergency_stop_robot":
            reason = str(arguments.get("reason") or "Voice assistant E-Stop request")
            status = await self._gateway.trigger_emergency_stop(reason=reason)
            return {"status": "estop_latched", "robot_status": status.model_dump(mode="json")}, []

        if tool_name == "switch_voice_dialect":
            code = str(arguments.get("dialect_code", "vs")).lower()
            if code not in SWISS_DIALECT_LABELS:
                return {"error": f"Unsupported dialect '{code}'"}, []
            return {"switched_dialect": code, "label": SWISS_DIALECT_LABELS[code]}, []

        if tool_name == "plan_robotics_task_with_er2":
            if self._er2 is None:
                return {"validated": False, "rejected_reasons": ["Gemini Robotics ER 2 provider not configured."]}, []

            english_instruction = str(arguments.get("english_instruction", "")).strip()
            raw_user_instruction = str(arguments.get("_raw_user_text", "")).strip() or None
            trigger_camera = bool(arguments.get("needs_camera", False)) or should_trigger_robot_camera(
                english_instruction
            )

            cam_bytes: bytes | None = None
            cam_mime = "image/jpeg"
            cam_frame_dict: dict[str, Any] | None = None
            if trigger_camera:
                cam_snap = await self._gateway.request_camera_frame(reason=english_instruction)
                if cam_snap is not None:
                    try:
                        cam_bytes = base64.b64decode(cam_snap.jpeg_base64)
                        cam_mime = cam_snap.mime_type
                        cam_frame_dict = cam_snap.model_dump(mode="json")
                    except Exception:
                        cam_bytes = None

            robot_status = self._gateway.get_status()
            er2_result = await self._er2.analyze_and_propose(
                instruction=english_instruction,
                robot_status=robot_status,
                image_bytes=cam_bytes,
                image_mime_type=cam_mime,
                raw_user_instruction=raw_user_instruction,
            )

            validated_cmds: list[dict[str, Any]] = []
            rejected_reasons: list[str] = []
            for prop in er2_result.proposals:
                try:
                    validated, _ = await self._gateway.submit_proposal(prop)
                    validated_cmds.append(validated.model_dump(mode="json"))
                except SafetyValidationError as exc:
                    rejected_reasons.append(str(exc))

            detected_json = [d.model_dump(mode="json") for d in er2_result.detected_objects]
            return (
                {
                    "validated": len(validated_cmds) > 0,
                    "er2_model": er2_result.model,
                    "scene_summary": er2_result.scene_summary,
                    "safety_observations": er2_result.safety_observations,
                    "camera_triggered": trigger_camera,
                    "camera_captured": cam_bytes is not None,
                    "camera_frame": cam_frame_dict,
                    "detected_objects": detected_json,
                    "validated_commands": validated_cmds,
                    "rejected_reasons": rejected_reasons,
                },
                validated_cmds,
            )

        if tool_name == "propose_robot_action":
            try:
                action_enum = AllowedActionType(str(arguments["action"]))
                target = None
                if any(k in arguments and arguments[k] is not None for k in ("x", "y", "z")):
                    target = Target3D(
                        x=float(arguments.get("x", 0.0) or 0.0),
                        y=float(arguments.get("y", 0.0) or 0.0),
                        z=float(arguments.get("z", 0.0) or 0.0),
                    )
                elif action_enum == AllowedActionType.RETURN_HOME:
                    target = Target3D(x=0.0, y=0.0, z=0.0)

                raw_speed = float(arguments.get("speed", 0.25) if arguments.get("speed") is not None else 0.25)
                safe_speed = (
                    0.0 if action_enum == AllowedActionType.STOP else min(max(raw_speed, 0.05), self._max_speed)
                )

                yaw_val: float | None = None
                if arguments.get("yaw_rad") is not None:
                    yaw_val = max(-3.14159, min(3.14159, float(arguments["yaw_rad"])))

                proposal = RobotActionProposal(
                    action=action_enum,
                    target=target,
                    yaw_rad=yaw_val,
                    object_label=str(arguments["object_label"]) if arguments.get("object_label") else None,
                    speed=safe_speed,
                    rationale=str(arguments.get("rationale", "Requested via voice assistant")),
                )
                validated, dispatched = await self._gateway.submit_proposal(proposal)
                cmd_json = validated.model_dump(mode="json")
                return (
                    {
                        "validated": True,
                        "awaiting_human_confirmation": validated.requires_confirmation and not validated.confirmed_by_human,
                        "dispatched_to_bridge": dispatched,
                        "command": cmd_json,
                    },
                    [cmd_json],
                )
            except (SafetyValidationError, ValueError, KeyError) as exc:
                return {"validated": False, "rejected_reason": str(exc)}, []

        return {"error": f"Unknown tool '{tool_name}'"}, []
