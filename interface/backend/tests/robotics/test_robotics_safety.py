"""Tests for Phase 5: All 10 mandatory robotics safety rules, ER 2 analysis, confirmation, and ROS 2 bridge WS."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.core.config import Settings
from app.main import create_app
from app.robotics.action_schema import (
    AllowedActionType,
    RobotActionProposal,
    RobotOperationalState,
    RobotStatus,
    Target3D,
)
from app.robotics.action_validator import ActionSafetyValidator, SafetyValidationError
from app.robotics.intent_parser import (
    parse_multilingual_robot_intent,
    should_trigger_robot_camera,
)


def _make_validator() -> ActionSafetyValidator:
    return ActionSafetyValidator(
        Settings(
            robot_require_confirmation=True,
            robot_command_ttl_seconds=30,
            robot_watchdog_timeout_seconds=5.0,
            robot_max_linear_speed_mps=0.5,
            robot_max_workspace_radius_m=2.5,
        )
    )


def test_rule1_and_rule2_reject_code_and_raw_ros_topics() -> None:
    validator = _make_validator()
    status = RobotStatus()

    with pytest.raises(SafetyValidationError, match="forbidden"):
        validator.parse_and_validate_proposal(
            {"action": "move_to", "ros_topic": "/cmd_vel", "target": {"x": 0.2, "y": 0.0, "z": 0.0}},
            robot_status=status,
        )

    with pytest.raises(SafetyValidationError, match="Forbidden code"):
        validator.parse_and_validate_proposal(
            {
                "action": "inspect_object",
                "object_label": "bottle; os.system('rm -rf /')",
                "speed": 0.1,
            },
            robot_status=status,
        )


def test_rule3_rejects_unlisted_actions_and_extra_fields() -> None:
    validator = _make_validator()
    status = RobotStatus()

    with pytest.raises(SafetyValidationError):
        validator.parse_and_validate_proposal(
            {"action": "spin_motors_raw", "speed": 0.1},
            robot_status=status,
        )

    with pytest.raises(SafetyValidationError):
        validator.parse_and_validate_proposal(
            {"action": "stop", "unexpected_field": 123},
            robot_status=status,
        )


def test_rule4_and_rule5_enforce_speed_and_workspace_limits() -> None:
    validator = _make_validator()
    status = RobotStatus()

    # Speed over 0.5 m/s rejected
    with pytest.raises(SafetyValidationError, match="exceeds limit"):
        validator.parse_and_validate_proposal(
            RobotActionProposal(
                action=AllowedActionType.MOVE_TO,
                target=Target3D(x=0.5, y=0.0, z=0.0),
                speed=0.85,
            ),
            robot_status=status,
        )

    # Position outside 2.5m workspace radius rejected
    with pytest.raises(SafetyValidationError, match="workspace radius"):
        validator.parse_and_validate_proposal(
            RobotActionProposal(
                action=AllowedActionType.MOVE_TO,
                target=Target3D(x=3.0, y=1.0, z=0.0),
                speed=0.2,
            ),
            robot_status=status,
        )


def test_rule6_rule7_rule8_rule9_rule10_lifecycle() -> None:
    validator = _make_validator()

    # Rule 6: Motion forces requires_confirmation=True even if LLM sets False
    cmd = validator.parse_and_validate_proposal(
        RobotActionProposal(
            action=AllowedActionType.MOVE_TO,
            target=Target3D(x=0.5, y=0.2, z=0.0),
            speed=0.15,
            requires_confirmation=False,
        ),
        robot_status=RobotStatus(),
    )
    assert cmd.requires_confirmation is True
    assert cmd.confirmed_by_human is False

    # Rule 7: Expired command rejected
    cmd.expires_at = datetime.now(tz=UTC) - timedelta(seconds=1)
    with pytest.raises(SafetyValidationError, match="expired"):
        validator.assert_not_expired(cmd)

    # Rule 8: Stale watchdog heartbeat rejected
    stale_status = RobotStatus(
        state=RobotOperationalState.READY,
        bridge_connected=True,
        last_heartbeat_utc=datetime.now(tz=UTC) - timedelta(seconds=20),
    )
    with pytest.raises(SafetyValidationError, match="stale"):
        validator.assert_robot_state_safe(stale_status, require_active_watchdog=True)

    # Rule 9 & 10: E-Stop and Fault states reject motion commands, but allow STOP
    estop_status = RobotStatus(state=RobotOperationalState.ESTOP, estop_active=True)
    with pytest.raises(SafetyValidationError, match="E-STOP"):
        validator.parse_and_validate_proposal(
            RobotActionProposal(
                action=AllowedActionType.MOVE_TO,
                target=Target3D(x=0.2, y=0.0, z=0.0),
                speed=0.1,
            ),
            robot_status=estop_status,
        )

    stop_cmd = validator.parse_and_validate_proposal(
        RobotActionProposal(action=AllowedActionType.STOP, speed=0.0),
        robot_status=estop_status,
    )
    assert stop_cmd.action == AllowedActionType.STOP


def test_er2_visual_analysis_confirmation_and_authenticated_bridge_ws() -> None:
    settings = Settings(
        app_env="test",
        stt_default_provider="mock",
        stt_swiss_german_provider="mock",
        tts_default_provider="mock",
        tts_swiss_german_provider="mock",
        llm_provider="mock",
        robotics_enabled=True,
        robotics_provider="mock",
        robot_bridge_shared_token="bridge-secret-token",
        serve_frontend=False,
    )
    app = create_app(settings_override=settings)

    with TestClient(app) as client:
        # Unauthenticated bridge rejected
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/robot-bridge") as bad_ws:
            bad_ws.receive_json()

        # Authenticated ROS 2 bridge connects and sends telemetry heartbeat
        with client.websocket_connect(
            "/ws/robot-bridge",
            headers={"Authorization": "Bearer bridge-secret-token"},
        ) as bridge_ws:
            bridge_ws.send_json(
                {
                    "type": "telemetry",
                    "payload": {
                        "robot_id": "leukerbad-bot-01",
                        "state": "ready",
                        "estop_active": False,
                        "battery_percent": 92.0,
                        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    },
                }
            )
            ack = bridge_ws.receive_json()
            assert ack["type"] == "heartbeat_ack"

            # Submit camera frame + instruction to ER 2 endpoint
            fake_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
            r_er2 = client.post(
                "/api/v1/robotics/analyze",
                data={"instruction": "Move forward carefully toward the bottle"},
                files={"image": ("camera.jpg", fake_jpeg, "image/jpeg")},
            )
            assert r_er2.status_code == 200
            er2_data = r_er2.json()
            assert len(er2_data["reasoning"]["detected_objects"]) == 1
            assert len(er2_data["validated_commands"]) == 1
            cmd_info = er2_data["validated_commands"][0]
            assert cmd_info["awaiting_confirmation"] is True
            cmd_id = cmd_info["command"]["command_id"]

            # Approve via human confirmation -> delivered over outbound bridge WS!
            r_confirm = client.post(f"/api/v1/robotics/confirm/{cmd_id}")
            assert r_confirm.status_code == 200
            assert r_confirm.json()["dispatched_to_bridge"] is True

            dispatched_msg = bridge_ws.receive_json()
            assert dispatched_msg["type"] == "execute_command"
            assert dispatched_msg["command"]["command_id"] == cmd_id
            assert dispatched_msg["command"]["confirmed_by_human"] is True

            # Trigger Emergency Stop -> bridge receives immediate emergency_stop message
            r_estop = client.post("/api/v1/robotics/estop", json={"reason": "Safety test E-Stop"})
            assert r_estop.status_code == 200
            estop_msg = bridge_ws.receive_json()
            assert estop_msg["type"] == "emergency_stop"


def test_multilingual_go_forward_5m_intent_parser_all_languages() -> None:
    cases = [
        ("go forward 5m", "en-US"),
        ("Can you move forward five meters please?", "en-GB"),
        ("Avance de 5 mètres s'il te plaît", "fr-FR"),
        ("Geh 5 Meter vorwärts", "de-DE"),
        ("Gang füüf Meter füre", "gsw"),
        ("请向前走五米", "zh-CN"),
        ("請往前移動 5 公尺", "zh-TW"),
    ]
    for utterance, lang in cases:
        parsed = parse_multilingual_robot_intent(utterance, lang)
        assert parsed is not None, f"Failed parsing: {utterance!r}"
        tool_name, args = parsed
        assert tool_name == "propose_robot_action"
        assert args["action"] == "move_to"
        assert args["x"] == pytest.approx(5.0)
        assert args["y"] == pytest.approx(0.0)

    # Guard against idioms and capability questions
    assert parse_multilingual_robot_intent("I look forward to visiting Leukerbad!", "en-US") is None
    assert parse_multilingual_robot_intent("How fast can the robot move forward?", "en-US") is None
    assert parse_multilingual_robot_intent("机器人最快能走多快？", "zh-CN") is None


def test_should_trigger_robot_camera_only_when_necessary_all_7_languages() -> None:
    # Visual / object / manipulation prompts MUST trigger the camera across all 7 languages
    visual_prompts = [
        "Look around and find the frisbee on the floor",
        "What do you see in front of you?",
        "Was siehst du vor dir? Such die Flasche.",
        "Lueg ume und suech s Frisbee",
        "Regarde autour et attrape la bouteille",
        "Mira alrededor y busca el frisbee",
        "Podívej se kolem a najdi frisbee",
        "看看周围找一下飞盘并抓起来",
    ]
    for prompt in visual_prompts:
        assert should_trigger_robot_camera(prompt) is True, f"Expected camera trigger for: {prompt!r}"

    # Pure blind kinematic commands & idioms MUST NOT trigger the camera
    blind_prompts = [
        "Move forward 5 m and turn around.",
        "Geh 5 Meter vorwärts und dreh dich um.",
        "Avance de 5 mètres et fais demi-tour.",
        "Avanza cinco metros hacia adelante y date la vuelta.",
        "Jdi pět metrů dopředu a otoč se.",
        "向前走五米然后掉头。",
        "I look forward to visiting Leukerbad!",
        "See you tomorrow!",
    ]
    for prompt in blind_prompts:
        assert should_trigger_robot_camera(prompt) is False, f"Unexpected camera trigger for: {prompt!r}"


def test_on_demand_panther_camera_trigger_over_bridge_ws_and_voice_turn() -> None:
    import base64
    import threading

    settings = Settings(
        app_env="test",
        stt_default_provider="mock",
        stt_swiss_german_provider="mock",
        tts_default_provider="mock",
        tts_swiss_german_provider="mock",
        llm_provider="mock",
        robotics_enabled=True,
        robotics_provider="mock",
        robot_bridge_shared_token="panther-secret-15",
        serve_frontend=False,
    )
    app = create_app(settings_override=settings)
    fake_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
    fake_b64 = base64.b64encode(fake_jpeg).decode("ascii")
    camera_requests: list[dict[str, object]] = []

    with TestClient(app) as client, client.websocket_connect(
        "/ws/robot-bridge",
        headers={"Authorization": "Bearer panther-secret-15"},
    ) as bridge_ws:
        bridge_ws.send_json(
            {
                "type": "telemetry",
                "payload": {
                    "robot_id": "unix-panther-01",
                    "state": "ready",
                    "estop_active": False,
                    "battery_percent": 89.0,
                    "ros_domain_id": 15,
                    "camera_ready": True,
                    "camera_source": "/camera/color/image_raw",
                    "lift_height_m": 0.18,
                    "front_obstacle_distance_m": 1.65,
                    "position": {"x": 0.2, "y": -0.1, "z": 0.0},
                },
            }
        )
        assert bridge_ws.receive_json()["type"] == "heartbeat_ack"

        stop_responder = threading.Event()

        def _bridge_camera_responder() -> None:
            while not stop_responder.is_set():
                try:
                    msg = bridge_ws.receive_json()
                except Exception:
                    break
                if msg.get("type") == "capture_camera":
                    camera_requests.append(msg)
                    bridge_ws.send_json(
                        {
                            "type": "camera_frame",
                            "request_id": msg["request_id"],
                            "jpeg_base64": fake_b64,
                            "mime_type": "image/jpeg",
                            "width": 640,
                            "height": 480,
                            "source": "/camera/color/image_raw",
                            "center_depth_m": 1.25,
                            "driver_triggered": True,
                        }
                    )
                elif msg.get("type") == "ping_stop":
                    break

        responder = threading.Thread(target=_bridge_camera_responder, daemon=True)
        responder.start()
        try:
            # 1. Blind motion command -> does NOT trigger camera
            with client.websocket_connect("/ws/voice?language=en-US") as voice_ws:
                assert voice_ws.receive_json()["type"] == "session_ready"
                voice_ws.send_json({"type": "text_input", "text": "Move forward 5 m and turn around."})
                blind_resp = None
                for _ in range(10):
                    ev = voice_ws.receive_json()
                    if ev["type"] == "assistant_response":
                        blind_resp = ev["assistant"]
                        break
                assert blind_resp is not None
                assert blind_resp["camera_triggered"] is False
                assert len(camera_requests) == 0

                # Drain remaining TTS stream frames
                for _ in range(8):
                    ev = voice_ws.receive_json()
                    if ev["type"] == "tts_stream_complete":
                        break

                # 2. Visual frisbee command -> AUTOMATICALLY triggers Panther /camera/color/image_raw!
                voice_ws.send_json(
                    {"type": "text_input", "text": "Look around and find the frisbee on the floor."}
                )
                vis_resp = None
                for _ in range(12):
                    ev = voice_ws.receive_json()
                    if ev["type"] == "assistant_response":
                        vis_resp = ev["assistant"]
                        break
                assert vis_resp is not None
                assert vis_resp["camera_triggered"] is True
                assert vis_resp["camera_captured"] is True
                assert vis_resp["camera_frame"]["source"] == "/camera/color/image_raw"
                assert vis_resp["detected_objects"][0]["label"] == "frisbee"
                assert len(camera_requests) == 1

            # 3. Explicit POST /api/v1/robotics/camera/capture endpoint triggers fresh capture (max_age=0)
            r_cap = client.post(
                "/api/v1/robotics/camera/capture",
                json={"reason": "Test manual capture", "max_age_seconds": 0.0},
            )
            assert r_cap.status_code == 200
            assert r_cap.json()["captured"] is True
            assert r_cap.json()["frame"]["source"] == "/camera/color/image_raw"
            assert len(camera_requests) == 2
        finally:
            stop_responder.set()
            # Trigger estop to wake bridge_ws.receive_json() cleanly
            client.post("/api/v1/robotics/estop", json={"reason": "End test responder"})
            responder.join(timeout=2.0)


def test_shorthand_units_decimals_halves_and_grammar_precision() -> None:
    import math

    from app.robotics.intent_parser import format_er2_plan_summary

    linear_cases = [
        ("5 m forward", "en-US", 5.0, 0.0),
        ("forward 3.25 meters", "en-US", 3.25, 0.0),
        ("two point five meters forward", "en-US", 2.5, 0.0),
        ("two and a half meters back", "en-US", -2.5, 0.0),
        ("50 cm left", "en-US", 0.0, 0.5),
        ("250 mm right", "en-US", 0.0, -0.25),
        ("anderthalb Meter vorwärts", "de-DE", 1.5, 0.0),
        ("zwei komma fünf Meter rückwärts", "de-DE", -2.5, 0.0),
        ("deux mètres et demi en avant", "fr-FR", 2.5, 0.0),
        ("dos metros y medio adelante", "es-ES", 2.5, 0.0),
        ("dva a půl metru dopředu", "cs-CZ", 2.5, 0.0),
        ("两米半向前", "zh-CN", 2.5, 0.0),
    ]
    for phrase, lang, exp_x, exp_y in linear_cases:
        parsed = parse_multilingual_robot_intent(phrase, lang)
        assert parsed is not None, f"Failed to parse shorthand/decimal: {phrase!r}"
        t_name, args = parsed
        assert t_name == "propose_robot_action"
        assert args["action"] == "move_to"
        assert args["x"] == pytest.approx(exp_x, abs=1e-3), phrase
        assert args["y"] == pytest.approx(exp_y, abs=1e-3), phrase

    angle_cases = [
        ("90 degrees right", "en-US", -math.pi / 2),
        ("45° left", "en-US", math.pi / 4),
        ("rotate 1.5708 radians counterclockwise", "en-US", 1.5708),
        ("turn thirty degrees clockwise", "en-US", -math.radians(30.0)),
    ]
    for phrase, lang, exp_yaw in angle_cases:
        parsed = parse_multilingual_robot_intent(phrase, lang)
        assert parsed is not None, f"Failed to parse angle shorthand: {phrase!r}"
        t_name, args = parsed
        assert t_name == "propose_robot_action"
        assert args["action"] == "rotate"
        assert args["yaw_rad"] == pytest.approx(exp_yaw, abs=1e-3), phrase

    # Verify singular vs plural grammar ("1 meter" vs "5 meters", never "1 meters")
    s_1m = format_er2_plan_summary(
        [{"action": "move_to", "target": {"x": 1.0, "y": 0.0, "z": 0.0}, "requires_confirmation": True}],
        [],
        "en-US",
    )
    assert s_1m == "Queued: move forward 1 meter. Please confirm in the robotics panel."

    s_5m = format_er2_plan_summary(
        [{"action": "move_to", "target": {"x": 5.0, "y": 0.0, "z": 0.0}, "requires_confirmation": True}],
        [],
        "en-US",
    )
    assert s_5m == "Queued: move forward 5 meters. Please confirm in the robotics panel."


@pytest.mark.asyncio
async def test_er2_client_enforces_exact_5m_precision_and_strips_spurious_fields() -> None:
    from types import SimpleNamespace

    from app.robotics.action_schema import DetectedObject2D, RoboticsReasoningResult
    from app.robotics.er2_client import GeminiRoboticsER2Client

    settings = Settings(app_env="test", gcp_project_id="test-proj")
    er2 = GeminiRoboticsER2Client(settings)

    # 1. Exact reproduction of user bug: ER 2 conservatively returns x=1.0, yaw_rad=0.0, object_label="safe_path"
    #    when asked to "move forward 5 meters" (from user utterance "5 m forward").
    shrunk_er2_payload = RoboticsReasoningResult(
        scene_summary="Moving forward safely.",
        task_progress_percent=10,
        safety_observations=["Confirm path is clear."],
        proposals=[
            RobotActionProposal(
                action=AllowedActionType.MOVE_TO,
                target=Target3D(x=1.0, y=0.0, z=0.0),
                yaw_rad=0.0,
                object_label="safe_path",
                speed=0.25,
                requires_confirmation=True,
                rationale="Advance cautiously.",
            )
        ],
    )

    class _FakeModels:
        def __init__(self, result_to_return: RoboticsReasoningResult) -> None:
            self._res = result_to_return

        async def generate_content(self, **kwargs: object) -> object:
            return SimpleNamespace(parsed=self._res, text="")

    er2._client = SimpleNamespace(aio=SimpleNamespace(models=_FakeModels(shrunk_er2_payload)))  # type: ignore[assignment]
    res = await er2.analyze_and_propose(
        instruction="move forward 5 meters",
        raw_user_instruction="5 m forward",
        robot_status=RobotStatus(),
        image_bytes=None,
    )
    assert len(res.proposals) == 1
    p0 = res.proposals[0]
    assert p0.action == AllowedActionType.MOVE_TO
    assert p0.target is not None
    assert p0.target.x == pytest.approx(5.0)
    assert p0.target.y == pytest.approx(0.0)
    assert p0.target.z == pytest.approx(0.0)
    assert p0.yaw_rad is None
    assert p0.object_label is None

    # 2. If ER 2 tries to subdivide "move forward 5 meters" into five 1m steps, collapse to 1 exact 5.0m step
    subdivided_payload = RoboticsReasoningResult(
        scene_summary="Subdivided steps.",
        proposals=[
            RobotActionProposal(
                action=AllowedActionType.MOVE_TO,
                target=Target3D(x=1.0, y=0.0, z=0.0),
                yaw_rad=0.0,
                object_label="waypoint",
                speed=0.25,
            )
            for _ in range(5)
        ],
    )
    er2._client = SimpleNamespace(aio=SimpleNamespace(models=_FakeModels(subdivided_payload)))  # type: ignore[assignment]
    res_sub = await er2.analyze_and_propose(
        instruction="move forward 5 meters",
        robot_status=RobotStatus(),
        image_bytes=None,
    )
    assert len(res_sub.proposals) == 1
    assert res_sub.proposals[0].target is not None
    assert res_sub.proposals[0].target.x == pytest.approx(5.0)
    assert res_sub.proposals[0].yaw_rad is None
    assert res_sub.proposals[0].object_label is None

    # 3. Visual camera task WITHOUT explicit metric distance MUST preserve ER 2's visual coordinates & object_label
    visual_payload = RoboticsReasoningResult(
        scene_summary="Orange frisbee visible 1.42m ahead-right.",
        detected_objects=[DetectedObject2D(label="orange_frisbee", point_yx_norm=[650, 580], confidence=0.96)],
        proposals=[
            RobotActionProposal(
                action=AllowedActionType.MOVE_TO,
                target=Target3D(x=1.42, y=-0.28, z=0.0),
                yaw_rad=0.0,
                object_label="orange_frisbee",
                speed=0.2,
                requires_confirmation=True,
                rationale="Approach visually localized orange frisbee.",
            )
        ],
    )
    er2._client = SimpleNamespace(aio=SimpleNamespace(models=_FakeModels(visual_payload)))  # type: ignore[assignment]
    res_vis = await er2.analyze_and_propose(
        instruction="Look for the orange frisbee and move toward it",
        robot_status=RobotStatus(),
        image_bytes=b"\xff\xd8\xff\xd9",
    )
    assert len(res_vis.proposals) == 1
    pv = res_vis.proposals[0]
    assert pv.target is not None
    assert pv.target.x == pytest.approx(1.42)
    assert pv.target.y == pytest.approx(-0.28)
    assert pv.object_label == "orange_frisbee"
    assert pv.yaw_rad is None
