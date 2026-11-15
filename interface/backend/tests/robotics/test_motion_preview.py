"""Simulation previews of pending commands: what they show, and that the bridge never receives them."""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.robotics.action_schema import AllowedActionType, Target3D, ValidatedRobotCommand
from app.robotics.motion_preview import THROW_DIR, THROW_FILES, build_preview, throw_preview

BRIDGE_SRC = Path(__file__).resolve().parents[3] / "robot_bridge" / "ros2_ws" / "src" / "swiss_robot_bridge"
REPO = Path(__file__).resolve().parents[4]
BRIDGE_TOKEN = "panther-secret-15"


def _cmd(action: AllowedActionType, **kw) -> ValidatedRobotCommand:
    return ValidatedRobotCommand(command_id="c1", action=action, speed=kw.pop("speed", 0.2), requires_confirmation=True,
                                 expires_at=datetime.now(tz=UTC) + timedelta(seconds=30), **kw)


@pytest.mark.parametrize("variant", list(THROW_FILES))
def test_throw_preview_matches_the_bridge_timing(variant: str) -> None:
    if str(BRIDGE_SRC) not in sys.path:
        sys.path.insert(0, str(BRIDGE_SRC))
    from swiss_robot_bridge.local_safety import throw_points

    p = throw_preview(variant)
    bridge = throw_points(1.0, variant)
    assert p.joint_names[-1] == "joint_rf" and len(p.joint_names) == 9
    assert [pt.t for pt in p.points] == pytest.approx([t for t, _q, _g in bridge])
    assert p.points[-1].positions[:8] == pytest.approx(list(bridge[-1][1]))


def test_throw_copies_match_the_files_the_bridge_plays() -> None:
    for name in THROW_FILES.values():
        assert (THROW_DIR / name).read_text() == (REPO / "motion" / "retarget" / name).read_text(), name


def test_previews_per_action() -> None:
    move = build_preview(_cmd(AllowedActionType.MOVE_TO, target=Target3D(x=1.0, y=0.0, z=0.0), speed=0.25))
    assert move.base_path == [[0.0, 0.0, 0.0, 0.0], [4.0, 1.0, 0.0, 0.0]] and not move.points

    turn = build_preview(_cmd(AllowedActionType.ROTATE, yaw_rad=math.pi / 2))
    assert turn.base_path[-1][3] == pytest.approx(math.pi / 2) and turn.duration_s == pytest.approx(math.pi / 2 / 0.4, abs=1e-3)

    pick = build_preview(_cmd(AllowedActionType.PICK_OBJECT, object_label="cup", rationale="with the right hand"))
    assert dict(zip(pick.joint_names, pick.points[0].positions))["joint_rf"] == 1.0

    look = build_preview(_cmd(AllowedActionType.LOOK_AT, target=Target3D(x=0.5, y=0.5, z=0.0)))
    yaw = dict(zip(look.joint_names, look.points[0].positions))["joint_head1"]
    assert yaw == pytest.approx(math.pi / 4, abs=1e-3)

    throw = build_preview(_cmd(AllowedActionType.THROW_OBJECT, object_label="frisbee", target=Target3D(x=1, y=0, z=0.3)))
    assert {"joint_head0", "body_joint"} <= set(throw.joint_names) and throw.joint_names.count("joint_rf") == 1
    assert all(len(pt.positions) == len(throw.joint_names) for pt in throw.points)

    assert build_preview(_cmd(AllowedActionType.STOP)) is None


def test_pending_commands_carry_a_preview_but_the_bridge_never_gets_it() -> None:
    settings = Settings(
        app_env="test", stt_default_provider="mock", stt_swiss_german_provider="mock", tts_default_provider="mock",
        tts_swiss_german_provider="mock", llm_provider="mock", robotics_enabled=True, robotics_provider="mock",
        robot_bridge_shared_token=BRIDGE_TOKEN, serve_frontend=False, pubsub_enabled=False,
    )
    with TestClient(create_app(settings_override=settings)) as client, client.websocket_connect(
        "/ws/robot-bridge", headers={"Authorization": f"Bearer {BRIDGE_TOKEN}"}
    ) as bridge_ws:
        bridge_ws.send_json({"type": "telemetry", "payload": {"state": "ready", "ros_domain_id": 77}})
        assert bridge_ws.receive_json()["type"] == "heartbeat_ack"
        res = client.post("/api/v1/robotics/propose", json={"action": "throw_object", "object_label": "frisbee"})
        cmd = res.json()["command"]
        assert res.json()["awaiting_confirmation"] and cmd["preview"]["name"].startswith("throw")
        pending = client.get("/api/v1/robotics/status").json()["pending_confirmations"]
        assert pending[0]["preview"]["points"]
        assert client.post(f"/api/v1/robotics/confirm/{cmd['command_id']}").status_code == 200
        sent = bridge_ws.receive_json()
        assert sent["type"] == "execute_command" and "preview" not in sent["command"]
