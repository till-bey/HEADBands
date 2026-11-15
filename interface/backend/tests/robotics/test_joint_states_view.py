"""Joint states streamed by the ROS 2 bridge for the Web UI simulation view (read-only)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

BRIDGE_TOKEN = "panther-secret-15"


def _app():
    settings = Settings(
        app_env="test",
        stt_default_provider="mock",
        stt_swiss_german_provider="mock",
        tts_default_provider="mock",
        tts_swiss_german_provider="mock",
        llm_provider="mock",
        robotics_enabled=True,
        robotics_provider="mock",
        robot_bridge_shared_token=BRIDGE_TOKEN,
        serve_frontend=False,
    )
    return create_app(settings_override=settings)


def _telemetry(bridge_ws) -> None:
    bridge_ws.send_json({"type": "telemetry", "payload": {"state": "ready", "ros_domain_id": 77}})
    assert bridge_ws.receive_json()["type"] == "heartbeat_ack"


def test_joints_endpoint_is_empty_before_the_bridge_streams() -> None:
    with TestClient(_app()) as client:
        data = client.get("/api/v1/robotics/joints").json()
    assert data == {"available": False, "positions": {}, "age_seconds": None, "timestamp_utc": None}


def test_bridge_joint_states_reach_the_joints_endpoint() -> None:
    with TestClient(_app()) as client, client.websocket_connect(
        "/ws/robot-bridge", headers={"Authorization": f"Bearer {BRIDGE_TOKEN}"}
    ) as bridge_ws:
        bridge_ws.send_json(
            {
                "type": "joint_states",
                "positions": {"joint_la1": -0.5, "body_joint": 0.4, "joint_head0": 0.1},
            }
        )
        _telemetry(bridge_ws)  # round trip: the joint_states frame was handled before this ack

        data = client.get("/api/v1/robotics/joints").json()
        assert data["available"] is True
        assert data["positions"] == {"joint_la1": -0.5, "body_joint": 0.4, "joint_head0": 0.1}
        assert 0 <= data["age_seconds"] < 5


def test_malformed_joint_states_do_not_drop_the_bridge() -> None:
    with TestClient(_app()) as client, client.websocket_connect(
        "/ws/robot-bridge", headers={"Authorization": f"Bearer {BRIDGE_TOKEN}"}
    ) as bridge_ws:
        bridge_ws.send_json({"type": "joint_states", "positions": {"joint_la0": "not-a-number"}})
        bridge_ws.send_json({"type": "joint_states", "positions": {"joint_la0": float("nan"), "joint_la2": 0.3}})
        _telemetry(bridge_ws)  # connection still alive

        data = client.get("/api/v1/robotics/joints").json()
        assert data["positions"] == {"joint_la2": 0.3}  # NaN dropped, bad frame ignored
        assert client.get("/api/v1/robotics/status").json()["status"]["bridge_connected"] is True
