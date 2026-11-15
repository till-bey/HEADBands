"""Manual panel: limits on both sides, the 0.2 s base deadman, and the page password."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

BRIDGE_SRC = Path(__file__).resolve().parents[3] / "robot_bridge" / "ros2_ws" / "src" / "swiss_robot_bridge"
if str(BRIDGE_SRC) not in sys.path:
    sys.path.insert(0, str(BRIDGE_SRC))

from swiss_robot_bridge.bridge_node import ROS2HardwareAdapter  # noqa: E402
from swiss_robot_bridge.camera_manager import PantherCameraManager  # noqa: E402
from swiss_robot_bridge.local_safety import (  # noqa: E402
    MANUAL_BASE_TIMEOUT_S,
    LocalSafetyRejectedError,
    LocalSafetySupervisor,
)

BRIDGE_TOKEN = "panther-secret-15"


def _app(api_token: str = ""):
    return create_app(
        settings_override=Settings(
            app_env="test", stt_default_provider="mock", stt_swiss_german_provider="mock",
            tts_default_provider="mock", tts_swiss_german_provider="mock", llm_provider="mock",
            robotics_enabled=True, robotics_provider="mock", robot_bridge_shared_token=BRIDGE_TOKEN,
            serve_frontend=False, pubsub_enabled=False, api_auth_token=api_token,
        )
    )


def _ready_supervisor() -> LocalSafetySupervisor:
    sup = LocalSafetySupervisor()
    sup.record_cloud_heartbeat(0.0)
    return sup


# ------------------------------------------------------------------ bridge: base deadman
def test_manual_jog_stops_after_its_deadline_unless_renewed() -> None:
    sup = _ready_supervisor()
    cmd = sup.check_manual({"base": [0.2, 0.0, 0.0]})
    assert sup.drive_manual(cmd.base, 1.0).linear_x == pytest.approx(0.2)
    assert sup.evaluate_watchdog(1.0 + MANUAL_BASE_TIMEOUT_S - 0.05).linear_x == pytest.approx(0.2)
    sup.drive_manual(cmd.base, 1.15)  # renewed by the next command
    assert sup.evaluate_watchdog(1.3).linear_x == pytest.approx(0.2)
    assert not sup.evaluate_watchdog(1.15 + MANUAL_BASE_TIMEOUT_S + 0.01).is_nonzero()
    assert sup.active_command_id is None


def test_manual_zero_jog_releases_the_base() -> None:
    sup = _ready_supervisor()
    sup.drive_manual((0.0, 0.0, 0.4), 1.0)
    assert not sup.drive_manual((0.0, 0.0, 0.0), 1.05).is_nonzero()
    assert not sup.evaluate_watchdog(1.06).is_nonzero()


def test_manual_checks_on_the_bridge() -> None:
    sup = _ready_supervisor()
    with pytest.raises(LocalSafetyRejectedError):
        sup.check_manual({"joints": {"joint_la0": 3.0}})  # outside the URDF limit
    with pytest.raises(LocalSafetyRejectedError):
        sup.check_manual({"joints": {"wheel": 0.0}})
    with pytest.raises(LocalSafetyRejectedError):
        sup.check_manual({"base": [2.0, 0.0, 0.0]})
    with pytest.raises(LocalSafetyRejectedError):
        sup.check_manual({"ros_topic": "/cmd_vel"})
    # head clamped to the conservative range until PANTHER_HEAD_FULL_RANGE=1
    assert sup.check_manual({"joints": {"joint_head0": 1.2}}).joints["joint_head0"] == pytest.approx(0.34)
    sup.front_obstacle_distance_m = 0.2
    with pytest.raises(LocalSafetyRejectedError):
        sup.check_manual({"base": [0.2, 0.0, 0.0]})
    sup.check_manual({"base": [-0.2, 0.0, 0.0]})  # backing away is allowed
    sup.trigger_local_estop()
    with pytest.raises(LocalSafetyRejectedError):
        sup.check_manual({"joints": {"joint_lf": 1.0}})
    sup.check_manual({"base": [0.0, 0.0, 0.0]})  # a release always passes


def test_manual_arm_target_moves_the_whole_arm_at_safe_speed() -> None:
    hw = ROS2HardwareAdapter(supervisor=LocalSafetySupervisor(), camera_manager=PantherCameraManager())
    cmd = hw._supervisor.check_manual({"joints": {"joint_ra3": 1.0}})
    with pytest.raises(LocalSafetyRejectedError):
        hw.apply_manual(cmd)  # pose unknown: the other joints could not hold
    hw.joint_positions = {f"joint_ra{i}": 0.0 for i in range(8)}
    hw.apply_manual(cmd)
    ((t, q, _),) = hw.last_master_trajectory
    assert q == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert 1.0 / t <= 0.5  # rad/s


# ------------------------------------------------------------------ backend
def test_manual_endpoint_checks_and_forwards_to_the_bridge() -> None:
    with TestClient(_app()) as client:
        limits = client.get("/api/v1/robotics/manual").json()["limits"]
        assert {"joint_la0", "joint_head1", "joint_rf", "body_joint"} <= set(limits)
        assert client.post("/api/v1/robotics/manual", json={"joints": {"joint_la0": 9.0}}).status_code == 422
        assert client.post("/api/v1/robotics/manual", json={"base": [0, 0, 0], "ros_topic": "x"}).status_code == 422
        # a release with no bridge says so instead of failing silently
        res = client.post("/api/v1/robotics/manual", json={"base": [0, 0, 0]})
        assert res.status_code == 422 and "not connected" in res.json()["detail"]

        with client.websocket_connect(
            "/ws/robot-bridge", headers={"Authorization": f"Bearer {BRIDGE_TOKEN}"}
        ) as bridge_ws:
            bridge_ws.send_json({"type": "telemetry", "payload": {"state": "ready", "ros_domain_id": 77}})
            assert bridge_ws.receive_json()["type"] == "heartbeat_ack"
            res = client.post("/api/v1/robotics/manual", json={"base": [0.2, 0, 0], "joints": {"joint_lf": 1}})
            assert res.status_code == 200
            assert bridge_ws.receive_json() == {
                "type": "manual",
                "command": {"base": [0.2, 0.0, 0.0], "joints": {"joint_lf": 1.0}},
            }


def test_page_password_guards_robot_endpoints() -> None:
    with TestClient(_app(api_token="open-sesame")) as client:
        assert client.post("/api/v1/robotics/manual", json={"base": [0, 0, 0]}).status_code == 401
        assert client.get("/api/v1/robotics/manual").status_code == 401
        ok = client.get("/api/v1/robotics/manual", headers={"Authorization": "Bearer open-sesame"})
        assert ok.status_code == 200
