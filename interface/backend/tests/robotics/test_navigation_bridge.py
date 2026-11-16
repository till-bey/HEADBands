"""Frisbee navigation page: the cloud proxy to the bridge, and how the bridge hands navigation's velocity to /cmd_vel."""

from __future__ import annotations

import base64
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

BRIDGE_SRC = Path(__file__).resolve().parents[3] / "robot_bridge" / "ros2_ws" / "src" / "swiss_robot_bridge"
if str(BRIDGE_SRC) not in sys.path:
    sys.path.insert(0, str(BRIDGE_SRC))

from swiss_robot_bridge.local_safety import LocalSafetySupervisor  # noqa: E402
from swiss_robot_bridge.navigation_mode import NavigationMode  # noqa: E402

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


def test_navigation_requests_go_to_the_bridge_and_back() -> None:
    with TestClient(_app(api_token="pw")) as client:
        auth = {"Authorization": "Bearer pw"}
        assert client.get("/api/v1/robotics/navigation/robot/status").status_code == 401
        assert client.get("/api/v1/robotics/navigation/etc/passwd", headers=auth).status_code == 404
        assert client.post("/api/v1/robotics/navigation/frame.jpg", headers=auth).status_code == 404
        assert client.get("/api/v1/robotics/navigation/robot/status", headers=auth).status_code == 503  # no bridge

        with client.websocket_connect("/ws/robot-bridge", headers={"Authorization": f"Bearer {BRIDGE_TOKEN}"}) as ws:
            result = {}
            post = threading.Thread(target=lambda: result.update(res=client.post(
                "/api/v1/robotics/navigation/robot/start", headers=auth, json={"generation": 3})))
            post.start()
            request = ws.receive_json()
            assert {k: request[k] for k in ("type", "method", "path")} == {
                "type": "navigation_http", "method": "POST", "path": "/robot/start"}
            assert request["body"] == '{"generation":3}'
            ws.send_json({"type": "navigation_http_response", "request_id": request["request_id"], "status": 409,
                          "content_type": "application/json",
                          "body_b64": base64.b64encode(b'{"error": "Connect ROS first"}').decode()})
            post.join(timeout=5)
            assert result["res"].status_code == 409
            assert result["res"].json() == {"error": "Connect ROS first"}


def _navigation() -> NavigationMode:
    nav = NavigationMode.__new__(NavigationMode)  # only the handoff state: no camera, server or ROS
    nav._lock, nav._twist, nav._driving = threading.Lock(), (0.0, 0.0, 0.0), False
    return nav


def test_bridge_drives_navigation_through_the_supervisor_and_releases_once() -> None:
    nav, sup = _navigation(), LocalSafetySupervisor()
    sup.record_cloud_heartbeat(100.0)
    assert nav.take_twist(100.0, 0.5, 1.0) is None  # idle: a manual jog is left alone

    nav._twist = (0.9, -3.0, 100.0)
    twist = nav.take_twist(100.05, 0.5, 1.0)
    assert twist == (0.5, 0.0, -1.0)  # clamped to the bridge's speed limits
    sup.drive_manual(twist, 100.05)
    assert sup.evaluate_watchdog(100.1).linear_x == pytest.approx(0.5)

    nav._twist = (-0.9, 0.0, 100.1)
    assert nav.take_twist(100.1, 0.5, 1.0) == (-0.5, 0.0, 0.0)  # backing off is clamped the same way
    nav._twist = (0.0, 0.0, 100.12)
    assert nav.take_twist(100.12, 0.5, 1.0) == (0.0, 0.0, 0.0)  # stopping releases the base once
    assert nav.take_twist(100.15, 0.5, 1.0) is None

    nav._twist = (0.1, 0.0, 100.0)
    assert nav.take_twist(100.5, 0.5, 1.0) is None  # a stale command (navigation loop stopped) never drives


def test_bridge_refuses_requests_outside_the_dashboard() -> None:
    nav = _navigation()
    assert nav.forward("GET", "/", None)["status"] == 404
    assert nav.forward("DELETE", "/settings", None)["status"] == 404
    assert nav.forward("POST", "/settings", "x" * 9000)["status"] == 413


def test_web_ui_camera_is_the_navigation_camera_frame() -> None:
    import time
    from types import SimpleNamespace

    import numpy as np

    nav = _navigation()
    nav._server = SimpleNamespace(camera_frame=None)
    assert nav.camera_frame() is None  # navigation camera not connected
    bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255  # red in OpenCV's BGR order
    nav._server.camera_frame = (bgr, time.monotonic())
    frame = nav.camera_frame()
    assert (frame.source, frame.width, frame.height) == ("navigation_camera", 64, 48)  # JPEG, or PNG without cv2/PIL
    nav._server.camera_frame = (bgr, time.monotonic() - 5.0)
    assert nav.camera_frame() is None  # a stale frame is never shown as live
