"""Frisbee navigation (src/base_navigation) inside the bridge, for the Web UI's Navigation page.

The navigation dashboard server runs here on loopback, unchanged; the cloud forwards the page's requests to it
over the bridge WebSocket (`navigation_http`). Base velocity goes through the bridge's safety supervisor
(the 20 Hz loop publishes /cmd_vel); lift and head use navigation's own ROS node and head convention.
"""

from __future__ import annotations

import base64
import logging
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("swiss_robot_bridge.navigation")

# The dashboard's own request paths; anything else is refused before it reaches the robot.
ALLOWED_REQUESTS = frozenset(
    [("GET", p) for p in ("/settings", "/settings/saved", "/robot/status", "/frame.jpg")]
    + [
        ("POST", p)
        for p in (
            "/settings", "/settings/save", "/settings/reload", "/disconnect",
            "/robot/connect", "/robot/start", "/robot/stop", "/robot/disconnect", "/robot/heartbeat",
            "/robot/lift/up", "/robot/lift/down", "/robot/head/up", "/robot/head/down",
        )
    ]
)
MAX_BODY_BYTES = 8192
TWIST_MAX_AGE_S = 0.2  # navigation refreshes every 0.05 s; older means its loop stopped


def _bridge_robot_factory(holder: "NavigationMode"):
    def factory():
        # Loaded on Connect ROS, as in the dashboard: camera preview works without ROS.
        from rclpy.executors import SingleThreadedExecutor
        from base_navigation.navigation import BaseController, RobotNavigation

        class BridgeRobotNavigation(RobotNavigation):
            """rclpy is already running in the bridge, and only the bridge publishes /cmd_vel."""

            def __enter__(self):
                self._node = BaseController()  # its cmd_vel publisher is only used to count subscribers
                self._executor = SingleThreadedExecutor()
                self._executor.add_node(self._node)
                holder.robot = self
                return self

            def set_velocity(self, linear_speed=0.0, angular_speed=0.0):
                connected = self.connected
                holder.set_twist(linear_speed if connected else 0.0, angular_speed if connected else 0.0)
                return connected

            def wait(self, seconds=0.05):
                deadline = time.monotonic() + seconds
                while self.running and (left := deadline - time.monotonic()) > 0:
                    self._executor.spin_once(timeout_sec=left)

            def __exit__(self, exc_type, exc_value, traceback):
                holder.set_twist(0.0, 0.0)
                holder.robot = None
                self._node._timer.cancel()
                try:
                    self._node.stop_lift()
                    self._node.stop_head()
                finally:
                    self._executor.shutdown()
                    self._node.destroy_node()  # never rclpy.shutdown(): the bridge keeps running

        return BridgeRobotNavigation()

    return factory


class NavigationMode:
    def __init__(self, repo_root: Path) -> None:
        sys.path[:0] = [str(repo_root / "src"), str(repo_root / "perception")]
        from base_navigation.config import parse_args
        from base_navigation.dashboard import run_dashboard
        from base_navigation.dashboard_server import DashboardServer
        from base_navigation.web_control import WebRobotControl

        self.robot = None
        self._lock = threading.Lock()
        self._twist = (0.0, 0.0, 0.0)  # linear m/s, angular rad/s, monotonic time
        self._driving = False
        args = parse_args([], dashboard=True)  # settings.json, as the standalone dashboard
        control = WebRobotControl(robot_factory=_bridge_robot_factory(self))
        # Port 0: never collides with a standalone dashboard; saving keeps the file's port.
        self._server = DashboardServer(0, settings=args, robot_control=control)
        self.control = control
        threading.Thread(
            target=run_dashboard, args=(self._server, args), kwargs={"robot_control": control},
            daemon=True, name="navigation-camera",
        ).start()

    def set_twist(self, linear: float, angular: float) -> None:
        with self._lock:
            self._twist = (float(linear), float(angular), time.monotonic())

    def take_twist(self, now: float, max_linear: float, max_angular: float) -> tuple[float, float, float] | None:
        """Base command for the bridge's 20 Hz loop: a twist while navigating, one zero when it stops, else None."""
        with self._lock:
            linear, angular, at = self._twist
        fresh = now - at <= TWIST_MAX_AGE_S
        linear = min(max(linear, -max_linear), max_linear) if fresh else 0.0  # navigation may back off
        angular = min(max(angular, -max_angular), max_angular) if fresh else 0.0
        if linear or angular:
            self._driving = True
            return (linear, 0.0, angular)
        if self._driving:
            self._driving = False
            return (0.0, 0.0, 0.0)  # releases the base; idle ticks leave manual jogs alone
        return None

    def camera_frame(self, max_age_s: float = 2.0):
        """The navigation camera's latest upright frame as a Web UI camera snapshot, or None when stale."""
        from datetime import datetime, timezone

        from swiss_robot_bridge.camera_manager import CapturedCameraFrame, encode_rgb_frame

        latest = self._server.camera_frame
        if latest is None or time.monotonic() - latest[1] > max_age_s:
            return None
        frame, at = latest
        data, mime, width, height = encode_rgb_frame(frame[:, :, ::-1])  # OpenCV frames are BGR
        return CapturedCameraFrame(
            image_bytes=data, mime_type=mime, width=width, height=height, source="navigation_camera",
            captured_monotonic=at, captured_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        )

    def stop(self, reason: str) -> None:
        self.control.stop(reason)
        self.set_twist(0.0, 0.0)

    def forward(self, method: str, path: str, body: str | None) -> dict:
        """Replay one Web UI request on the local dashboard server; JSON-safe reply for the WebSocket."""
        if (method, path) not in ALLOWED_REQUESTS:
            return _reply(404, "application/json", b'{"error": "Unknown navigation request"}')
        data = (body or "").encode()
        if len(data) > MAX_BODY_BYTES:
            return _reply(413, "application/json", b'{"error": "Navigation request too large"}')
        request = urllib.request.Request(
            self._server.url + path,
            data=data if method == "POST" else None,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=4.0) as response:
                return _reply(response.status, response.headers.get_content_type(), response.read())
        except urllib.error.HTTPError as exc:
            return _reply(exc.code, exc.headers.get_content_type(), exc.read())
        except (urllib.error.URLError, OSError) as exc:
            return _reply(503, "application/json", f'{{"error": "Navigation server unavailable: {exc}"}}'.encode())

    def close(self) -> None:
        self._server.close()


def _reply(status: int, content_type: str, content: bytes) -> dict:
    return {"status": status, "content_type": content_type, "body_b64": base64.b64encode(content).decode()}


def start_navigation() -> NavigationMode | None:
    """None (logged) when navigation cannot load, e.g. no OpenCV: the rest of the bridge runs as before."""
    try:
        return NavigationMode(Path(__file__).resolve().parents[6])  # the repository root (docker/run.sh mounts it)
    except (Exception, SystemExit) as exc:  # noqa: BLE001  (an invalid settings.json exits argparse)
        logger.warning("Frisbee navigation unavailable (%s); the Navigation page will report it", exc)
        return None
