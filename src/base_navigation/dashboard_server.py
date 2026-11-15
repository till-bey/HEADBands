"""Local browser display for the navigation dashboard image."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock, Thread

import cv2

if __package__:
    from .config import SETTINGS_FIELDS, DEFAULT_SETTINGS_FILE, validate_settings, load_settings, save_settings
    from .web_control import WebRobotControl
else:
    from config import SETTINGS_FIELDS, DEFAULT_SETTINGS_FILE, validate_settings, load_settings, save_settings
    from web_control import WebRobotControl


class DashboardServer:
    def __init__(self, port=8765, *, settings=None, robot_control=None):
        self.robot_control = robot_control if robot_control is not None else WebRobotControl()
        self._jpeg = None
        self._lock = Lock()
        self._settings = {name: getattr(settings, name) for name in SETTINGS_FIELDS} if settings else {}
        self._settings_file = Path(getattr(settings, "settings_file", DEFAULT_SETTINGS_FILE))
        self._settings_port = getattr(settings, "port", port)
        self._pending = ("connect", settings) if settings and settings.url else None
        view = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/":
                    content = Path(__file__).with_name("dashboard.html").read_bytes()
                    content_type = "text/html; charset=utf-8"
                elif path == "/settings":
                    with view._lock:
                        content = json.dumps(view._settings, allow_nan=False).encode()
                    content_type = "application/json"
                elif path == "/robot/status":
                    content = json.dumps(view.robot_control.status(), allow_nan=False).encode()
                    content_type = "application/json"
                elif path == "/frame.jpg":
                    with view._lock:
                        content = view._jpeg
                    if content is None:
                        self.send_error(503, "Waiting for first frame")
                        return
                    content_type = "image/jpeg"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                try:
                    self.wfile.write(content)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # A browser may leave while an image is being served.

            def do_POST(self):
                robot_actions = {
                    "/robot/connect": view.robot_control.connect,
                    "/robot/start": view.robot_control.start,
                    "/robot/stop": view.robot_control.stop,
                    "/robot/disconnect": view.robot_control.disconnect,
                    "/robot/heartbeat": view.robot_control.heartbeat,
                    "/robot/lift/up": lambda **kw: view.robot_control.jog_lift("up", **kw),
                    "/robot/lift/down": lambda **kw: view.robot_control.jog_lift("down", **kw),
                }
                if self.path in robot_actions:
                    # Only the local JSON UI may arm motion; reject cross-site forms.
                    allowed_origins = {view.url, f"http://localhost:{view._server.server_port}"}
                    if self.headers.get("Origin") not in (None, *allowed_origins):
                        self.reply_json(403, {"error": "Robot commands must come from this local page"})
                        return
                    if self.headers.get_content_type() != "application/json":
                        self.reply_json(415, {"error": "Robot commands require application/json"})
                        return
                    requires_generation = self.path == "/robot/start" or self.path.startswith("/robot/lift/")
                    if requires_generation:
                        try:
                            length = int(self.headers.get("Content-Length", "0"))
                            if not 0 < length <= 512:
                                raise ValueError("Motion commands require a current generation token")
                            data = json.loads(self.rfile.read(length))
                            generation = data.get("generation") if isinstance(data, dict) else None
                            if isinstance(generation, bool) or not isinstance(generation, int):
                                raise ValueError("Motion commands require a current generation token")
                        except (ValueError, TypeError, OverflowError) as exc:
                            self.reply_json(400, {"error": str(exc)})
                            return
                    try:
                        message = (robot_actions[self.path](expected_generation=generation)
                                   if requires_generation else robot_actions[self.path]())
                    except ValueError as exc:
                        self.reply_json(409, {"error": str(exc)})
                        return
                    self.reply_json(200, {"message": message})
                    return
                if self.path == "/disconnect":
                    with view._lock:
                        view._pending = ("disconnect", None)
                    view.robot_control.clear_target("Camera disconnected; navigation stopped")
                    self.reply_json(200, {"message": "Disconnect requested"})
                    return
                if self.path == "/settings/reload":
                    try:
                        saved = load_settings(view._settings_file)
                    except (OSError, ValueError, TypeError, OverflowError) as exc:
                        self.reply_json(400, {"error": f"Could not reload saved settings: {exc}"})
                        return
                    with view._lock:
                        view._settings = {name: getattr(saved, name) for name in SETTINGS_FIELDS}
                        view._settings_port = saved.port
                        values = view._settings.copy()
                    self.reply_json(200, {"message": "Saved values reloaded.",
                                          "settings": values})
                    return
                if self.path not in ("/settings", "/settings/save"):
                    self.send_error(404)
                    return
                saving = self.path == "/settings/save"
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 8192:
                        raise ValueError("Invalid settings request size")
                    settings = validate_settings(json.loads(self.rfile.read(length)), allow_empty_url=saving)
                except (ValueError, TypeError, OverflowError) as exc:
                    self.reply_json(400, {"error": str(exc)})
                    return
                if saving:
                    try:
                        with view._lock:
                            port = view._settings_port
                        save_settings(view._settings_file, vars(settings), port=port)
                    except OSError as exc:
                        self.reply_json(500, {"error": f"Could not save settings: {exc}"})
                        return
                    self.reply_json(200, {"message": "Settings saved to JSON. Click Connect / apply settings to use them."})
                    return
                with view._lock:
                    view._settings = vars(settings).copy()
                    view._pending = ("connect", settings)
                view.robot_control.clear_target("Camera settings changed; navigation stopped")
                self.reply_json(200, {"message": "Settings accepted; connecting camera"})

            def reply_json(self, status, data):
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        print(f"Open {self.url} for navigation dashboard. Stop with Ctrl+C.", flush=True)

    def take_update(self):
        """Consume the latest request; only the dashboard loop touches the camera."""
        with self._lock:
            update, self._pending = self._pending, None
            return update

    def show(self, frame):
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError("Could not encode dashboard frame as JPEG")
        with self._lock:
            self._jpeg = encoded.tobytes()

    def close(self):
        self.robot_control.close()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1)
