"""Read an RTSP/MJPEG URL or repeatedly fetch a JPEG snapshot in a background thread.

Only the most recent decoded frame is retained: slow inference cannot build up a
queue of old frames. This module deliberately has no Streamlit dependency so it
can be unit-tested and reused from a command-line program.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import cv2
import numpy as np

VIDEO_STREAM = "MJPEG / RTSP video stream"
SNAPSHOT = "HTTP JPEG snapshot (polling)"
STREAM_MODES = (VIDEO_STREAM, SNAPSHOT)


def validate_source(url: str, mode: str) -> str:
    """Accept explicit HTTP(S) / RTSP camera URLs, never a viewer HTML page."""
    url = url.strip()
    parts = urlsplit(url)
    if mode not in STREAM_MODES:
        raise ValueError("Unsupported input mode")
    allowed = ("http", "https") if mode == SNAPSHOT else ("http", "https", "rtsp", "rtsps")
    if parts.scheme.lower() not in allowed or not parts.hostname:
        raise ValueError(f"Enter a complete camera URL beginning with {', '.join(x + '://' for x in allowed)}")
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("Invalid camera port") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("Camera port must be between 1 and 65535")
    return url


def safe_url_label(url: str) -> str:
    """Display target without password or query tokens."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.hostname or '?'}:{parts.port or 'default'}{parts.path or '/'}"


@dataclass(frozen=True)
class FrameSnapshot:
    frame: np.ndarray | None
    sequence: int
    age_seconds: float | None
    capture_fps: float
    status: str
    error: str | None


class LatestFrameCamera:
    """Read camera in a daemon thread; Streamlit polls .get_latest().

    The reader stops after a period without client polling, or when .stop() is
    called. A failed remote stream automatically retries after a short delay.
    """

    def __init__(
        self,
        url: str,
        mode: str = VIDEO_STREAM,
        *,
        retry_seconds: float = 1.5,
        snapshot_interval: float = 0.2,
        idle_timeout: float = 90.0,
        request_timeout: float = 4.0,
    ) -> None:
        self.url = validate_source(url, mode)
        self.mode = mode
        self.retry_seconds = retry_seconds
        self.snapshot_interval = max(0.05, snapshot_interval)
        self.idle_timeout = idle_timeout
        self.request_timeout = request_timeout
        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._latest_frame: np.ndarray | None = None
        self._sequence = 0
        self._last_frame_at: float | None = None
        self._last_polled = monotonic()
        self._frame_times: deque[float] = deque(maxlen=40)
        self._status = "Not started"
        self._error: str | None = None

    def start(self) -> "LatestFrameCamera":
        if self._thread is None:
            self._thread = Thread(target=self._loop, name="frisbee-camera-reader", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        # Avoid waiting for a slow remote server on the UI thread.
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.15)
        with self._lock:
            self._status = "Stopped"

    def get_latest(self) -> FrameSnapshot:
        now = monotonic()
        with self._lock:
            self._last_polled = now
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
            age = now - self._last_frame_at if self._last_frame_at is not None else None
            fps = 0.0
            if len(self._frame_times) > 1:
                dt = self._frame_times[-1] - self._frame_times[0]
                if dt > 0:
                    fps = (len(self._frame_times) - 1) / dt
            return FrameSnapshot(frame, self._sequence, age, fps, self._status, self._error)

    def _set_status(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self._status = status
            self._error = error

    def _put_frame(self, frame: np.ndarray) -> None:
        now = monotonic()
        with self._lock:
            self._latest_frame = frame.copy()
            self._sequence += 1
            self._last_frame_at = now
            self._frame_times.append(now)
            self._status = "Receiving frames"
            self._error = None

    def _inactive(self) -> bool:
        with self._lock:
            return monotonic() - self._last_polled > self.idle_timeout

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                if self._inactive():
                    self._set_status("Stopped (viewer inactive)")
                    break
                if self.mode == SNAPSHOT:
                    self._read_snapshot_once()
                    self._stop.wait(self.snapshot_interval)
                else:
                    self._read_video_stream()
                    if not self._stop.is_set():
                        self._stop.wait(self.retry_seconds)
        except Exception as exc:
            # Do not surface full URLs or credentials in exception messages.
            self._set_status("Stream error", type(exc).__name__)

    def _read_snapshot_once(self) -> None:
        self._set_status("Connecting / polling snapshot")
        try:
            req = Request(self.url, headers={"User-Agent": "FrisbeeGroundFinder/1.0", "Cache-Control": "no-cache"})
            with urlopen(req, timeout=self.request_timeout) as response:
                content = response.read(12 * 1024 * 1024 + 1)
            if len(content) > 12 * 1024 * 1024:
                raise ValueError("Image is too large (>12 MB)")
            frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Endpoint did not return a JPEG/PNG image; use video mode for MJPEG")
            self._put_frame(frame)
        except Exception as exc:
            self._set_status("Snapshot connection failed", type(exc).__name__)

    def _open_video(self) -> cv2.VideoCapture:
        # Timeout props are supported by OpenCV's FFmpeg backend; fall back to
        # the default decoder on installations without this support.
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 4000]
        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
            if cap.isOpened():
                return cap
            cap.release()
        except (cv2.error, TypeError):
            pass
        return cv2.VideoCapture(self.url)

    def _read_video_stream(self) -> None:
        self._set_status("Connecting to video stream")
        cap = self._open_video()
        try:
            if not cap.isOpened():
                self._set_status("Video connection failed", "Check camera IP, port, stream path, and network access")
                return
            while not self._stop.is_set():
                if self._inactive():
                    self._set_status("Stopped (viewer inactive)")
                    return
                ok, frame = cap.read()
                if not ok or frame is None:
                    self._set_status("Video stream interrupted", "Trying to reconnect")
                    return
                self._put_frame(frame)
        finally:
            cap.release()
