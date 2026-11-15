"""On-demand camera trigger & frame capture manager for the UniX AI Panther robot.

Supports:
1. Live ROS 2 `/camera/color/image_raw` (`sensor_msgs/msg/Image`), `/camera/color/image_raw/compressed`,
   and `/camera/depth/image_raw` from the Panther's Orbbec Gemini 2 RGB-D camera.
2. Automatic on-demand Orbbec Gemini 2 driver startup (`ros2 launch orbbec_camera gemini2.launch.py`)
   when `/camera/color/image_raw` has no active publisher, plus automatic idle shutdown after
   `idle_timeout_seconds` (default 90s, matching `HEADBands-main/Detect/camera_stream.py`).
3. HTTP(S) JPEG snapshot / MJPEG / RTSP / V4L2 (`/dev/video4`) fallback matching `Detect/camera_stream.py`.
4. Lightweight synthetic workspace frame fallback when running in offline hardware-sim mode.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import signal
import struct
import subprocess
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np

from swiss_robot_bridge.panther_topics import (
    DEFAULT_ORBBEC_LAUNCH_CMD,
    ORBBEC_WS_SETUP_PATH,
)

logger = logging.getLogger("swiss_robot_bridge.camera")


@dataclass
class CapturedCameraFrame:
    """Encoded camera snapshot ready for transmission to Gemini Robotics ER 2."""

    image_bytes: bytes
    mime_type: str
    width: int
    height: int
    source: str
    captured_monotonic: float
    captured_at_utc: str
    center_depth_m: float | None = None
    driver_triggered: bool = False

    def to_ws_payload(self, request_id: str) -> dict[str, Any]:
        b64 = base64.b64encode(self.image_bytes).decode("ascii")
        return {
            "type": "camera_frame",
            "request_id": request_id,
            "jpeg_base64": b64,
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "source": self.source,
            "center_depth_m": self.center_depth_m,
            "driver_triggered": self.driver_triggered,
            "timestamp_utc": self.captured_at_utc,
        }


def _encode_png_stdlib(rgb: np.ndarray) -> bytes:
    """Zero-dependency pure-stdlib PNG encoder fallback when Pillow/OpenCV are absent."""
    height, width, _ = rgb.shape
    raw_rows = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", crc)

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(raw_rows, level=6)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def encode_rgb_frame(rgb: np.ndarray, max_width: int = 640, jpeg_quality: int = 82) -> tuple[bytes, str, int, int]:
    """Resize RGB uint8 frame to `<= max_width` and encode as JPEG (or PNG fallback)."""
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    rgb = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
    h, w, _ = rgb.shape

    if w > max_width and max_width > 0:
        stride = max(1, int(round(w / max_width)))
        rgb = np.ascontiguousarray(rgb[::stride, ::stride, :])
        h, w, _ = rgb.shape

    try:
        from PIL import Image

        pil_img = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        return buf.getvalue(), "image/jpeg", w, h
    except ImportError:
        pass

    try:
        import cv2

        bgr = rgb[:, :, ::-1]
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if ok:
            return enc.tobytes(), "image/jpeg", w, h
    except ImportError:
        pass

    return _encode_png_stdlib(rgb), "image/png", w, h


def decode_ros_image_to_rgb(msg: Any) -> np.ndarray:
    """Convert a `sensor_msgs/msg/Image` (`rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8`, `yuyv`) into uint8 RGB."""
    height = int(msg.height)
    width = int(msg.width)
    encoding = str(getattr(msg, "encoding", "rgb8")).lower()
    raw = bytes(msg.data)

    if height <= 0 or width <= 0 or not raw:
        raise ValueError("Empty ROS Image message")

    if encoding in {"rgb8", "8uc3"}:
        arr = np.frombuffer(raw, dtype=np.uint8)
        step = int(getattr(msg, "step", width * 3)) or (width * 3)
        rows = arr.reshape((height, step))[:, : width * 3]
        return rows.reshape((height, width, 3)).copy()

    if encoding == "bgr8":
        arr = np.frombuffer(raw, dtype=np.uint8)
        step = int(getattr(msg, "step", width * 3)) or (width * 3)
        rows = arr.reshape((height, step))[:, : width * 3]
        bgr = rows.reshape((height, width, 3))
        return bgr[:, :, ::-1].copy()

    if encoding == "rgba8":
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, -1))[:, : width * 4]
        return arr.reshape((height, width, 4))[:, :, :3].copy()

    if encoding == "bgra8":
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, -1))[:, : width * 4]
        bgra = arr.reshape((height, width, 4))
        return bgra[:, :, 2::-1].copy()

    if encoding in {"mono8", "8uc1"}:
        arr = np.frombuffer(raw, dtype=np.uint8)
        step = int(getattr(msg, "step", width)) or width
        gray = arr.reshape((height, step))[:, :width]
        return np.stack([gray, gray, gray], axis=-1)

    if encoding in {"yuyv", "yuv422_yuy2", "yuv422"}:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width // 2, 4))
        y0 = arr[:, :, 0].astype(np.float32)
        u = arr[:, :, 1].astype(np.float32) - 128.0
        y1 = arr[:, :, 2].astype(np.float32) - 0.0
        v = arr[:, :, 3].astype(np.float32) - 128.0

        y = np.empty((height, width), dtype=np.float32)
        y[:, 0::2] = y0
        y[:, 1::2] = y1
        u_full = np.repeat(u, 2, axis=1)
        v_full = np.repeat(v, 2, axis=1)

        r = np.clip(y + 1.402 * v_full, 0, 255).astype(np.uint8)
        g = np.clip(y - 0.344136 * u_full - 0.714136 * v_full, 0, 255).astype(np.uint8)
        b = np.clip(y + 1.772 * u_full, 0, 255).astype(np.uint8)
        return np.stack([r, g, b], axis=-1)

    raise ValueError(f"Unsupported ROS Image encoding: {encoding}")


def extract_center_depth_meters(msg: Any) -> float | None:
    """Compute median center-ROI depth in meters from `/camera/depth/image_raw` (`16UC1` mm or `32FC1` m)."""
    height = int(getattr(msg, "height", 0))
    width = int(getattr(msg, "width", 0))
    encoding = str(getattr(msg, "encoding", "")).upper()
    raw = bytes(getattr(msg, "data", b""))
    if height < 4 or width < 4 or not raw:
        return None

    y0, y1 = height // 3, (2 * height) // 3
    x0, x1 = width // 3, (2 * width) // 3

    if "16UC1" in encoding or "MONO16" in encoding:
        arr = np.frombuffer(raw, dtype=np.uint16).reshape((height, -1))[:, :width]
        roi = arr[y0:y1, x0:x1]
        valid = roi[(roi > 150) & (roi < 10000)]
        if valid.size == 0:
            return None
        return round(float(np.median(valid)) / 1000.0, 3)

    if "32FC1" in encoding:
        arr = np.frombuffer(raw, dtype=np.float32).reshape((height, -1))[:, :width]
        roi = arr[y0:y1, x0:x1]
        valid = roi[np.isfinite(roi) & (roi > 0.15) & (roi < 10.0)]
        if valid.size == 0:
            return None
        return round(float(np.median(valid)), 3)

    return None


def validate_camera_http_url(url: str) -> str:
    """Validate HTTP(S) snapshot URL (matching `HEADBands/perception`)."""
    cleaned = url.strip()
    parts = urlsplit(cleaned)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError("Camera snapshot URL must begin with http:// or https://")
    if parts.port is not None and not (1 <= parts.port <= 65535):
        raise ValueError("Camera snapshot port out of range")
    return cleaned


@dataclass
class GroundFrisbeeCandidate:
    """Classical CPU ground-frisbee candidate matching `HEADBands/perception/detector/__init__.py`."""

    x: int
    y: int
    w: int
    h: int
    cx: int
    cy: int
    area: float
    axes: tuple[int, int]
    angle: float
    score: float


def detect_ground_frisbee_candidates(
    rgb: np.ndarray,
    *,
    rgb_sum_min: int = 435,
    rgb_diff_max: int = 30,
    ground_start_pct: int = 25,
    min_area: int = 120,
    max_area: int = 120000,
) -> list[GroundFrisbeeCandidate]:
    """Detect ground frisbees using the CPU color+ellipse algorithm from `HEADBands/perception/detector`."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return []
    frame_h, frame_w, _ = rgb.shape
    roi_y = int(round(frame_h * (max(0, min(85, ground_start_pct)) / 100.0)))
    roi = rgb[roi_y:, :, :3].astype(np.int16)

    # 1. White/neutral frisbee mask from perception/detector/__init__.py: R+G+B >= 435 and max-min <= 30
    channel_sum = roi[:, :, 0] + roi[:, :, 1] + roi[:, :, 2]
    channel_diff = np.max(roi, axis=2) - np.min(roi, axis=2)
    white_mask = (channel_sum >= rgb_sum_min) & (channel_diff <= rgb_diff_max)

    # 2. Bright orange/colored frisbee disc mask (for standard orange Panther test frisbees)
    orange_mask = (roi[:, :, 0] >= 180) & (roi[:, :, 1] >= 65) & (roi[:, :, 1] <= 175) & (roi[:, :, 2] <= 80)
    combined_mask = (white_mask | orange_mask).astype(np.uint8) * 255

    try:
        import cv2

        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, open_k)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_k)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[GroundFrisbeeCandidate] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area or len(cnt) < 5:
                continue
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            (ecx, ecy), (ma1, ma2), angle = cv2.fitEllipse(cnt)
            major, minor = max(ma1, ma2), min(ma1, ma2)
            if major <= 1.0 or (minor / major) < 0.18:
                continue
            candidates.append(
                GroundFrisbeeCandidate(
                    x=int(rx),
                    y=int(ry + roi_y),
                    w=int(rw),
                    h=int(rh),
                    cx=int(round(ecx)),
                    cy=int(round(ecy + roi_y)),
                    area=round(area, 1),
                    axes=(max(1, int(round(major / 2.0))), max(1, int(round(minor / 2.0)))),
                    angle=round(float(angle), 1),
                    score=round(min(1.0, area / max(float(rw * rh), 1.0)), 3),
                )
            )
        candidates.sort(key=lambda c: (c.score, c.area), reverse=True)
        return candidates
    except ImportError:
        ys, xs = np.nonzero(combined_mask)
        if ys.size < min_area:
            return []
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()) + roi_y, int(ys.max()) + roi_y
        rw, rh = max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)
        return [
            GroundFrisbeeCandidate(
                x=x0,
                y=y0,
                w=rw,
                h=rh,
                cx=int(round(xs.mean())),
                cy=int(round(ys.mean())) + roi_y,
                area=float(ys.size),
                axes=(max(1, rw // 2), max(1, rh // 2)),
                angle=0.0,
                score=round(min(1.0, float(ys.size) / float(rw * rh)), 3),
            )
        ]


def build_synthetic_panther_frame(width: int = 320, height: int = 240) -> tuple[bytes, str, int, int]:
    """Generate a clean synthetic Panther workspace image when running without physical camera hardware."""
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    # Upper wall & floor perspective gradient
    horizon = height // 3
    rgb[:horizon, :] = (38, 44, 54)
    for row in range(horizon, height):
        shade = 72 + int(75 * ((row - horizon) / max(1, height - horizon)))
        rgb[row, :] = (shade, shade + 4, shade + 8)

    # Bright orange frisbee disc on the floor (center-right)
    yy, xx = np.ogrid[:height, :width]
    disc_mask = ((xx - int(width * 0.56)) / 30.0) ** 2 + ((yy - int(height * 0.68)) / 14.0) ** 2 <= 1.0
    rgb[disc_mask] = (242, 108, 34)

    # Blue water bottle standing on the floor (center-left)
    bx0, bx1 = int(width * 0.34), int(width * 0.40)
    by0, by1 = int(height * 0.46), int(height * 0.72)
    rgb[by0:by1, bx0:bx1] = (46, 134, 222)

    return encode_rgb_frame(rgb, max_width=width)


class PantherCameraManager:
    """Manages on-demand camera driver triggering, ROS 2 frame ingestion, and idle shutdown."""

    def __init__(
        self,
        *,
        camera_launch_cmd: str | None = None,
        snapshot_url: str | None = None,
        v4l2_or_stream_source: str | None = None,
        idle_timeout_seconds: float = 90.0,
        startup_timeout_seconds: float = 4.5,
        allow_sim_fallback: bool = False,
    ) -> None:
        env_cmd = os.getenv("PANTHER_CAMERA_LAUNCH_CMD", "").strip()
        if camera_launch_cmd is not None:
            self.camera_launch_cmd = camera_launch_cmd.strip()
        elif env_cmd:
            self.camera_launch_cmd = env_cmd
        elif Path(ORBBEC_WS_SETUP_PATH).is_file():
            self.camera_launch_cmd = DEFAULT_ORBBEC_LAUNCH_CMD
        else:
            self.camera_launch_cmd = ""

        raw_snap = (
            snapshot_url if snapshot_url is not None else os.getenv("PANTHER_CAMERA_SNAPSHOT_URL", "")
        ).strip()
        self.snapshot_url = validate_camera_http_url(raw_snap) if raw_snap else ""
        self.v4l2_or_stream_source = (
            v4l2_or_stream_source
            if v4l2_or_stream_source is not None
            else os.getenv("PANTHER_CAMERA_STREAM_URL", os.getenv("PANTHER_V4L2_DEVICE", ""))
        ).strip()

        self.idle_timeout_seconds = max(5.0, float(idle_timeout_seconds))
        self.startup_timeout_seconds = max(0.3, float(startup_timeout_seconds))
        self.allow_sim_fallback = allow_sim_fallback or (
            os.getenv("PANTHER_SIM_CAMERA", "").lower() in {"1", "true", "yes"}
        )

        self._latest_frame: CapturedCameraFrame | None = None
        self._latest_depth_m: float | None = None
        self._last_requested_monotonic: float | None = None
        self._driver_proc: subprocess.Popen[bytes] | None = None

    @property
    def driver_active(self) -> bool:
        return self._driver_proc is not None and self._driver_proc.poll() is None

    @property
    def latest_source(self) -> str | None:
        return self._latest_frame.source if self._latest_frame is not None else None

    def has_fresh_frame(self, max_age_seconds: float = 2.5, monotonic_now: float | None = None) -> bool:
        if self._latest_frame is None:
            return False
        ref = monotonic_now if monotonic_now is not None else time.monotonic()
        return (ref - self._latest_frame.captured_monotonic) <= max_age_seconds

    def ingest_ros_image(self, msg: Any, source: str = "/camera/color/image_raw") -> CapturedCameraFrame | None:
        """Decode incoming ROS 2 `sensor_msgs/msg/Image` into JPEG snapshot."""
        try:
            rgb = decode_ros_image_to_rgb(msg)
            img_bytes, mime, w, h = encode_rgb_frame(rgb)
            frame = CapturedCameraFrame(
                image_bytes=img_bytes,
                mime_type=mime,
                width=w,
                height=h,
                source=source,
                captured_monotonic=time.monotonic(),
                captured_at_utc=datetime.now(tz=UTC).isoformat(),
                center_depth_m=self._latest_depth_m,
                driver_triggered=self.driver_active,
            )
            self._latest_frame = frame
            return frame
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed decoding ROS camera frame: %s", exc)
            return None

    def ingest_ros_compressed_image(
        self,
        msg: Any,
        source: str = "/camera/color/image_raw/compressed",
    ) -> CapturedCameraFrame | None:
        raw = bytes(getattr(msg, "data", b""))
        if not raw:
            return None
        fmt = str(getattr(msg, "format", "jpeg")).lower()
        mime = "image/png" if "png" in fmt else "image/jpeg"
        frame = CapturedCameraFrame(
            image_bytes=raw,
            mime_type=mime,
            width=640,
            height=480,
            source=source,
            captured_monotonic=time.monotonic(),
            captured_at_utc=datetime.now(tz=UTC).isoformat(),
            center_depth_m=self._latest_depth_m,
            driver_triggered=self.driver_active,
        )
        self._latest_frame = frame
        return frame

    def ingest_ros_depth_image(self, msg: Any) -> None:
        depth_m = extract_center_depth_meters(msg)
        if depth_m is not None:
            self._latest_depth_m = depth_m

    def start_driver_if_configured(self) -> bool:
        """Launch Orbbec Gemini 2 ROS 2 driver subprocess on demand if not already running."""
        if not self.camera_launch_cmd:
            return False
        if self.driver_active:
            return True
        try:
            kwargs: dict[str, Any] = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
            if os.name != "nt":
                kwargs["start_new_session"] = True
            self._driver_proc = subprocess.Popen(
                self.camera_launch_cmd,
                shell=True,
                **kwargs,
            )
            logger.info("Triggered on-demand Panther camera driver: %s", self.camera_launch_cmd)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed launching on-demand camera driver: %s", exc)
            self._driver_proc = None
            return False

    def stop_driver(self) -> None:
        """Terminate on-demand camera driver subprocess cleanly when idle or shutting down."""
        proc = self._driver_proc
        self._driver_proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        logger.info("Stopped idle on-demand Panther camera driver")

    def check_idle_shutdown(
        self,
        monotonic_now: float,
        deactivate_ros_camera_cb: Callable[[], None] | None = None,
    ) -> bool:
        """Stop camera subscriptions & driver subprocess after `idle_timeout_seconds` without vision requests."""
        if self._last_requested_monotonic is None:
            return False
        if (monotonic_now - self._last_requested_monotonic) <= self.idle_timeout_seconds:
            return False

        self._last_requested_monotonic = None
        if deactivate_ros_camera_cb is not None:
            deactivate_ros_camera_cb()
        if self.driver_active:
            self.stop_driver()
        return True

    def _capture_http_snapshot_sync(self) -> CapturedCameraFrame | None:
        if not self.snapshot_url:
            return None
        req = Request(
            self.snapshot_url,
            headers={"User-Agent": "PantherSafetyBridge/1.0", "Cache-Control": "no-cache"},
        )
        with urlopen(req, timeout=3.5) as resp:
            content = resp.read(10 * 1024 * 1024 + 1)
        if not content or len(content) > 10 * 1024 * 1024:
            return None
        mime = "image/png" if content.startswith(b"\x89PNG") else "image/jpeg"
        frame = CapturedCameraFrame(
            image_bytes=content,
            mime_type=mime,
            width=640,
            height=480,
            source="http_snapshot",
            captured_monotonic=time.monotonic(),
            captured_at_utc=datetime.now(tz=UTC).isoformat(),
            center_depth_m=self._latest_depth_m,
        )
        self._latest_frame = frame
        return frame

    def _capture_v4l2_or_stream_sync(self) -> CapturedCameraFrame | None:
        if not self.v4l2_or_stream_source:
            return None
        try:
            import cv2

            src: int | str = (
                int(self.v4l2_or_stream_source)
                if self.v4l2_or_stream_source.isdigit()
                else self.v4l2_or_stream_source
            )
            cap = cv2.VideoCapture(src)
            try:
                if not cap.isOpened():
                    return None
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    return None
                rgb = bgr[:, :, ::-1]
                img_bytes, mime, w, h = encode_rgb_frame(rgb)
                frame = CapturedCameraFrame(
                    image_bytes=img_bytes,
                    mime_type=mime,
                    width=w,
                    height=h,
                    source=f"video:{self.v4l2_or_stream_source}",
                    captured_monotonic=time.monotonic(),
                    captured_at_utc=datetime.now(tz=UTC).isoformat(),
                    center_depth_m=self._latest_depth_m,
                )
                self._latest_frame = frame
                return frame
            finally:
                cap.release()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fallback stream/V4L2 capture failed: %s", exc)
            return None

    async def capture_frame_on_demand(
        self,
        *,
        max_age_seconds: float = 1.5,
        activate_ros_camera_cb: Callable[[], None] | None = None,
        spin_ros_once_cb: Callable[[], None] | None = None,
        ros_color_publisher_count_cb: Callable[[], int] | None = None,
    ) -> CapturedCameraFrame | None:
        """Trigger camera only when necessary and return a fresh encoded snapshot."""
        now_mono = time.monotonic()
        self._last_requested_monotonic = now_mono

        if activate_ros_camera_cb is not None:
            activate_ros_camera_cb()

        # 1. Return immediately if a fresh ROS 2 frame is already buffered
        if self.has_fresh_frame(max_age_seconds=max_age_seconds, monotonic_now=now_mono):
            return self._latest_frame

        # 2. Spin ROS briefly in case /camera/color/image_raw is already publishing
        if spin_ros_once_cb is not None:
            probe_deadline = time.monotonic() + 0.35
            while time.monotonic() < probe_deadline:
                spin_ros_once_cb()
                if self.has_fresh_frame(max_age_seconds=max_age_seconds):
                    return self._latest_frame
                await asyncio.sleep(0.05)

        # 3. If /camera/color/image_raw has no active publisher or no fresh frame, launch Orbbec driver on demand
        pub_count = ros_color_publisher_count_cb() if ros_color_publisher_count_cb is not None else 0
        if pub_count == 0 or not self.has_fresh_frame(max_age_seconds=max_age_seconds):
            launched = self.start_driver_if_configured()
            if launched and spin_ros_once_cb is not None:
                wait_deadline = time.monotonic() + self.startup_timeout_seconds
                while time.monotonic() < wait_deadline:
                    spin_ros_once_cb()
                    if self.has_fresh_frame(max_age_seconds=max_age_seconds):
                        assert self._latest_frame is not None
                        self._latest_frame.driver_triggered = True
                        return self._latest_frame
                    await asyncio.sleep(0.08)

        # 4. Fallback to HTTP snapshot or RTSP/MJPEG/V4L2 (`/dev/video4`) stream
        if self.snapshot_url:
            snap = await asyncio.to_thread(self._capture_http_snapshot_sync)
            if snap is not None:
                return snap

        if self.v4l2_or_stream_source:
            vid_frame = await asyncio.to_thread(self._capture_v4l2_or_stream_sync)
            if vid_frame is not None:
                return vid_frame

        # 5. Offline simulation camera frame fallback
        if self.allow_sim_fallback:
            img_bytes, mime, w, h = build_synthetic_panther_frame()
            sim_frame = CapturedCameraFrame(
                image_bytes=img_bytes,
                mime_type=mime,
                width=w,
                height=h,
                source="panther_sim_camera",
                captured_monotonic=time.monotonic(),
                captured_at_utc=datetime.now(tz=UTC).isoformat(),
                center_depth_m=1.15,
                driver_triggered=True,
            )
            self._latest_frame = sim_frame
            return sim_frame

        return None
