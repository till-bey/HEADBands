"""Return a white-disc center from an MJPEG/RTSP or HTTP snapshot URL.

Run: python -m camera 'http://192.168.1.50:8080/stream.mjpg'
Captured landscape frames rotate clockwise to portrait before analysis.
Coordinates are pixels in that upright image, with (0, 0) at top left.
"""
from __future__ import annotations

import argparse
import json
import math
from math import isfinite
from time import monotonic, sleep

import numpy as np
import cv2

from .camera_stream import LatestFrameCamera, SNAPSHOT, VIDEO_STREAM
from detector import DetectorSettings, find_ellipses, prepare_frame, preprocess


def pixel_to_ground(
    px, py,
    image_width, image_height,
    camera_height,
    tilt_deg,
    horizontal_fov_deg,
    vertical_fov_deg=None,
):
    """Intersect a pixel ray with flat ground (no roll, undistorted image).

    Tilt is downward from horizontal. Pixel origin is top-left, Y downward.
    Ground origin is below the camera: X right, Y forward. Distance units match
    camera_height; bearing is degrees, 0 forward and positive toward the right.
    Omitting vertical FOV assumes square pixels. image_angle_rad/deg measure
    horizontal image offset from the optical axis, independent of ground tilt.
    """
    if image_width <= 0 or image_height <= 0 or camera_height <= 0:
        raise ValueError("Image dimensions and camera height must be positive.")
    if not 0 < horizontal_fov_deg < 180:
        raise ValueError("Horizontal FOV must be between 0 and 180 degrees.")
    if vertical_fov_deg is not None and not 0 < vertical_fov_deg < 180:
        raise ValueError("Vertical FOV must be between 0 and 180 degrees.")

    alpha = math.radians(tilt_deg)
    fx = image_width / (2 * math.tan(math.radians(horizontal_fov_deg) / 2))
    fy = (
        fx if vertical_fov_deg is None
        else image_height / (2 * math.tan(math.radians(vertical_fov_deg) / 2))
    )
    cx = image_width / 2
    cy = image_height / 2
    u = (px - cx) / fx
    v = (py - cy) / fy
    denominator = math.sin(alpha) + v * math.cos(alpha)
    if denominator <= 1e-12:
        raise ValueError("Pixel ray does not intersect the ground ahead.")
    x = camera_height * u / denominator
    y = camera_height * (math.cos(alpha) - v * math.sin(alpha)) / denominator
    return {
        "distance": math.hypot(x, y),
        "bearing_deg": math.degrees(math.atan2(x, y)),
        "image_angle_rad": math.atan(u),
        "image_angle_deg": math.degrees(math.atan(u)),
        "x": x,
        "y": y,
    }


CAMERA_BASE_HEIGHT_M = 0.83


def _robot_camera_pose(robot_z, tilt_rad, camera_base_height_m=CAMERA_BASE_HEIGHT_M):
    """Convert robot lift and tilt to camera height (m) and tilt (degrees)."""
    if not isfinite(robot_z) or robot_z < 0:
        raise ValueError("Robot lift must be a finite nonnegative number of meters.")
    if not isfinite(tilt_rad):
        raise ValueError("Robot tilt must be a finite angle in radians.")
    if not isfinite(camera_base_height_m) or camera_base_height_m <= 0:
        raise ValueError("Camera base height must be finite and positive.")
    return camera_base_height_m + robot_z, math.degrees(tilt_rad)


def pixel_to_ground_robot(
    px, py, image_width, image_height,
    robot_z, tilt_rad, horizontal_fov_deg, vertical_fov_deg=None,
    *, camera_base_height_m=CAMERA_BASE_HEIGHT_M,
):
    """Ground position for lift above a configurable camera base (default 0.83 m).

    Robot ground X/Y are fixed at zero. Tilt is radians downward from horizontal;
    FOV remains degrees. Adds bearing_rad while retaining the original outputs.
    """
    camera_height, tilt_deg = _robot_camera_pose(robot_z, tilt_rad, camera_base_height_m)
    result = pixel_to_ground(
        px, py, image_width, image_height, camera_height,
        tilt_deg, horizontal_fov_deg, vertical_fov_deg,
    )
    return {**result, "bearing_rad": math.radians(result["bearing_deg"])}


class FrisbeeCamera:
    """Find the strongest white-disc candidate in each fresh camera frame.

    Uses the same RGB white thresholds, mask cleanup, ellipse detector, and
    processing resolution as the website's white detector. Settings are passed
    explicitly; website slider values are not saved or imported automatically.
    Captured frames are portrait by default; use portrait=False when a caller
    handles orientation itself. Explicit frames passed to detect_* stay as given.
    Use as a context manager, or call close() when finished. Not thread-safe.
    """

    def __init__(
        self,
        url: str,
        settings: DetectorSettings | None = None,
        *,
        mode: str = VIDEO_STREAM,
        portrait: bool = True,
    ) -> None:
        self.settings = settings if settings is not None else DetectorSettings()
        self._camera = LatestFrameCamera(url, mode)
        self._portrait = portrait
        self._last_sequence = 0
        self._closed = False

    def detect_center(self, frame: np.ndarray) -> tuple[int, int] | None:
        """Detect in an existing BGR frame; return original-image (x, y) or None.

        The strongest match wins if several frisbees are visible. Detection
        thresholds and radius limits apply to the selected processing resolution.
        """
        prepared = prepare_frame(frame, self.settings.processing_width)
        _, white_mask = preprocess(prepared, self.settings)
        candidates = find_ellipses(prepared, white_mask, self.settings)
        if not candidates:
            return None
        height, width = frame.shape[:2]
        target_height, target_width = prepared.shape[:2]
        scale = min(target_width / width, target_height / height)
        resized_width = max(1, min(target_width, round(width * scale)))
        resized_height = max(1, min(target_height, round(height * scale)))
        left = (target_width - resized_width) // 2
        top = (target_height - resized_height) // 2
        for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
            cx, cy = candidate.center
            if left <= cx < left + resized_width and top <= cy < top + resized_height:
                x = round((cx - left) * width / resized_width)
                y = round((cy - top) * height / resized_height)
                return min(width - 1, max(0, x)), min(height - 1, max(0, y))
        return None

    def get_center(self, timeout: float = 5.0) -> tuple[int, int] | None:
        """Wait for a fresh frame, then return its detected center (or None).

        Raises TimeoutError if no new frame captured within the last second is
        available before timeout. The timeout covers frame acquisition only.
        Repeated calls never process the same captured frame twice.
        """
        return self.detect_center(self._get_frame(timeout))

    def read_frame(self, timeout: float = 5.0) -> np.ndarray:
        """Return a fresh BGR frame, rotated upright to portrait by default.

        Uses the same freshness, timeout, and close checks as get_center().
        Landscape input rotates clockwise; already-portrait input stays upright.
        Reading a frame consumes it for subsequent camera calls.
        """
        return self._get_frame(timeout)

    def detect_ground_position(
        self,
        frame: np.ndarray,
        *,
        camera_height: float,
        tilt_deg: float,
        horizontal_fov_deg: float,
        vertical_fov_deg: float | None = None,
    ) -> dict[str, float] | None:
        """Return distance/bearing/X/Y for a detected center, or None if absent.

        Uses the original frame dimensions, before detector resizing/padding.
        Assumes flat ground, no roll, and an undistorted image. A detected ray
        on/above the horizon raises ValueError, as in pixel_to_ground().
        """
        center = self.detect_center(frame)
        if center is None:
            return None
        height, width = frame.shape[:2]
        return pixel_to_ground(
            *center, width, height, camera_height, tilt_deg,
            horizontal_fov_deg, vertical_fov_deg,
        )

    def get_ground_position(
        self,
        *,
        camera_height: float,
        tilt_deg: float,
        horizontal_fov_deg: float,
        vertical_fov_deg: float | None = None,
        timeout: float = 5.0,
    ) -> dict[str, float] | None:
        """Read one fresh frame and project its center onto the ground.

        Returns None if no frisbee is detected. Camera timeouts raise
        TimeoutError; a ray that cannot intersect ground raises ValueError.
        """
        return self.detect_ground_position(
            self._get_frame(timeout),
            camera_height=camera_height,
            tilt_deg=tilt_deg,
            horizontal_fov_deg=horizontal_fov_deg,
            vertical_fov_deg=vertical_fov_deg,
        )

    def detect_robot_ground_position(
        self, frame: np.ndarray, *, robot_z: float, tilt_rad: float,
        horizontal_fov_deg: float, vertical_fov_deg: float | None = None,
        camera_base_height_m: float = CAMERA_BASE_HEIGHT_M,
    ) -> dict[str, float] | None:
        """Wrap detect_ground_position with robot height and radian angles."""
        camera_height, tilt_deg = _robot_camera_pose(robot_z, tilt_rad, camera_base_height_m)
        result = self.detect_ground_position(
            frame, camera_height=camera_height, tilt_deg=tilt_deg,
            horizontal_fov_deg=horizontal_fov_deg,
            vertical_fov_deg=vertical_fov_deg,
        )
        return None if result is None else {
            **result, "bearing_rad": math.radians(result["bearing_deg"]),
        }

    def get_robot_ground_position(
        self, *, robot_z: float, tilt_rad: float, horizontal_fov_deg: float,
        vertical_fov_deg: float | None = None, timeout: float = 5.0,
        camera_base_height_m: float = CAMERA_BASE_HEIGHT_M,
    ) -> dict[str, float] | None:
        """Read a fresh frame and calculate position from robot lift and radians."""
        return self.detect_robot_ground_position(
            self._get_frame(timeout), robot_z=robot_z, tilt_rad=tilt_rad,
            horizontal_fov_deg=horizontal_fov_deg, vertical_fov_deg=vertical_fov_deg,
            camera_base_height_m=camera_base_height_m,
        )

    def _get_frame(self, timeout: float) -> np.ndarray:
        """Share fresh-frame acquisition between pixel and ground methods."""
        if self._closed:
            raise RuntimeError("FrisbeeCamera is closed")
        if not isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a finite positive number")
        self._camera.start()
        deadline = monotonic() + timeout
        while True:
            latest = self._camera.get_latest()
            if (latest.frame is not None and latest.sequence > self._last_sequence
                    and latest.age_seconds is not None and latest.age_seconds <= 1.0):
                self._last_sequence = latest.sequence
                frame = latest.frame
                if self._portrait and frame.shape[1] > frame.shape[0]:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                return frame
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("No fresh camera frame arrived before the timeout")
            sleep(min(0.02, remaining))

    def close(self) -> None:
        """Stop the camera reader; safe to call more than once."""
        if not self._closed:
            self._camera.stop()
            self._closed = True

    def __enter__(self) -> FrisbeeCamera:
        if self._closed:
            raise RuntimeError("FrisbeeCamera is closed")
        self._camera.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Direct camera stream URL")
    parser.add_argument("--snapshot", action="store_true", help="URL returns a JPEG/PNG snapshot")
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for a fresh frame")
    parser.add_argument("--resolution", type=int, choices=[320, 480, 640, 960, 1280], default=640,
                        help="Processing landscape width; portrait swaps dimensions")
    args = parser.parse_args()
    try:
        settings = DetectorSettings(processing_width=args.resolution)
        with FrisbeeCamera(args.url, settings, mode=SNAPSHOT if args.snapshot else VIDEO_STREAM) as camera:
            center = camera.get_center(timeout=args.timeout)
        print(json.dumps({"center": center}))
    except (ValueError, TimeoutError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
