"""Frisbee detection in one image, plain OpenCV (no ROS, no GPU, no neural network).

Color is not assumed. Candidate masks come from one of:
  - "motion":  absolute difference to the previous frame (the frisbee moves, the camera mostly does not);
  - "hsv":     an HSV range (optional, e.g. a white or yellow disc);
  - "motion_and_hsv": both.
Each candidate blob is fitted with an ellipse, because a flying disc seen at an angle is elliptical, and kept only
if its size, aspect ratio, solidity and ellipse fill look like a disc. Confidence is a 0..1 shape score; it is a
heuristic, not a calibrated probability. Temporal consistency is the tracker's job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    method: str = "background"  # background | motion | hsv | background_and_hsv | motion_and_hsv
    processing_width: int = 360  # images are resized to this width before detection (CPU budget)
    bg_history: int = 90  # MOG2 frames of memory
    bg_var_threshold: float = 32.0
    bg_warmup_frames: int = 5
    blur_kernel: int = 5
    motion_threshold: int = 25  # gray-level difference counted as motion
    stabilize: bool = True  # align previous frames by global translation (handheld / vibrating camera)
    hsv_lower: tuple[int, int, int] = (0, 0, 170)  # default: bright, unsaturated (white-ish); configurable
    hsv_upper: tuple[int, int, int] = (180, 70, 255)
    open_kernel: int = 3
    close_kernel: int = 7
    min_area_px: float = 30.0  # in processing pixels
    max_area_frac: float = 0.25  # of the processed image
    min_axis_px: float = 6.0  # major axis
    min_aspect: float = 0.15  # minor/major; an edge-on disc is thin
    min_solidity: float = 0.80
    min_ellipse_fill: float = 0.55  # contour area / fitted ellipse area
    min_confidence: float = 0.35
    max_detections: int = 5
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # x0, y0, x1, y1 as image fractions

    @classmethod
    def from_dict(cls, d: dict) -> DetectorConfig:
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                cur = getattr(cfg, k)
                setattr(cfg, k, tuple(v) if isinstance(cur, tuple) else type(cur)(v))
        return cfg


@dataclass
class Detection:
    u: float  # center, full-resolution pixels
    v: float
    bbox: tuple[float, float, float, float]  # x, y, w, h (full resolution)
    major_px: float  # ellipse axes (full resolution); major ~ true diameter, minor shrinks with tilt
    minor_px: float
    angle_deg: float
    confidence: float
    area_px: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def diameter_px(self) -> float:
        """Apparent diameter: the major axis, which tilt does not shorten (unlike the minor axis)."""
        return self.major_px


class FrisbeeDetector:
    """Stateful only for the motion method (keeps the previous processed frame)."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.cfg = config or DetectorConfig()
        self._gray_hist: list[np.ndarray] = []
        self._bg = None
        self._frames = 0
        self.last_mask: np.ndarray | None = None
        self.last_scale: float = 1.0

    def reset(self) -> None:
        self._gray_hist = []
        self._bg = None
        self._frames = 0

    @staticmethod
    def _align(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        """Shift `prev` onto `cur` by the global translation found with phase correlation."""
        (dx, dy), response = cv2.phaseCorrelate(np.float32(prev), np.float32(cur))
        if response < 0.05 or abs(dx) > prev.shape[1] * 0.2 or abs(dy) > prev.shape[0] * 0.2:
            return prev
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(prev, m, (prev.shape[1], prev.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    def _masks(self, small: np.ndarray) -> np.ndarray | None:
        cfg = self.cfg
        k = max(1, cfg.blur_kernel | 1)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (k, k), 0)
        motion = None
        if cfg.method in ("background", "background_and_hsv"):
            # MOG2 background model: robust to the moving thrower's ghosting; needs a mostly static camera.
            if self._bg is None:
                self._bg = cv2.createBackgroundSubtractorMOG2(
                    history=cfg.bg_history, varThreshold=cfg.bg_var_threshold, detectShadows=False
                )
            motion = self._bg.apply(small, learningRate=-1)
            self._frames += 1
            if self._frames < cfg.bg_warmup_frames:
                return None
            if cfg.method == "background":
                mask = motion
            else:
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                hsv_mask = cv2.inRange(hsv, np.array(cfg.hsv_lower, np.uint8), np.array(cfg.hsv_upper, np.uint8))
                mask = cv2.bitwise_and(motion, hsv_mask)
            if cfg.open_kernel > 1:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((cfg.open_kernel,) * 2, np.uint8))
            if cfg.close_kernel > 1:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((cfg.close_kernel,) * 2, np.uint8))
            return mask
        if cfg.method in ("motion", "motion_and_hsv"):
            # Three-frame differencing: a pixel is moving if it differs from BOTH of the last two frames.
            # This keeps the object's current position and drops its ghost at the previous position.
            # Previous frames are first aligned by a global translation (phase correlation) so small
            # camera shake does not light up the whole image.
            hist = self._gray_hist
            if len(hist) == 2 and all(h.shape == gray.shape for h in hist):
                diffs = []
                for prev in hist:
                    aligned = self._align(prev, gray) if cfg.stabilize else prev
                    d = cv2.absdiff(gray, aligned)
                    _, m = cv2.threshold(d, cfg.motion_threshold, 255, cv2.THRESH_BINARY)
                    diffs.append(m)
                motion = cv2.bitwise_and(diffs[0], diffs[1])
            self._gray_hist = (hist + [gray])[-2:]
            if motion is None:
                return None
        hsv_mask = None
        if cfg.method in ("hsv", "motion_and_hsv"):
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hsv_mask = cv2.inRange(hsv, np.array(cfg.hsv_lower, np.uint8), np.array(cfg.hsv_upper, np.uint8))
        if cfg.method == "motion":
            mask = motion
        elif cfg.method == "hsv":
            mask = hsv_mask
        elif cfg.method == "motion_and_hsv":
            # Motion locates the moving object, color confirms it; dilate motion so a disc's interior counts.
            mask = cv2.bitwise_and(cv2.dilate(motion, np.ones((9, 9), np.uint8)), hsv_mask)
        else:
            raise ValueError(f"unknown detection method '{cfg.method}'")
        if cfg.open_kernel > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((cfg.open_kernel,) * 2, np.uint8))
        if cfg.close_kernel > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((cfg.close_kernel,) * 2, np.uint8))
        return mask

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        cfg = self.cfg
        h0, w0 = bgr.shape[:2]
        scale = cfg.processing_width / float(w0) if cfg.processing_width and w0 > cfg.processing_width else 1.0
        small = cv2.resize(bgr, (round(w0 * scale), round(h0 * scale)), interpolation=cv2.INTER_AREA) if scale != 1.0 else bgr
        self.last_scale = scale
        mask = self._masks(small)
        self.last_mask = mask
        if mask is None:
            return []
        hs, ws = mask.shape[:2]
        x0, y0, x1, y1 = cfg.roi
        if (x0, y0, x1, y1) != (0.0, 0.0, 1.0, 1.0):
            roi = np.zeros_like(mask)
            roi[int(y0 * hs):int(y1 * hs), int(x0 * ws):int(x1 * ws)] = 255
            mask = cv2.bitwise_and(mask, roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = cfg.max_area_frac * hs * ws
        out: list[Detection] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < cfg.min_area_px or area > max_area or len(c) < 5:
                continue
            (cx, cy), (a1, a2), ang = cv2.fitEllipse(c)
            major, minor = max(a1, a2), min(a1, a2)
            if major < cfg.min_axis_px or major <= 0:
                continue
            aspect = minor / major
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity = area / hull_area if hull_area > 0 else 0.0
            ellipse_area = math.pi * major * minor / 4.0
            fill = area / ellipse_area if ellipse_area > 0 else 0.0
            if aspect < cfg.min_aspect or solidity < cfg.min_solidity or fill < cfg.min_ellipse_fill or fill > 1.35:
                continue
            # Shape score: solid, well-filled ellipses score high; very thin ones a little lower.
            fill_score = max(0.0, 1.0 - abs(1.0 - fill) * 1.5)
            conf = 0.45 * solidity + 0.40 * fill_score + 0.15 * min(1.0, aspect / 0.5)
            if conf < cfg.min_confidence:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            inv = 1.0 / scale
            out.append(Detection(
                u=cx * inv, v=cy * inv, bbox=(x * inv, y * inv, bw * inv, bh * inv),
                major_px=major * inv, minor_px=minor * inv, angle_deg=float(ang),
                confidence=float(round(conf, 4)), area_px=area * inv * inv,
                extra={"solidity": solidity, "fill": fill, "aspect": aspect},
            ))
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out[: cfg.max_detections]


def draw_detections(bgr: np.ndarray, detections: list[Detection], best: int | None = 0) -> np.ndarray:
    img = bgr.copy()
    for i, d in enumerate(detections):
        color = (0, 255, 0) if i == best else (0, 200, 255)
        x, y, w, h = (int(round(t)) for t in d.bbox)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.ellipse(img, ((d.u, d.v), (d.major_px, d.minor_px), d.angle_deg), color, 2)
        cv2.drawMarker(img, (int(d.u), int(d.v)), color, cv2.MARKER_CROSS, 14, 2)
        cv2.putText(img, f"{d.confidence:.2f}", (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img
