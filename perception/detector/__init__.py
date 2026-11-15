"""White ground-disc candidates using RGB thresholding and fitted ellipses.

This is a classical computer-vision heuristic, not a trained object classifier.
All coordinates refer to the input BGR image.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectorSettings:
    ground_start_pct: int = 25  # ignore the top of the image
    rgb_sum_min: int = 435  # R + G + B, range 0-765
    channel_difference_max: int = 30  # max(R,G,B) - min(R,G,B), range 0-255
    min_radius: int = 12
    max_radius: int = 120
    min_ellipse_ratio: float = 0.28
    min_area: int = 150
    min_white_fraction: float = 0.48


@dataclass(frozen=True)
class Candidate:
    center: tuple[int, int]
    axes: tuple[int, int]    # semiaxes, pixels (NOT diameters)
    angle: float           # angle in degrees
    method: str
    score: float           # heuristic 0-1 shape/white match, NOT probability

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        cx, cy = self.center
        ax, ay = self.axes
        rad = max(ax, ay)
        return (cx - rad, cy - rad, cx + rad, cy + rad)


def prepare_frame(frame: np.ndarray) -> np.ndarray:
    """Fit into 640x480 (landscape) or 480x640 (portrait), without distortion.

    Black letterboxing preserves the aspect ratio for sources that are not 4:3.
    Candidate coordinates and pixel-based settings refer to this prepared frame.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError("Expected a non-empty OpenCV BGR image with three channels")
    height, width = frame.shape[:2]
    target_width, target_height = (640, 480) if width >= height else (480, 640)
    if (width, height) == (target_width, target_height):
        return frame
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, min(target_width, round(width * scale)))
    resized_height = max(1, min(target_height, round(height * scale)))
    resized = cv2.resize(
        frame, (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    return cv2.copyMakeBorder(
        resized, top, target_height - resized_height - top,
        left, target_width - resized_width - left, cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )


def _inside_roi(y: int, shape: tuple[int, ...], cfg: DetectorSettings) -> bool:
    return y >= int(shape[0] * cfg.ground_start_pct / 100)


def _disk_mask(shape: tuple[int, ...], center: tuple[int, int], axes: tuple[int, int], angle: float) -> np.ndarray:
    out = np.zeros(shape[:2], dtype=np.uint8)
    cv2.ellipse(out, center, axes, angle, 0, 360, 255, -1)
    return out


def _fraction_white(mask: np.ndarray, ellipse: np.ndarray) -> float:
    denom = int(cv2.countNonZero(ellipse))
    if denom == 0:
        return 0.0
    return float(cv2.countNonZero(cv2.bitwise_and(mask, ellipse))) / denom


def preprocess(frame: np.ndarray, cfg: DetectorSettings) -> tuple[np.ndarray, np.ndarray]:
    """Return raw RGB threshold mask and cleaned detection mask, both uint8."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected an OpenCV BGR image with three channels")

    # Sum/spread are invariant to BGR channel order. Promote before summing
    # so bright pixels (up to 765) cannot overflow an 8-bit value.
    channels = frame.astype(np.uint16)
    brightness = channels.sum(axis=2)
    difference = channels.max(axis=2) - channels.min(axis=2)
    threshold_mask = (
        (brightness >= cfg.rgb_sum_min) & (difference <= cfg.channel_difference_max)
    ).astype(np.uint8) * 255
    cutoff = int(frame.shape[0] * cfg.ground_start_pct / 100)
    threshold_mask[:cutoff] = 0
    mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask[:cutoff] = 0
    return threshold_mask, mask


def find_ellipses(frame: np.ndarray, white_mask: np.ndarray, cfg: DetectorSettings) -> list[Candidate]:
    """Fit contours of thresholded white regions, then validate ellipse geometry."""
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out: list[Candidate] = []
    image_area = frame.shape[0] * frame.shape[1]

    for contour in contours:
        if len(contour) < 5:
            continue
        area = float(cv2.contourArea(contour))
        if area < cfg.min_area or area > image_area * 0.25:
            continue

        (cx, cy), (width, height), angle = cv2.fitEllipse(contour)
        semi_major = max(width, height) / 2
        semi_minor = min(width, height) / 2
        ratio = semi_minor / max(1.0, semi_major)
        equivalent_radius = np.sqrt(semi_major * semi_minor)
        if not cfg.min_radius <= equivalent_radius <= cfg.max_radius:
            continue
        if ratio < cfg.min_ellipse_ratio:
            continue

        # Reject irregular contours whose points do not follow the fitted ellipse.
        points = contour[:, 0, :].astype(np.float32) - (cx, cy)
        theta = np.deg2rad(angle)
        local_x = points[:, 0] * np.cos(theta) + points[:, 1] * np.sin(theta)
        local_y = -points[:, 0] * np.sin(theta) + points[:, 1] * np.cos(theta)
        radii = np.hypot(local_x / max(width / 2, 1e-6), local_y / max(height / 2, 1e-6))
        if float(np.mean(np.abs(radii - 1))) > 0.12:
            continue
        center = (int(round(cx)), int(round(cy)))
        axes = (max(1, int(round(width / 2))), max(1, int(round(height / 2))))
        if not _inside_roi(center[1], frame.shape, cfg):
            continue

        filled_ellipse = _disk_mask(frame.shape, center, axes, angle)
        whiteness = _fraction_white(white_mask, filled_ellipse)
        if whiteness < cfg.min_white_fraction:
            continue

        ellipse_area = max(1.0, np.pi * axes[0] * axes[1])
        shape_fill = min(1.0, area / ellipse_area)
        if shape_fill < 0.58:
            continue
        outer = _disk_mask(frame.shape, center, (int(axes[0] * 1.25), int(axes[1] * 1.25)), angle)
        ring = cv2.subtract(outer, filled_ellipse)
        outside_white = _fraction_white(white_mask, ring)
        if outside_white > 0.65:
            continue

        score = float(np.clip(0.5 * whiteness + 0.3 * shape_fill + 0.2 * (1 - outside_white), 0, 1))
        out.append(Candidate(center, axes, float(angle), "Ellipse", score))

    return _suppress_overlapping_ellipses(out)


def _suppress_overlapping_ellipses(candidates: list[Candidate]) -> list[Candidate]:
    """Keep the strongest detection for overlapping fitted ellipses."""
    chosen: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda d: d.score, reverse=True):
        rect = (candidate.center, (candidate.axes[0] * 2, candidate.axes[1] * 2), candidate.angle)
        duplicate = False
        for other in chosen:
            other_rect = (other.center, (other.axes[0] * 2, other.axes[1] * 2), other.angle)
            _, intersection = cv2.rotatedRectangleIntersection(rect, other_rect)
            if intersection is None:
                continue
            overlap = cv2.contourArea(intersection)
            union = 4 * (candidate.axes[0] * candidate.axes[1] + other.axes[0] * other.axes[1]) - overlap
            if overlap / max(union, 1.0) > 0.5:
                duplicate = True
                break
        if not duplicate:
            chosen.append(candidate)
    return chosen


def detect(frame: np.ndarray, cfg: DetectorSettings) -> tuple[list[Candidate], dict[str, np.ndarray]]:
    """Threshold RGB, clean the mask, fit its contours, and return debug masks."""
    threshold_mask, mask = preprocess(frame, cfg)
    candidates = find_ellipses(frame, mask, cfg)
    return candidates[:20], {"threshold_mask": threshold_mask, "white_mask": mask}


def draw_detections(frame: np.ndarray, candidates: list[Candidate], cfg: DetectorSettings) -> np.ndarray:
    """Draw ROI and detections on a copy of the BGR input image."""
    result = frame.copy()
    cutoff = int(frame.shape[0] * cfg.ground_start_pct / 100)
    if cutoff > 0:
        cv2.line(result, (0, cutoff), (frame.shape[1] - 1, cutoff), (255, 190, 40), 2)
        cv2.putText(result, "Ground ROI below", (12, max(18, cutoff - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 190, 40), 2, cv2.LINE_AA)

    for index, candidate in enumerate(candidates, start=1):
        color = (75, 225, 80)
        cv2.ellipse(result, candidate.center, candidate.axes, candidate.angle, 0, 360, color, 3)
        cv2.circle(result, candidate.center, 3, (255, 80, 180), -1)
        text = f"#{index} {candidate.method}: {candidate.score:.2f}"
        cv2.putText(result, text, (candidate.center[0] - candidate.axes[0],
                                  max(20, candidate.center[1] - max(candidate.axes) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return result
