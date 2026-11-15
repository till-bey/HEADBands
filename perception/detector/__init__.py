"""Frisbee rims and white discs detected across the image with contour ellipses.

This is a classical computer-vision heuristic, not a trained object classifier.
All coordinates refer to the input BGR image.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import dist as math_distance

import cv2
import numpy as np

RGB_MEAN_RADIUS_SCALE = 0.75


@dataclass(frozen=True)
class DetectorSettings:
    canny_low: int = 40
    canny_high: int = 120
    rgb_sum_min: int = 160  # R + G + B, range 0-765
    channel_difference_max: int = 55  # max(R,G,B) - min(R,G,B), range 0-255
    min_radius: int = 10
    max_radius: int = 200
    min_ellipse_ratio: float = 0.65
    min_area: int = 150
    min_white_fraction: float = 0.48
    processing_width: int = 640  # landscape width; portrait swaps width/height


@dataclass(frozen=True)
class Candidate:
    center: tuple[int, int]
    axes: tuple[int, int]    # semiaxes, pixels (NOT diameters)
    angle: float           # angle in degrees
    method: str
    score: float           # heuristic 0-1 shape/white match, NOT probability
    mean_rgb_sum: float = 0.0  # Mean R+G+B inside the inner 75% of both ellipse radii.

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        cx, cy = self.center
        ax, ay = self.axes
        rad = max(ax, ay)
        return (cx - rad, cy - rad, cx + rad, cy + rad)


def prepare_frame(frame: np.ndarray, processing_width: int = 640) -> np.ndarray:
    """Fit into processing_width × 3/4 width, swapping axes for portrait.

    Black letterboxing preserves the aspect ratio for sources that are not 4:3.
    Candidate coordinates and pixel-based settings refer to this prepared frame.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError("Expected a non-empty OpenCV BGR image with three channels")
    height, width = frame.shape[:2]
    if not isinstance(processing_width, int) or processing_width < 4 or processing_width % 4:
        raise ValueError("processing_width must be a positive multiple of 4")
    short_side = processing_width * 3 // 4
    target_width, target_height = (processing_width, short_side) if width >= height else (short_side, processing_width)
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
    b, g, r = cv2.split(frame)
    brightness = cv2.add(cv2.add(b, g, dtype=cv2.CV_16U), r, dtype=cv2.CV_16U)
    difference = cv2.subtract(cv2.max(cv2.max(b, g), r), cv2.min(cv2.min(b, g), r))
    threshold_mask = cv2.bitwise_and(
        cv2.inRange(brightness, cfg.rgb_sum_min, 765),
        cv2.inRange(difference, 0, cfg.channel_difference_max),
    )
    mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return threshold_mask, mask


def rim_edges(frame: np.ndarray, cfg: DetectorSettings) -> np.ndarray:
    """Find outer/rim boundaries independently of the disc's interior color."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), cfg.canny_low, cfg.canny_high)


def find_ellipses(frame: np.ndarray, white_mask: np.ndarray, cfg: DetectorSettings,
                  *, edge_mask: np.ndarray | None = None) -> list[Candidate]:
    """Fit rim edges and white regions; report unthresholded interior brightness."""
    if edge_mask is None:
        edge_mask = rim_edges(frame, cfg)
    rim_contours, _ = cv2.findContours(edge_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sources = [(contour, 'Rim ellipse') for contour in rim_contours]
    sources.extend((contour, 'White ellipse') for contour in white_contours)
    out: list[Candidate] = []
    image_area = frame.shape[0] * frame.shape[1]

    for contour, method in sources:
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
        fit_error = float(np.mean(np.abs(radii - 1)))
        fit_limit = 0.07
        if fit_error > fit_limit:
            continue
        # A short arc can fit an ellipse very well. Require support around most
        # of its perimeter so a single straight/curved edge is not a disc.
        phases = np.arctan2(local_y / max(height / 2, 1e-6), local_x / max(width / 2, 1e-6))
        bins = np.floor((phases + np.pi) * (16 / (2 * np.pi))).astype(int) % 16
        if np.unique(bins).size < 12:
            continue
        center = (int(round(cx)), int(round(cy)))
        axes = (max(1, int(round(width / 2))), max(1, int(round(height / 2))))
        # Validate only the candidate's bounding patch, rather than allocating
        # and scanning several full-frame masks per contour.
        reach = int(np.ceil(max(axes) * 1.25)) + 2
        x0, y0 = max(0, center[0] - reach), max(0, center[1] - reach)
        x1, y1 = min(frame.shape[1], center[0] + reach + 1), min(frame.shape[0], center[1] + reach + 1)
        patch = white_mask[y0:y1, x0:x1]
        local_center = (center[0] - x0, center[1] - y0)
        filled_ellipse = _disk_mask(patch.shape, local_center, axes, angle)
        whiteness = _fraction_white(patch, filled_ellipse)
        if method == 'White ellipse' and whiteness < cfg.min_white_fraction:
            continue

        ellipse_area = max(1.0, np.pi * axes[0] * axes[1])
        shape_fill = min(1.0, area / ellipse_area)
        if shape_fill < 0.58:
            continue
        outer = _disk_mask(patch.shape, local_center, (int(axes[0] * 1.25), int(axes[1] * 1.25)), angle)
        ring = cv2.subtract(outer, filled_ellipse)
        outside_white = _fraction_white(patch, ring)
        if method == 'White ellipse' and outside_white > 0.65:
            continue

        if method == 'Rim ellipse':
            score = float(np.clip(0.7 * (1 - fit_error / fit_limit) + 0.3 * shape_fill, 0, 1))
        else:
            score = float(np.clip(0.5 * whiteness + 0.3 * shape_fill + 0.2 * (1 - outside_white), 0, 1))
        # Compute on the original, unannotated BGR patch. Summing channel means
        # is the mean of per-pixel R+G+B, with no uint8 overflow or white-only bias.
        # Fixed-point coordinates preserve the exact 0.75 radius scale before
        # rasterization, including small or thin ellipses.
        inner_ellipse = np.zeros(patch.shape[:2], dtype=np.uint8)
        cv2.ellipse(inner_ellipse, tuple(v * 4 for v in local_center),
                    tuple(round(v * RGB_MEAN_RADIUS_SCALE * 4) for v in axes),
                    angle, 0, 360, 255, -1, shift=2)
        mean_rgb_sum = float(sum(cv2.mean(frame[y0:y1, x0:x1], mask=inner_ellipse)[:3]))
        out.append(Candidate(center, axes, float(angle), method, score, mean_rgb_sum))

    return _suppress_overlapping_ellipses(out)


def _suppress_overlapping_ellipses(candidates: list[Candidate]) -> list[Candidate]:
    """Keep the strongest detection for overlapping fitted ellipses."""
    chosen: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda d: d.score, reverse=True):
        rect = (candidate.center, (candidate.axes[0] * 2, candidate.axes[1] * 2), candidate.angle)
        duplicate = False
        for index, other in enumerate(chosen):
            other_rect = (other.center, (other.axes[0] * 2, other.axes[1] * 2), other.angle)
            _, intersection = cv2.rotatedRectangleIntersection(rect, other_rect)
            if intersection is None:
                continue
            overlap = cv2.contourArea(intersection)
            candidate_area = np.pi * candidate.axes[0] * candidate.axes[1]
            other_area = np.pi * other.axes[0] * other.axes[1]
            union = candidate_area + other_area - overlap
            nested_rim = (math_distance(candidate.center, other.center) <= max(3, 0.2 * min(max(candidate.axes), max(other.axes)))
                          and overlap / max(1, min(candidate_area, other_area)) > 0.85
                          and min(candidate_area, other_area) / max(candidate_area, other_area) > 0.45)
            if overlap / max(union, 1.0) > 0.5 or nested_rim:
                # Keep the outer rim rather than a second ellipse on its inside.
                if candidate_area > other_area:
                    chosen[index] = candidate
                duplicate = True
                break
        if not duplicate:
            chosen.append(candidate)
    return sorted(chosen, key=lambda candidate: candidate.score, reverse=True)


def detect(frame: np.ndarray, cfg: DetectorSettings) -> tuple[list[Candidate], dict[str, np.ndarray]]:
    """Find disc boundaries and measure RGB within the inner 75% of each radius."""
    threshold_mask, mask = preprocess(frame, cfg)
    edges = rim_edges(frame, cfg)
    candidates = find_ellipses(frame, mask, cfg, edge_mask=edges)
    return candidates[:20], {"threshold_mask": threshold_mask, "white_mask": mask, "edge_mask": edges}


def draw_detections(frame: np.ndarray, candidates: list[Candidate], cfg: DetectorSettings) -> np.ndarray:
    """Draw detections on a copy of the BGR input image."""
    result = frame.copy()

    for index, candidate in enumerate(candidates, start=1):
        color = (75, 225, 80)
        cv2.ellipse(result, candidate.center, candidate.axes, candidate.angle, 0, 360, color, 3)
        cv2.circle(result, candidate.center, 3, (255, 80, 180), -1)
        text = f"#{index} {candidate.method}: {candidate.score:.2f} | RGB75% {candidate.mean_rgb_sum:.1f}"
        cv2.putText(result, text, (candidate.center[0] - candidate.axes[0],
                                  max(20, candidate.center[1] - max(candidate.axes) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return result
