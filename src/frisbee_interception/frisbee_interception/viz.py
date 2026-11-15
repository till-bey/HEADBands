"""Debug drawing shared by the nodes and the offline runner (OpenCV only)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from frisbee_interception.core.detection import draw_detections

GREEN, YELLOW, RED, CYAN, MAGENTA, WHITE = (0, 220, 0), (0, 220, 255), (0, 0, 255), (255, 220, 0), (255, 0, 255), (255, 255, 255)


def _text(img, s, org, color=WHITE, scale=0.8):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_overlay(frame, detections, tracker, prediction) -> np.ndarray:
    """Detections (green best / yellow others), tracked history (cyan), predicted catch pixel (magenta),
    and a status block: track state, time to interception, confidence, validity and reason."""
    img = draw_detections(frame, detections) if detections else frame.copy()
    hist = list(tracker.history) if tracker is not None else []
    for a, b in zip(hist, hist[1:]):
        cv2.line(img, (int(a.u), int(a.v)), (int(b.u), int(b.v)), CYAN, 3, cv2.LINE_AA)
    est = tracker.estimate() if tracker is not None else None
    if est is not None:
        c = (int(est["u"]), int(est["v"]))
        cv2.circle(img, c, max(4, int(est["d"] / 2)), CYAN, 2, cv2.LINE_AA)
        tip = (int(est["u"] + est["du"] * 0.2), int(est["v"] + est["dv"] * 0.2))
        cv2.arrowedLine(img, c, tip, CYAN, 3, cv2.LINE_AA, tipLength=0.25)
    y = 40
    state = tracker.state if tracker is not None else "lost"
    _text(img, f"track: {state}", (15, y), CYAN if state in ("confirmed", "coasting") else WHITE)
    if prediction is not None:
        y += 34
        color = GREEN if prediction.valid else RED
        tti = prediction.time_to_intercept
        _text(img, f"{'VALID' if prediction.valid else 'invalid'}  conf {prediction.confidence:.2f}", (15, y), color)
        if not math.isnan(tti):
            y += 34
            _text(img, f"time to intercept {tti:+.2f} s", (15, y), color)
        if not prediction.valid and prediction.reason:
            y += 30
            _text(img, prediction.reason[:48], (15, y), RED, 0.6)
        if prediction.pixel is not None and prediction.valid:
            px = (int(prediction.pixel[0]), int(prediction.pixel[1]))
            cv2.drawMarker(img, px, MAGENTA, cv2.MARKER_TILTED_CROSS, 40, 4)
            cv2.circle(img, px, 26, MAGENTA, 3, cv2.LINE_AA)
            if est is not None:
                cv2.line(img, (int(est["u"]), int(est["v"])), px, MAGENTA, 2, cv2.LINE_AA)
    return img
