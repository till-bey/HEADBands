"""Straight-line interception prediction from a short history of (t, u, v, d) measurements.

Two modes, chosen by what calibration exists. Nothing here invents calibration values.

1. Metric (camera intrinsics fx, fy, cx, cy and the physical frisbee diameter D are known):
   for a disc of diameter D, Z = fx * D / d, X = (u - cx) * D / d, Y = (v - cy) * D / d * fx / fy
   (camera optical frame: x right, y down, z forward). For straight, constant-velocity flight all three are
   linear in time, so a weighted least-squares line through them is the 3D trajectory. It is transformed
   into the target frame (rotation R, translation p of the camera in that frame) and intersected with the
   catching plane (point + normal in the target frame) -> catch point, time to interception, uncertainty.

2. Image-space fallback (no metric calibration):
   u(t) and v(t) are fitted with lines; 1/d(t) is linear in time on a straight approach, so the time when
   the disc reaches the calibrated apparent diameter at the catching plane, d_catch, is where the 1/d line
   reaches 1/d_catch. The predicted pixel at that time can be mapped to a catch position with a
   pixel -> catch-plane homography measured by the catching-plane calibration procedure.

Estimates are rejected (valid=False with a reason) instead of being reported when they are unreliable:
too few points, the disc not approaching, a poor size fit (tumbling or tilting disc), interception in the
past or too far in the future, or uncertainty above the configured limits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PredictorConfig:
    min_points: int = 4
    max_points: int = 12  # most recent measurements used
    min_span_s: float = 0.06  # time covered by the points
    frisbee_diameter_m: float = 0.27  # standard 175 g disc; set to your frisbee
    min_approach_speed_mps: float = 0.5  # metric mode: must approach the plane at least this fast
    min_size_growth: float = 0.3  # image mode: 1/d must shrink by this fraction over the window (conditioning)
    max_size_fit_residual: float = 0.12  # relative RMS residual of the 1/d (or Z) fit; tumbling discs fail
    max_time_to_intercept_s: float = 2.0
    min_time_to_intercept_s: float = 0.0
    max_position_sigma_m: float = 0.25
    max_pixel_sigma: float = 60.0
    max_time_sigma_s: float = 0.15
    max_measurement_age_s: float = 0.1  # predictions from older data (coasting track) are not valid
    catch_diameter_px: float = 0.0  # image mode: apparent diameter at the catching plane (calibrated); 0 = unset

    @classmethod
    def from_dict(cls, d: dict) -> PredictorConfig:
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, type(getattr(cfg, k))(v))
        return cfg


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Plane:
    """Catching plane in the target frame: a point on it and its normal (need not be unit)."""

    point: np.ndarray
    normal: np.ndarray


@dataclass
class Prediction:
    valid: bool
    reason: str = ""
    mode: str = ""
    t_ref: float = 0.0  # time of the newest measurement used
    time_to_intercept: float = float("nan")  # seconds after t_ref
    intercept_time: float = float("nan")  # absolute (same clock as the measurements)
    point: np.ndarray | None = None  # catch point in the target frame (metric) or catch plane (homography)
    point_sigma_m: float = float("nan")
    time_sigma_s: float = float("nan")
    pixel: tuple[float, float] | None = None  # predicted image position at interception
    pixel_sigma: float = float("nan")
    confidence: float = 0.0
    line_points: list[np.ndarray] = field(default_factory=list)  # trajectory samples for visualization
    diagnostics: dict = field(default_factory=dict)


def weighted_line_fit(t: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    """y = a + b * t. Returns (a, b, covariance 2x2 of (a, b), relative-free RMS residual)."""
    w = np.ones_like(t) if w is None else w
    A = np.stack([np.ones_like(t), t], axis=1)
    W = np.diag(w)
    AtWA = A.T @ W @ A
    cov_unscaled = np.linalg.inv(AtWA)
    coef = cov_unscaled @ A.T @ W @ y
    res = y - A @ coef
    dof = max(1, len(t) - 2)
    s2 = float(res @ W @ res) / dof
    return float(coef[0]), float(coef[1]), cov_unscaled * s2, math.sqrt(float(np.mean(res**2)))


def _line_var(cov: np.ndarray, t: float) -> float:
    return float(cov[0, 0] + 2 * t * cov[0, 1] + t * t * cov[1, 1])


class InterceptPredictor:
    def __init__(self, config: PredictorConfig | None = None) -> None:
        self.cfg = config or PredictorConfig()

    def _window(self, history) -> tuple[np.ndarray, ...] | str:
        pts = list(history)[-self.cfg.max_points:]
        if len(pts) < self.cfg.min_points:
            return f"only {len(pts)} points (need {self.cfg.min_points})"
        t = np.array([p.t for p in pts], float)
        if t[-1] - t[0] < self.cfg.min_span_s:
            return f"points span {t[-1] - t[0]:.3f} s (need {self.cfg.min_span_s:.3f} s)"
        u = np.array([p.u for p in pts], float)
        v = np.array([p.v for p in pts], float)
        d = np.array([p.d for p in pts], float)
        w = np.array([max(0.05, p.confidence) for p in pts], float)
        return t, u, v, d, w

    # ------------------------------------------------------------------ metric mode
    def predict_metric(
        self,
        history,
        K: Intrinsics,
        R_target_cam: np.ndarray,
        p_target_cam: np.ndarray,
        plane: Plane,
    ) -> Prediction:
        win = self._window(history)
        if isinstance(win, str):
            return Prediction(False, win, "metric")
        t, u, v, d, w = win
        t0 = t[-1]
        tt = t - t0  # fit relative to the newest sample for conditioning
        D = self.cfg.frisbee_diameter_m
        Z = K.fx * D / d
        X = (u - K.cx) * D / d
        Y = (v - K.cy) * D / d * (K.fx / K.fy)
        fits = [weighted_line_fit(tt, c, w) for c in (X, Y, Z)]
        z_rel_res = fits[2][3] / max(1e-6, float(np.mean(Z)))
        diag = {"z_rel_residual": z_rel_res, "z_now": fits[2][0], "vz_cam": fits[2][1]}
        if z_rel_res > self.cfg.max_size_fit_residual:
            return Prediction(False, f"size fit too noisy ({z_rel_res:.2f}): tumbling or tilting disc", "metric",
                              t0, diagnostics=diag)
        p0_cam = np.array([f[0] for f in fits])
        vel_cam = np.array([f[1] for f in fits])
        R = np.asarray(R_target_cam, float)
        p0 = R @ p0_cam + np.asarray(p_target_cam, float)
        vel = R @ vel_cam
        n = np.asarray(plane.normal, float)
        n = n / np.linalg.norm(n)
        closing = -float(n @ vel) if float(n @ (p0 - plane.point)) > 0 else float(n @ vel)
        diag["closing_speed_mps"] = closing
        if closing < self.cfg.min_approach_speed_mps:
            return Prediction(False, f"not approaching the catching plane ({closing:.2f} m/s)", "metric", t0,
                              diagnostics=diag)
        tc = float(n @ (plane.point - p0)) / float(n @ vel)
        if not (self.cfg.min_time_to_intercept_s <= tc <= self.cfg.max_time_to_intercept_s):
            return Prediction(False, f"interception in {tc:.2f} s is outside the allowed window", "metric", t0,
                              time_to_intercept=tc, diagnostics=diag)
        point = p0 + vel * tc
        # Uncertainty: line variance of each camera axis at tc, rotated into the target frame.
        cov_cam = np.diag([_line_var(f[2], tc) for f in fits])
        cov = R @ cov_cam @ R.T
        point_sigma = math.sqrt(max(0.0, float(np.trace(cov))))
        time_sigma = math.sqrt(max(0.0, float(n @ cov @ n))) / abs(float(n @ vel))
        pix = None
        if point is not None:
            pc = p0_cam + vel_cam * tc
            if pc[2] > 1e-3:
                pix = (K.cx + K.fx * pc[0] / pc[2], K.cy + K.fy * pc[1] / pc[2])
        pred = Prediction(
            True, "", "metric", t0, tc, t0 + tc, point, point_sigma, time_sigma, pix,
            line_points=[p0 + vel * s for s in np.linspace(tt[0], tc, 12)], diagnostics=diag,
        )
        return self._finalize(pred, len(t), w)

    # ------------------------------------------------------------------ image mode
    def predict_image(
        self,
        history,
        homography: np.ndarray | None = None,
        principal_point: tuple[float, float] | None = None,
    ) -> Prediction:
        """Image-space prediction. `principal_point` (cx, cy) defaults to the image center the caller passes;
        no focal length is needed: (u - cx) / d and (v - cy) / d are linear in time for a straight flight."""
        win = self._window(history)
        if isinstance(win, str):
            return Prediction(False, win, "image")
        if self.cfg.catch_diameter_px <= 0:
            return Prediction(False, "catch_diameter_px not calibrated (image mode needs it)", "image")
        t, u, v, d, w = win
        t0 = t[-1]
        tt = t - t0
        inv_d = 1.0 / d
        a, b, cov_i, rms = weighted_line_fit(tt, inv_d, w)
        rel_res = rms / max(1e-9, float(np.mean(inv_d)))
        growth = 1.0 - (a + b * tt[-1]) / (a + b * tt[0]) if (a + b * tt[0]) != 0 else 0.0
        diag = {"inv_d_rel_residual": rel_res, "size_growth": growth, "d_now": 1.0 / a if a > 0 else float("nan")}
        if rel_res > self.cfg.max_size_fit_residual:
            return Prediction(False, f"size fit too noisy ({rel_res:.2f}): tumbling or tilting disc", "image", t0,
                              diagnostics=diag)
        if b >= 0 or growth < self.cfg.min_size_growth:
            return Prediction(False, f"disc not approaching (size growth {growth:.2f})", "image", t0,
                              diagnostics=diag)
        target = 1.0 / self.cfg.catch_diameter_px
        tc = (target - a) / b
        if not (self.cfg.min_time_to_intercept_s <= tc <= self.cfg.max_time_to_intercept_s):
            return Prediction(False, f"interception in {tc:.2f} s is outside the allowed window", "image", t0,
                              time_to_intercept=tc, diagnostics=diag)
        # Time uncertainty from the 1/d line: d(tc)/d(a) = -1/b, d(tc)/d(b) = -tc/b.
        J = np.array([-1.0 / b, -tc / b])
        time_sigma = math.sqrt(max(0.0, float(J @ cov_i @ J)))
        # Perspective-correct image position: (u - cx) / d = X / D is linear in time (likewise for v), so
        # extrapolate it and multiply by the apparent diameter at the catching plane.
        cx, cy = principal_point if principal_point is not None else (float(np.mean(u)), float(np.mean(v)))
        dc = self.cfg.catch_diameter_px
        fu = weighted_line_fit(tt, (u - cx) / d, w)
        fv = weighted_line_fit(tt, (v - cy) / d, w)
        pix = (cx + dc * (fu[0] + fu[1] * tc), cy + dc * (fv[0] + fv[1] * tc))
        pix_sigma = dc * math.sqrt(max(0.0, _line_var(fu[2], tc) + _line_var(fv[2], tc)))
        diag["principal_point"] = (cx, cy)
        point = None
        if homography is not None:
            q = np.asarray(homography, float) @ np.array([pix[0], pix[1], 1.0])
            if abs(q[2]) > 1e-9:
                point = np.array([q[0] / q[2], q[1] / q[2], 0.0])  # catch-plane coordinates
        pred = Prediction(
            True, "", "image", t0, tc, t0 + tc, point, float("nan"), time_sigma, pix, pix_sigma,
            diagnostics=diag,
        )
        return self._finalize(pred, len(t), w)

    def at_time(self, pred: Prediction, t_now: float) -> Prediction:
        """Express a prediction relative to t_now (e.g. while the track coasts on old measurements).

        time_to_intercept becomes intercept_time - t_now; a crossing that is already (nearly) over, or a
        prediction built on measurements older than `max_measurement_age_s`, is no longer valid.
        """
        if not pred.valid or math.isnan(pred.intercept_time):
            return pred
        age = t_now - pred.t_ref
        pred.time_to_intercept = pred.intercept_time - t_now
        if age > self.cfg.max_measurement_age_s:
            pred.valid, pred.reason = False, f"newest measurement {age:.2f} s old (track coasting)"
        elif pred.time_to_intercept < self.cfg.min_time_to_intercept_s:
            pred.valid, pred.reason = False, f"interception in {pred.time_to_intercept:.2f} s is outside the allowed window"
        return pred

    # ------------------------------------------------------------------ shared checks
    def _finalize(self, pred: Prediction, n: int, w: np.ndarray) -> Prediction:
        cfg = self.cfg
        if pred.time_sigma_s > cfg.max_time_sigma_s:
            pred.valid, pred.reason = False, f"time uncertainty {pred.time_sigma_s:.2f} s too large"
        elif pred.mode == "metric" and pred.point_sigma_m > cfg.max_position_sigma_m:
            pred.valid, pred.reason = False, f"position uncertainty {pred.point_sigma_m:.2f} m too large"
        elif pred.mode == "image" and pred.pixel_sigma > cfg.max_pixel_sigma:
            pred.valid, pred.reason = False, f"pixel uncertainty {pred.pixel_sigma:.0f} px too large"
        # Confidence: more points, tighter fits and confident detections score higher (heuristic, 0..1).
        pts_score = min(1.0, n / max(cfg.max_points, 1))
        if pred.mode == "metric":
            unc = min(1.0, pred.point_sigma_m / cfg.max_position_sigma_m)
        else:
            unc = min(1.0, pred.pixel_sigma / cfg.max_pixel_sigma)
        t_unc = min(1.0, pred.time_sigma_s / cfg.max_time_sigma_s)
        pred.confidence = float(round(
            max(0.0, 0.3 * pts_score + 0.3 * (1 - unc) + 0.2 * (1 - t_unc) + 0.2 * float(np.mean(np.clip(w, 0, 1)))),
            4,
        ))
        return pred
