"""Prediction: catch point and time from a straight approach, image-space fallback, and rejections."""

import math
import random

import numpy as np
import pytest

from frisbee_interception.core.prediction import (
    InterceptPredictor,
    Intrinsics,
    Plane,
    PredictorConfig,
    weighted_line_fit,
)
from frisbee_interception.core.tracking import Measurement

K = Intrinsics(fx=800.0, fy=800.0, cx=360.0, cy=640.0)
D = 0.27


def _project(p):
    x, y, z = p
    return K.cx + K.fx * x / z, K.cy + K.fy * y / z, K.fx * D / z


def _approach(n=10, noise=0.0, seed=3, start=(0.6, -0.3, 7.0), vel=(-0.4, 0.2, -8.0), dt=1 / 30, jitter=True):
    """A disc flying straight at the camera (optical frame), sampled with irregular timestamps."""
    rng = random.Random(seed)
    t, out, truth = 0.0, [], []
    for _ in range(n):
        t += dt * (rng.uniform(0.8, 1.2) if jitter else 1.0)
        p = np.array(start) + np.array(vel) * t
        u, v, d = _project(p)
        out.append(Measurement(t, u + rng.gauss(0, noise), v + rng.gauss(0, noise), d * (1 + rng.gauss(0, noise / 200)), 0.9))
        truth.append(p)
    return out, np.array(start), np.array(vel)


def test_line_fit_recovers_slope_and_intercept():
    t = np.linspace(0, 1, 20)
    a, b, cov, rms = weighted_line_fit(t, 2.0 + 3.0 * t)
    assert (a, b) == (pytest.approx(2.0), pytest.approx(3.0)) and rms < 1e-9


def test_metric_prediction_hits_the_plane_where_the_disc_crosses_it():
    hist, start, vel = _approach(noise=0.5)
    plane = Plane(point=np.array([0.0, 0.0, 1.0]), normal=np.array([0.0, 0.0, 1.0]))  # 1 m in front of the camera
    pred = InterceptPredictor(PredictorConfig()).predict_metric(hist, K, np.eye(3), np.zeros(3), plane)
    assert pred.valid, pred.reason
    t_cross = (1.0 - start[2]) / vel[2]
    expected = start + vel * t_cross
    assert pred.intercept_time == pytest.approx(t_cross, abs=0.03)
    assert pred.time_to_intercept == pytest.approx(t_cross - hist[-1].t, abs=0.03)
    assert np.linalg.norm(pred.point - expected) < 0.05
    assert pred.point_sigma_m < 0.1 and 0 < pred.confidence <= 1


def test_metric_prediction_in_a_robot_frame():
    """Camera 1.2 m up, looking along robot +x; plane x = 0.4 in the robot frame."""
    hist, start, vel = _approach(noise=0.3)
    # optical (x right, y down, z forward) -> robot (x forward, y left, z up)
    R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    p = np.array([0.0, 0.0, 1.2])
    plane = Plane(point=np.array([0.4, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0]))
    pred = InterceptPredictor(PredictorConfig()).predict_metric(hist, K, R, p, plane)
    assert pred.valid, pred.reason
    t_cross = (0.4 - start[2]) / vel[2]
    expected = R @ (start + vel * t_cross) + p
    assert np.linalg.norm(pred.point - expected) < 0.05
    assert pred.point[0] == pytest.approx(0.4, abs=1e-6)


def test_image_mode_time_and_pixel():
    hist, start, vel = _approach(noise=0.5)
    z_catch = 1.0
    cfg = PredictorConfig(catch_diameter_px=K.fx * D / z_catch)
    pred = InterceptPredictor(cfg).predict_image(hist, principal_point=(K.cx, K.cy))
    assert pred.valid, pred.reason
    t_cross = (z_catch - start[2]) / vel[2]
    assert pred.intercept_time == pytest.approx(t_cross, abs=0.03)
    u_exp, v_exp, _ = _project(start + vel * t_cross)
    assert math.hypot(pred.pixel[0] - u_exp, pred.pixel[1] - v_exp) < 15  # 0.5 px noise, extrapolated


def test_image_mode_maps_pixel_through_homography():
    hist, _, _ = _approach()
    H = np.diag([0.001, 0.001, 1.0])  # px -> m on the catch plane
    pred = InterceptPredictor(PredictorConfig(catch_diameter_px=216.0)).predict_image(hist, H, (K.cx, K.cy))
    assert pred.valid and pred.point is not None
    assert pred.point[0] == pytest.approx(pred.pixel[0] * 0.001, rel=1e-6)


def test_rejects_a_receding_disc():
    hist, _, _ = _approach(vel=(0.0, 0.0, 6.0), start=(0.0, 0.0, 2.0))
    plane = Plane(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]))
    pred = InterceptPredictor(PredictorConfig()).predict_metric(hist, K, np.eye(3), np.zeros(3), plane)
    assert not pred.valid and "approaching" in pred.reason
    pred_i = InterceptPredictor(PredictorConfig(catch_diameter_px=216)).predict_image(hist)
    assert not pred_i.valid


def test_rejects_a_tumbling_disc_with_erratic_size():
    hist, _, _ = _approach()
    for i, m in enumerate(hist):
        m.d *= 1.6 if i % 2 else 0.6  # apparent size jumping around (tilt / tumbling)
    plane = Plane(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]))
    pred = InterceptPredictor(PredictorConfig()).predict_metric(hist, K, np.eye(3), np.zeros(3), plane)
    assert not pred.valid and "noisy" in pred.reason


def test_rejects_too_few_points_and_far_future():
    hist, _, _ = _approach(n=3)
    plane = Plane(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]))
    assert "points" in InterceptPredictor().predict_metric(hist, K, np.eye(3), np.zeros(3), plane).reason
    slow, _, _ = _approach(vel=(0.0, 0.0, -1.0), start=(0.0, 0.0, 9.0), n=12)
    pred = InterceptPredictor(PredictorConfig(max_time_to_intercept_s=2.0, min_approach_speed_mps=0.5)).predict_metric(
        slow, K, np.eye(3), np.zeros(3), plane)
    assert not pred.valid and "window" in pred.reason


def test_image_mode_requires_catch_size_calibration():
    hist, _, _ = _approach()
    pred = InterceptPredictor(PredictorConfig(catch_diameter_px=0)).predict_image(hist)
    assert not pred.valid and "calibrated" in pred.reason


def test_prediction_from_a_coasting_track_expires():
    """The same (old) history evaluated later must not stay valid with a stale time to intercept."""
    hist, _, _ = _approach(noise=0.5)
    p = InterceptPredictor(PredictorConfig(catch_diameter_px=K.fx * D / 1.0))
    pred = p.at_time(p.predict_image(hist, None, (K.cx, K.cy)), hist[-1].t)
    assert pred.valid
    tti = pred.time_to_intercept
    later = p.at_time(p.predict_image(hist, None, (K.cx, K.cy)), hist[-1].t + 0.05)
    assert later.time_to_intercept == pytest.approx(tti - 0.05)
    stale = p.at_time(p.predict_image(hist, None, (K.cx, K.cy)), hist[-1].t + 0.2)
    assert not stale.valid and "old" in stale.reason
