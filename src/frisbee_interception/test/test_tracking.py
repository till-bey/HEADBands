"""Tracking: velocity and size-rate estimates, false-detection rejection, detection loss, real timestamps."""

import math
import random

import pytest

from frisbee_interception.core.tracking import (
    CONFIRMED,
    COASTING,
    LOST,
    TENTATIVE,
    FrisbeeTracker,
    Measurement,
    TrackerConfig,
)


def _throw(n=20, seed=1, jitter=True):
    """Disc moving at (120, -60) px/s with diameter growing 30 -> ~90 px, irregular timestamps."""
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += (1 / 30) * (rng.uniform(0.7, 1.3) if jitter else 1.0)
        out.append(Measurement(t, 300 + 120 * t + rng.gauss(0, 1.5), 400 - 60 * t + rng.gauss(0, 1.5),
                               30 + 100 * t + rng.gauss(0, 1.0), 0.9))
    return out


def test_confirms_and_estimates_velocity_and_size_rate():
    tr = FrisbeeTracker(TrackerConfig())
    states = [tr.step(m.t, [m]) for m in _throw()]
    assert states[0] == TENTATIVE and states[2] == CONFIRMED
    assert len(tr.history) == 20
    # Instantaneous velocity is noisy with 1.5 px measurement noise; averaged over seeds it is unbiased.
    du, dv, dd = [], [], []
    for seed in range(1, 21):
        tr = FrisbeeTracker(TrackerConfig())
        for m in _throw(seed=seed):
            tr.step(m.t, [m])
        e = tr.estimate()
        du.append(e["du"])
        dv.append(e["dv"])
        dd.append(e["dd"])
    assert sum(du) / 20 == pytest.approx(120, abs=10)
    assert sum(dv) / 20 == pytest.approx(-60, abs=10)
    assert sum(dd) / 20 == pytest.approx(100, abs=10)
    assert all(abs(x - 120) < 50 for x in du)  # and never wildly off


def test_isolated_false_detection_is_never_confirmed():
    tr = FrisbeeTracker(TrackerConfig())
    assert tr.step(0.00, [Measurement(0.00, 100, 100, 40, 0.9)]) == TENTATIVE
    # Next frames: only unrelated blobs scattered around the image, never twice in a consistent place.
    for t, (u, v) in zip((0.033, 0.066, 0.1, 0.133), ((600, 900), (120, 1200), (690, 210), (300, 650))):
        tr.step(t, [Measurement(t, u, v, 40, 0.9)])
    assert tr.state != CONFIRMED and tr.estimate() is None


def test_clutter_does_not_hijack_a_confirmed_track():
    tr = FrisbeeTracker(TrackerConfig())
    for m in _throw(10):
        tr.step(m.t, [m, Measurement(m.t, 50, 1200, 35, 0.95)])  # a confident static distractor
    est = tr.estimate()
    assert tr.state == CONFIRMED and est["u"] > 300 and est["v"] < 400


def test_fast_disc_takes_primary_from_a_slow_walker():
    """A slowly moving person/head is confirmed first; a frisbee appearing later must become primary."""
    tr = FrisbeeTracker(TrackerConfig())
    t = 0.0
    for _ in range(30):  # walker: 40 px/s, confirmed long before the throw
        t += 1 / 30
        tr.step(t, [Measurement(t, 100 + 40 * t, 900, 60, 0.9)])
    assert tr.state == CONFIRMED and tr.estimate()["u"] < 200
    t0 = t
    for _ in range(6):  # disc: 600 px/s and growing, while the walker keeps being detected
        t += 1 / 30
        walker = Measurement(t, 100 + 40 * t, 900, 60, 0.9)
        disc = Measurement(t, 600 - 600 * (t - t0), 300, 30 + 200 * (t - t0), 0.8)
        tr.step(t, [walker, disc])
    est = tr.estimate()
    assert est["v"] < 500 and est["du"] < -300  # the primary is the disc


def test_coasts_through_a_short_gap_and_drops_after_timeout():
    cfg = TrackerConfig(max_coast_s=0.2)
    tr = FrisbeeTracker(cfg)
    ms = _throw(8, jitter=False)
    for m in ms:
        tr.step(m.t, [m])
    t = ms[-1].t
    assert tr.step(t + 0.05, []) == COASTING
    resumed = Measurement(t + 0.1, 300 + 120 * (t + 0.1), 400 - 60 * (t + 0.1), 30 + 100 * (t + 0.1), 0.9)
    assert tr.step(resumed.t, [resumed]) == CONFIRMED  # re-acquired inside the gate
    assert tr.step(resumed.t + 0.15, []) == COASTING
    assert tr.step(resumed.t + 0.3, []) == LOST


def test_uses_timestamps_not_frame_count():
    """Same motion sampled at 15 fps and 60 fps gives the same velocity."""
    out = []
    for fps in (15, 60):
        tr = FrisbeeTracker(TrackerConfig())
        for i in range(1, 16):
            t = i / fps
            tr.step(t, [Measurement(t, 200 + 150 * t, 300, 40 + 50 * t, 0.9)])
        out.append(tr.estimate()["du"])
    assert out[0] == pytest.approx(150, abs=10) and out[1] == pytest.approx(150, abs=10)


def test_big_size_jump_is_rejected():
    tr = FrisbeeTracker(TrackerConfig())
    for m in _throw(6):
        tr.step(m.t, [m])
    before = tr.hits
    t = tr.last_meas_t + 0.033
    est = tr.estimate(t)
    tr.step(t, [Measurement(t, est["u"], est["v"], est["d"] * 3, 0.9)])
    assert tr.hits == before and tr.rejected >= 1
    assert math.isfinite(tr.estimate()["u"])
