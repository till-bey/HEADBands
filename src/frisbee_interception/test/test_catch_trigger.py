"""Catch trigger: close the grippers at arrival minus latency, only for throws that cross the catch window."""

import pytest

from frisbee_interception.core.catch_trigger import CatchConfig, CatchScheduler


def _cfg(**kw):
    base = dict(gripper_latency_s=0.15, catch_window_px=[300.0, 500.0, 420.0, 700.0])
    base.update(kw)
    return CatchConfig(**base)


def test_unmeasured_latency_or_window_blocks_arming():
    assert set(CatchConfig().calibration_problems()) == {"gripper_latency_s not measured", "catch_window_px not set"}
    assert _cfg().calibration_problems() == []


def test_fires_at_arrival_minus_latency_and_updates_with_new_predictions():
    s = CatchScheduler(_cfg())
    s.update(1.00, True, 1.50, (360, 600), 0.8)
    s.update(1.10, True, 1.46, (362, 598), 0.85)  # a newer prediction refines the plan
    assert s.plan.close_at == pytest.approx(1.31)
    assert s.due(1.30) is None
    fired = s.due(1.31)
    assert fired is not None and fired.intercept_time == pytest.approx(1.46)
    assert s.due(1.40) is None  # only once


def test_throw_outside_the_window_or_unconfident_is_not_caught():
    s = CatchScheduler(_cfg())
    assert s.update(1.0, True, 1.5, (100, 600), 0.9) is None and "miss" in s.last_reason
    assert s.update(1.0, True, 1.5, (360, 600), 0.2) is None and "confidence" in s.last_reason
    s.update(1.0, True, 1.5, (360, 600), 0.9)
    assert s.plan is not None
    s.update(1.1, True, 1.5, (600, 600), 0.9)  # newest prediction leaves the window: cancel
    assert s.plan is None


def test_plan_survives_losing_the_disc_in_the_last_frames():
    s = CatchScheduler(_cfg())
    s.update(1.0, True, 1.3, (360, 600), 0.9)
    s.update(1.1, False, float("nan"), None, 0.0)  # disc too close / blurred: no new prediction
    assert s.due(1.16) is not None


def test_refractory_and_missed_plans():
    s = CatchScheduler(_cfg(refractory_s=1.0))
    s.update(1.0, True, 1.3, (360, 600), 0.9)
    assert s.due(1.2) is not None
    assert s.update(1.5, True, 1.9, (360, 600), 0.9) is None and "refractory" in s.last_reason
    s2 = CatchScheduler(_cfg())
    s2.update(1.0, True, 1.3, (360, 600), 0.9)
    assert s2.due(1.5) is None and "missed" in s2.last_reason  # never close long after the disc arrived
