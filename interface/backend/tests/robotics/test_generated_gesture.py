"""Language-generated arm gestures: planner, validator, gateway, API, ER 2 path and robot bridge."""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.infrastructure.pubsub import TelemetryPubSubPublisher
from app.main import create_app
from app.robotics.action_schema import AllowedActionType, GestureTrajectory, RobotActionProposal
from app.robotics.action_validator import ActionSafetyValidator, SafetyValidationError
from app.robotics.gesture_planner import (
    MAX_JOINT_VEL,
    GestureKeyframePlan,
    HandKeyframe,
    _mock_keyframes,
    load_arm_chains,
    plan_from_keyframes,
)
from app.robotics.robot_gateway import RobotSafetyGateway

BRIDGE_SRC = Path(__file__).resolve().parents[3] / "robot_bridge" / "ros2_ws" / "src" / "swiss_robot_bridge"


def _settings() -> Settings:
    return Settings(
        app_env="test",
        stt_default_provider="mock",
        stt_swiss_german_provider="mock",
        tts_default_provider="mock",
        tts_swiss_german_provider="mock",
        llm_provider="mock",
        robotics_enabled=True,
        robotics_provider="mock",
        robot_bridge_shared_token="t",
        serve_frontend=False,
        pubsub_enabled=False,
    )


@pytest.mark.parametrize("description", ["salute", "wave hello", "raise both arms", "point forward"])
def test_mock_gestures_plan_within_limits(description: str) -> None:
    result = plan_from_keyframes(_mock_keyframes(description), description)
    assert result.ok, result.reasons
    traj = result.trajectory
    assert traj.peak_joint_vel_rps <= MAX_JOINT_VEL + 1e-6
    assert traj.min_clearance_m >= 0.03
    assert 3.0 < traj.duration_s < 30.0
    ActionSafetyValidator(_settings()).assert_gesture_trajectory_safe(traj)


def test_salute_puts_the_right_hand_at_the_forehead() -> None:
    traj = plan_from_keyframes(_mock_keyframes("salute"), "salute").trajectory
    chain = load_arm_chains()["right"]
    hand = [chain.fk(__import__("numpy").array(p.positions))[0] for p in traj.points]
    highest = max(hand, key=lambda p: p[2])
    assert highest[2] > 0.22 and highest[0] > 0.15  # up at forehead height, in front of the face
    assert abs(traj.points[-1].positions[1]) < 0.05  # back near the hanging pose at the end


def test_hand_inside_the_head_is_refused() -> None:
    plan = GestureKeyframePlan(name="bad", keyframes=[HandKeyframe(t=2, arm="right", x=0.05, y=0.0, z=0.25)])
    result = plan_from_keyframes(plan, "bad")
    assert not result.ok and "head" in result.reasons[0]


def test_unreachable_keyframe_is_refused() -> None:
    plan = GestureKeyframePlan(name="far", keyframes=[HandKeyframe(t=2, arm="right", x=1.2, y=-0.3, z=0.0)])
    result = plan_from_keyframes(plan, "far")
    assert not result.ok and "cannot reach" in result.reasons[0]


def _planned() -> GestureTrajectory:
    return plan_from_keyframes(_mock_keyframes("salute"), "salute").trajectory


def test_validator_refuses_tampered_trajectories() -> None:
    v = ActionSafetyValidator(_settings())
    fast = _planned().model_copy(deep=True)
    for p in fast.points:
        p.t = p.t / 4  # four times faster than planned
    with pytest.raises(SafetyValidationError, match="speed"):
        v.assert_gesture_trajectory_safe(fast)
    wrong = _planned().model_copy(deep=True)
    wrong.joint_names[0] = "joint_head0"
    with pytest.raises(SafetyValidationError, match="outside the arms"):
        v.assert_gesture_trajectory_safe(wrong)
    out = _planned().model_copy(deep=True)
    out.points[3].positions[2] = 2.5  # joint_ra2 limit is [-1.57, 0]
    with pytest.raises(SafetyValidationError, match="outside"):
        v.assert_gesture_trajectory_safe(out)


def test_gesture_action_needs_a_planned_trajectory() -> None:
    v = ActionSafetyValidator(_settings())
    from app.robotics.action_schema import RobotStatus

    proposal = RobotActionProposal(action=AllowedActionType.GENERATED_GESTURE, object_label="salute", speed=0.2)
    with pytest.raises(SafetyValidationError, match="no planned trajectory"):
        v.parse_and_validate_proposal(proposal, RobotStatus())
    with pytest.raises(SafetyValidationError, match="only allowed"):
        v.parse_and_validate_proposal(
            RobotActionProposal(action=AllowedActionType.LOOK_AT, object_label="x", speed=0.2),
            RobotStatus(),
            gesture=_planned(),
        )


def test_gateway_plans_and_queues_for_confirmation() -> None:
    s = _settings()
    gw = RobotSafetyGateway(s, ActionSafetyValidator(s), TelemetryPubSubPublisher(s))

    async def go():
        return await gw.submit_proposal(
            {"action": "generated_gesture", "object_label": "salute with the right hand", "speed": 0.2}
        )

    cmd, dispatched = asyncio.run(go())
    assert dispatched is False
    assert cmd.requires_confirmation and not cmd.confirmed_by_human
    assert cmd.gesture is not None and cmd.gesture.name == "salute"
    assert [c.command_id for c in gw.list_pending_confirmations()] == [cmd.command_id]


def test_caller_supplied_trajectory_is_never_used() -> None:
    s = _settings()
    gw = RobotSafetyGateway(s, ActionSafetyValidator(s), TelemetryPubSubPublisher(s))
    evil = _planned().model_dump()
    evil["name"] = "injected"
    for p in evil["points"]:
        p["t"] /= 10

    async def go():
        return await gw.submit_proposal(
            {"action": "generated_gesture", "object_label": "salute", "speed": 0.2, "gesture": evil}
        )

    cmd, _ = asyncio.run(go())
    assert cmd.gesture.name == "salute"  # re-planned, the injected one was dropped


def test_gesture_endpoint_and_er2_path() -> None:
    with TestClient(create_app(settings_override=_settings())) as client:
        r = client.post("/api/v1/robotics/gesture", json={"description": "wave hello"})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["awaiting_confirmation"] is True and d["dispatched"] is False
        assert d["command"]["gesture"]["name"] == "wave"

        r = client.post("/api/v1/robotics/analyze", data={"instruction": "salute"})
        assert r.status_code == 200, r.text
        steps = r.json()["validated_commands"]
        assert [s["command"]["action"] for s in steps] == ["generated_gesture"]
        assert steps[0]["awaiting_confirmation"] is True
        assert steps[0]["command"]["gesture"]["joint_names"][0] == "joint_ra0"


# ---------------------------------------------------------------- robot bridge (no ROS needed)
@pytest.fixture()
def bridge():
    sys.path.insert(0, str(BRIDGE_SRC))
    try:
        from swiss_robot_bridge import local_safety

        yield local_safety
    finally:
        sys.path.remove(str(BRIDGE_SRC))


def _command(gesture: dict, confirmed: bool = True) -> dict:
    return {
        "command_id": "g1",
        "action": "generated_gesture",
        "object_label": "salute",
        "speed": 0.2,
        "requires_confirmation": True,
        "confirmed_by_human": confirmed,
        "expires_at": (datetime.now(tz=UTC) + timedelta(seconds=30)).isoformat(),
        "rationale": "Generated arm gesture: salute",
        "gesture": gesture,
    }


def test_bridge_accepts_a_planned_gesture_and_waits_for_it(bridge) -> None:
    sup = bridge.LocalSafetySupervisor()
    sup.record_cloud_heartbeat(100.0)
    g = _planned().model_dump(mode="json")
    plan, now = sup.submit_cloud_command(_command(g), monotonic_now=100.0)
    assert now is True
    assert plan.master_trajectory_joints[0] == "joint_ra0"
    assert plan.trajectory_duration_seconds == pytest.approx(g["points"][-1]["t"])
    assert sup.motion_deadline_monotonic == pytest.approx(100.0 + plan.trajectory_duration_seconds + 0.5)


def test_bridge_refuses_unconfirmed_or_tampered_gestures(bridge) -> None:
    sup = bridge.LocalSafetySupervisor()
    g = _planned().model_dump(mode="json")
    with pytest.raises(bridge.LocalSafetyRejectedError, match="confirmed_by_human"):
        sup.validate_and_compute_panther_plan(_command(g, confirmed=False))
    fast = _planned().model_dump(mode="json")
    for p in fast["points"]:
        p["t"] /= 4
    with pytest.raises(bridge.LocalSafetyRejectedError, match="speed"):
        sup.validate_and_compute_panther_plan(_command(fast))
    head = _planned().model_dump(mode="json")
    head["joint_names"][0] = "joint_head0"
    with pytest.raises(bridge.LocalSafetyRejectedError, match="not allowed"):
        sup.validate_and_compute_panther_plan(_command(head))


def test_bridge_delays_start_when_the_arm_is_elsewhere(bridge) -> None:
    names = ("joint_ra1",)
    pts = [(0.5, (-1.0,), None), (1.0, (-1.2,), None)]
    shifted = bridge.retime_from_current_pose(names, pts, {"joint_ra1": 0.0}, max_vel=0.45)
    assert shifted[0][0] == pytest.approx(1.0 / 0.45, abs=1e-3)  # 1 rad at 0.45 rad/s
    assert bridge.retime_from_current_pose(names, pts, {"joint_ra1": -0.95}, max_vel=0.45) == pts
