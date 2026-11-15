"""Head control: direct head commands, wider look range, and head keyframes in generated gestures."""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.robotics.action_schema import AllowedActionType, RobotActionProposal, RobotStatus
from app.robotics.action_validator import ActionSafetyValidator, SafetyValidationError
from app.robotics.er2_client import GeminiRoboticsER2Client
from app.robotics.gesture_planner import _mock_keyframes, plan_from_keyframes
from app.robotics.head_limits import HEAD_LIMITS, HEAD_MAX_VEL_RPS
from app.robotics.intent_parser import parse_explicit_clause_spec, parse_multilingual_robot_intent

BRIDGE_SRC = Path(__file__).resolve().parents[3] / "robot_bridge" / "ros2_ws" / "src" / "swiss_robot_bridge"


def _settings() -> Settings:
    return Settings(
        app_env="test", stt_default_provider="mock", stt_swiss_german_provider="mock",
        tts_default_provider="mock", tts_swiss_german_provider="mock", llm_provider="mock",
        robotics_enabled=True, robotics_provider="mock", robot_bridge_shared_token="t",
        serve_frontend=False, pubsub_enabled=False,
    )


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize(
    ("text", "pitch_deg", "yaw_deg"),
    [
        ("look left", None, 45), ("look to the right 30 degrees", None, -30), ("look up", -15, None),
        ("look down 60°", 60, None), ("turn your head left", None, 45), ("look straight ahead", 0, 0),
        ("schau nach links", None, 45), ("dreh den Kopf nach rechts", None, -45), ("lueg nach obe", -15, None),
        ("regarde en bas", 40, None), ("mira a la derecha", None, -45), ("podívej se doleva", None, 45),
        ("向左看", None, 45), ("低头", 40, None),
    ],
)
def test_head_phrases_become_move_head(text: str, pitch_deg: float | None, yaw_deg: float | None) -> None:
    tool, args = parse_multilingual_robot_intent(text)
    assert args["action"] == "move_head"
    deg = lambda v: None if v is None else round(math.degrees(v))  # noqa: E731
    assert (deg(args["head_pitch_rad"]), deg(args["head_yaw_rad"])) == (pitch_deg, yaw_deg)
    assert parse_explicit_clause_spec(text).action == "move_head"


def test_turning_the_head_never_rotates_the_base_but_turn_left_still_does() -> None:
    assert parse_multilingual_robot_intent("turn your head left")[1]["action"] == "move_head"
    assert parse_multilingual_robot_intent("turn left")[1]["action"] == "rotate"
    assert parse_multilingual_robot_intent("look around") is None  # stays a camera task for ER 2


def test_head_angle_is_clamped_to_the_limits() -> None:
    _, args = parse_multilingual_robot_intent("look left 120 degrees")
    assert args["head_yaw_rad"] == pytest.approx(HEAD_LIMITS["joint_head1"][1], abs=1e-3)


# ------------------------------------------------------------------ validation
def test_validator_rules_for_head_angles() -> None:
    v = ActionSafetyValidator(_settings())
    ok = v.parse_and_validate_proposal(
        RobotActionProposal(action=AllowedActionType.MOVE_HEAD, head_yaw_rad=0.5, speed=0.2, requires_confirmation=False),
        RobotStatus(),
    )
    assert ok.head_yaw_rad == 0.5 and not ok.requires_confirmation
    with pytest.raises(SafetyValidationError, match="needs head"):
        v.parse_and_validate_proposal(RobotActionProposal(action=AllowedActionType.MOVE_HEAD, speed=0.2), RobotStatus())
    with pytest.raises(SafetyValidationError, match="outside"):
        v.parse_and_validate_proposal(
            RobotActionProposal(action=AllowedActionType.MOVE_HEAD, head_pitch_rad=1.6, speed=0.2), RobotStatus()
        )
    with pytest.raises(SafetyValidationError, match="not allowed on 'move_to'"):
        v.parse_and_validate_proposal(
            RobotActionProposal(action=AllowedActionType.MOVE_TO, target={"x": 1, "y": 0}, head_yaw_rad=0.3, speed=0.2),
            RobotStatus(),
        )


# ------------------------------------------------------------------ ER 2 enforcement
def test_er2_look_at_for_look_left_becomes_exact_move_head() -> None:
    specs = GeminiRoboticsER2Client._build_clause_specs("look left")
    look = RobotActionProposal(action=AllowedActionType.LOOK_AT, object_label="left side", speed=0.2)
    out = GeminiRoboticsER2Client._enforce_exact_clause_metrics([look], specs, has_image=True)
    assert [p.action for p in out] == [AllowedActionType.MOVE_HEAD]
    assert out[0].head_yaw_rad == pytest.approx(math.radians(45), abs=1e-3) and out[0].object_label is None


def test_er2_dropped_head_move_is_restored_in_order() -> None:
    specs = GeminiRoboticsER2Client._build_clause_specs("look down at the floor and then pick up the frisbee")
    pick = RobotActionProposal(action=AllowedActionType.PICK_OBJECT, object_label="frisbee", speed=0.2)
    out = GeminiRoboticsER2Client._enforce_exact_clause_metrics([pick], specs, has_image=True)
    assert [p.action for p in out] == [AllowedActionType.MOVE_HEAD, AllowedActionType.PICK_OBJECT]


def test_api_head_command_is_dispatched_without_confirmation() -> None:
    with TestClient(create_app(settings_override=_settings())) as client:
        r = client.post("/api/v1/robotics/analyze", data={"instruction": "look to the right 30 degrees"})
        steps = r.json()["validated_commands"]
        assert [s["command"]["action"] for s in steps] == ["move_head"]
        assert steps[0]["awaiting_confirmation"] is False
        assert round(math.degrees(steps[0]["command"]["head_yaw_rad"])) == -30


# ------------------------------------------------------------------ gestures with the head
@pytest.mark.parametrize("description", ["nod", "shake your head", "look around"])
def test_head_gestures_plan_and_validate(description: str) -> None:
    result = plan_from_keyframes(_mock_keyframes(description), description)
    assert result.ok, result.reasons
    traj = result.trajectory
    assert traj.joint_names == ["joint_head0", "joint_head1"]
    ActionSafetyValidator(_settings()).assert_gesture_trajectory_safe(traj)
    prev_t, prev = 0.0, traj.start_positions
    for p in traj.points:
        assert max(abs(a - b) for a, b in zip(p.positions, prev, strict=True)) / (p.t - prev_t) <= HEAD_MAX_VEL_RPS + 1e-6
        prev_t, prev = p.t, p.positions


# ------------------------------------------------------------------ bridge
@pytest.fixture()
def bridge():
    sys.path.insert(0, str(BRIDGE_SRC))
    try:
        from swiss_robot_bridge import local_safety

        yield local_safety
    finally:
        sys.path.remove(str(BRIDGE_SRC))


def _cmd(action: str, **extra) -> dict:
    return {
        "command_id": "h1", "action": action, "speed": 0.2, "requires_confirmation": False,
        "confirmed_by_human": True, "rationale": "test",
        "expires_at": (datetime.now(tz=UTC) + timedelta(seconds=30)).isoformat(), **extra,
    }


def test_bridge_move_head_respects_the_active_range(bridge) -> None:
    conservative = bridge.LocalSafetySupervisor()
    plan = conservative.validate_and_compute_panther_plan(_cmd("move_head", head_pitch_rad=1.2))
    assert plan.head_targets == {"joint_head0": pytest.approx(0.34)}  # yaw untouched, pitch clamped
    full = bridge.LocalSafetySupervisor(head_full_range=True)
    plan = full.validate_and_compute_panther_plan(_cmd("move_head", head_pitch_rad=1.2))
    assert plan.head_targets == {"joint_head0": pytest.approx(1.2)}


def test_bridge_look_at_floor_is_no_longer_damped(bridge) -> None:
    full = bridge.LocalSafetySupervisor(head_full_range=True)
    plan = full.validate_and_compute_panther_plan(
        _cmd("look_at", object_label="frisbee", target={"x": 0.5, "y": 0.0, "z": 0.0})
    )
    assert plan.head_targets["joint_head0"] == pytest.approx(math.atan2(1.05, 0.5), abs=1e-3)  # ~64 deg down


def test_bridge_refuses_head_gesture_beyond_conservative_range(bridge) -> None:
    from app.robotics.gesture_planner import GestureKeyframePlan, HeadKeyframe

    wide = GestureKeyframePlan(name="look far left", head_keyframes=[HeadKeyframe(t=2.0, yaw_deg=80.0)])
    traj = plan_from_keyframes(wide, "look far left").trajectory.model_dump(mode="json")  # 80 deg > 69 deg
    cmd = _cmd("generated_gesture", object_label="look far left", requires_confirmation=True, gesture=traj)
    with pytest.raises(bridge.LocalSafetyRejectedError, match="PANTHER_HEAD_FULL_RANGE"):
        bridge.LocalSafetySupervisor().validate_and_compute_panther_plan(cmd)
    plan = bridge.LocalSafetySupervisor(head_full_range=True).validate_and_compute_panther_plan(cmd)
    assert plan.master_trajectory_joints == ("joint_head0", "joint_head1")
