"""Pick-and-throw instructions must always reach the human confirmation queue as two steps."""

from __future__ import annotations

import pytest

from app.robotics.action_schema import AllowedActionType, RobotActionProposal
from app.robotics.er2_client import GeminiRoboticsER2Client

PICK_AND_THROW = [
    "pick up the frisbee and throw it",
    "Pick up the frisbee and throw it correctly",
    "heb das Frisbee auf und wirf es",
    "nimm s frisbee und rüehr s",
]


def _pick() -> RobotActionProposal:
    return RobotActionProposal(
        action=AllowedActionType.PICK_OBJECT,
        object_label="frisbee",
        speed=0.2,
        requires_confirmation=True,
        rationale="Grasp the frisbee on the floor",
    )


def _throw() -> RobotActionProposal:
    return RobotActionProposal(
        action=AllowedActionType.THROW_OBJECT,
        object_label="frisbee",
        speed=0.25,
        requires_confirmation=True,
        rationale="Throw the frisbee",
    )


@pytest.mark.parametrize("instruction", PICK_AND_THROW)
@pytest.mark.parametrize("has_image", [True, False])
def test_dropped_throw_is_restored_after_pick(instruction: str, has_image: bool) -> None:
    specs = GeminiRoboticsER2Client._build_clause_specs(instruction)
    out = GeminiRoboticsER2Client._enforce_exact_clause_metrics([_pick()], specs, has_image=has_image)
    assert [p.action for p in out] == [AllowedActionType.PICK_OBJECT, AllowedActionType.THROW_OBJECT]
    throw = out[1]
    assert throw.requires_confirmation is True
    assert throw.object_label == "frisbee"


@pytest.mark.parametrize("instruction", PICK_AND_THROW)
def test_throw_from_er2_is_kept_once(instruction: str) -> None:
    specs = GeminiRoboticsER2Client._build_clause_specs(instruction)
    out = GeminiRoboticsER2Client._enforce_exact_clause_metrics([_pick(), _throw()], specs, has_image=True)
    assert [p.action for p in out] == [AllowedActionType.PICK_OBJECT, AllowedActionType.THROW_OBJECT]


def test_throw_without_pick_is_appended() -> None:
    specs = GeminiRoboticsER2Client._build_clause_specs("look at the frisbee and throw it")
    look = RobotActionProposal(
        action=AllowedActionType.LOOK_AT, object_label="frisbee", speed=0.2, requires_confirmation=False
    )
    out = GeminiRoboticsER2Client._enforce_exact_clause_metrics([look], specs, has_image=True)
    assert [p.action for p in out] == [AllowedActionType.LOOK_AT, AllowedActionType.THROW_OBJECT]


def test_throw_is_named_in_the_er2_constraints_prompt() -> None:
    specs = GeminiRoboticsER2Client._build_clause_specs("pick up the frisbee and throw it")
    prompt = GeminiRoboticsER2Client._format_metric_constraints_prompt(specs)
    assert "action=throw_object" in prompt


def test_throw_steps_go_to_the_confirmation_queue() -> None:
    """End to end through the safety gateway: both steps wait for a human, nothing is dispatched."""
    import asyncio

    from app.core.config import Settings
    from app.infrastructure.pubsub import TelemetryPubSubPublisher
    from app.robotics.action_validator import ActionSafetyValidator
    from app.robotics.robot_gateway import RobotSafetyGateway

    settings = Settings(app_env="test", robot_require_confirmation=True, pubsub_enabled=False)
    gateway = RobotSafetyGateway(settings, ActionSafetyValidator(settings), TelemetryPubSubPublisher(settings))
    specs = GeminiRoboticsER2Client._build_clause_specs("pick up the frisbee and throw it")
    plan = GeminiRoboticsER2Client._enforce_exact_clause_metrics([_pick()], specs, has_image=True)

    async def submit_all() -> list[bool]:
        return [(await gateway.submit_proposal(p))[1] for p in plan]

    dispatched = asyncio.run(submit_all())
    assert dispatched == [False, False]
    pending = [c.action for c in gateway.list_pending_confirmations()]
    assert pending == [AllowedActionType.PICK_OBJECT, AllowedActionType.THROW_OBJECT]


@pytest.mark.parametrize(
    "text, variant",
    [
        ("throw the frisbee", None),
        ("throw the frisbee early", "early"),
        ("wirf d'Frisbee früe", "early"),
        ("lance le frisbee plus tôt", "early"),
        ("早点扔飞盘", "early"),
    ],
)
def test_the_throw_version_is_taken_from_the_words(text: str, variant: str | None) -> None:
    from app.robotics.action_schema import RobotActionProposal
    from app.robotics.action_validator import ActionSafetyValidator
    from app.robotics.intent_parser import parse_explicit_clause_spec, parse_multilingual_robot_intent

    tool, args = parse_multilingual_robot_intent(text)
    assert (tool, args["action"], args["throw_variant"]) == ("propose_robot_action", "throw_object", variant)
    assert (variant or "standard") in args["rationale"]
    spec = parse_explicit_clause_spec(text)
    assert spec is not None and spec.throw_variant == variant
    from app.core.config import Settings
    from app.robotics.action_schema import RobotStatus

    settings = Settings(app_env="test", robot_require_confirmation=True, pubsub_enabled=False)
    cmd = ActionSafetyValidator(settings).parse_and_validate_proposal(RobotActionProposal(**args), RobotStatus())
    assert cmd.throw_variant == variant


@pytest.mark.parametrize("has_image", [True, False])
def test_er2_plans_keep_the_throw_version(has_image: bool) -> None:
    specs = GeminiRoboticsER2Client._build_clause_specs("pick up the frisbee and throw it early")
    # ER 2 kept the throw, or dropped it (restored from the clause): either way the version survives
    for er2 in ([_pick(), _throw()], [_pick()]):
        out = GeminiRoboticsER2Client._enforce_exact_clause_metrics(er2, specs, has_image=has_image)
        throws = [p for p in out if p.action == AllowedActionType.THROW_OBJECT]
        assert [t.throw_variant for t in throws] == ["early"]
