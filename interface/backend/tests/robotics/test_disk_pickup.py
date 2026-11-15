"""'pick up the disk' reaches the bridge as pick_object/disk, and the bridge plays the key-pose pickup."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.robotics.action_schema import AllowedActionType, RobotActionProposal
from app.robotics.er2_client import GeminiRoboticsER2Client
from app.robotics.intent_parser import parse_multilingual_robot_intent

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "robot_bridge" / "ros2_ws" / "src" / "swiss_robot_bridge"))
from swiss_robot_bridge.local_safety import PICKUP_JOINTS, LocalSafetySupervisor  # noqa: E402
from swiss_robot_bridge.panther_topics import ARM_JOINT_LIMITS, MAX_ARM_JOINT_VEL_RPS  # noqa: E402


@pytest.mark.parametrize("text", ["pick up the disk", "Please pick up the disc"])
def test_trigger_phrase_gives_pick_disk(text: str) -> None:
    tool, args = parse_multilingual_robot_intent(text)
    assert (tool, args["action"], args["object_label"]) == ("propose_robot_action", "pick_object", "disk")


@pytest.mark.parametrize("has_image", [True, False])
def test_er2_keeps_or_restores_the_disk_pickup(has_image: bool) -> None:
    specs = GeminiRoboticsER2Client._build_clause_specs("pick up the disk")
    er2_pick = RobotActionProposal(action=AllowedActionType.PICK_OBJECT, object_label="frisbee", speed=0.2)
    for er2 in ([er2_pick], []):
        out = GeminiRoboticsER2Client._enforce_exact_clause_metrics(er2, specs, has_image=has_image)
        assert [(p.action, p.object_label) for p in out] == [(AllowedActionType.PICK_OBJECT, "disk")]


def test_bridge_plays_the_key_poses_safely() -> None:
    now = datetime.now(tz=UTC)
    plan = LocalSafetySupervisor().validate_and_compute_panther_plan(
        {
            "command_id": "disk",
            "action": "pick_object",
            "object_label": "disk",
            "confirmed_by_human": True,
            "expires_at": (now + timedelta(seconds=30)).isoformat(),
        },
        now_utc=now,
    )
    assert plan.master_trajectory_joints == PICKUP_JOINTS and plan.lift_position_m == pytest.approx(0.19)
    pts = plan.master_trajectory_points
    assert len(pts) == 11  # 4 poses + 3 holds + 4 steps of the gripper close ramp
    arms = [n for n in PICKUP_JOINTS if n in ARM_JOINT_LIMITS]
    for (t0, q0, _), (t1, q1, _) in zip(pts, pts[1:]):
        assert t1 > t0
        assert max(abs(a - b) for a, b in zip(q0[: len(arms)], q1[: len(arms)])) / (t1 - t0) <= MAX_ARM_JOINT_VEL_RPS
    for _, q, _ in pts:
        assert all(ARM_JOINT_LIMITS[n][0] <= v <= ARM_JOINT_LIMITS[n][1] for n, v in zip(arms, q))
    rf = [dict(zip(PICKUP_JOINTS, q))["joint_rf"] for _, q, _ in pts]
    assert rf == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0])  # ramps closed before the move away
    assert pts[-2][0] - pts[5][0] == pytest.approx(0.4)  # 0.6 at -0.4 s ... 1.0 when the move away starts
    assert plan.trajectory_duration_seconds == pts[-1][0]


def test_throw_waits_for_the_lift_with_the_arms_held() -> None:
    from swiss_robot_bridge.bridge_node import ROS2HardwareAdapter
    from swiss_robot_bridge.camera_manager import PantherCameraManager

    now = datetime.now(tz=UTC)
    sup = LocalSafetySupervisor()
    plan = sup.validate_and_compute_panther_plan(
        {"command_id": "t", "action": "throw_object", "object_label": "frisbee", "confirmed_by_human": True,
         "expires_at": (now + timedelta(seconds=30)).isoformat()},
        now_utc=now,
    )
    hw = ROS2HardwareAdapter(supervisor=sup, camera_manager=PantherCameraManager(allow_sim_fallback=True))
    hw.lift_height_m = 0.19
    current = {n: 0.0 for n in plan.master_trajectory_joints}
    current.update({f"joint_la{i}": q for i, q in enumerate(plan.master_trajectory_points[0][1][:8])})
    current.update({f"joint_ra{i}": q for i, q in enumerate(plan.master_trajectory_points[0][1][8:16])})
    hw.joint_positions.update(current)  # arms already at the throw's start: only the lift wait remains
    hw.apply_panther_plan(plan)
    pts = hw.last_master_trajectory
    assert pts[0][0] == pytest.approx((0.55 - 0.19) / 0.05)  # held where they are until the lift is at 0.55
    assert pts[0][1] == tuple(current[n] for n in plan.master_trajectory_joints[:16])
    assert all(b[0] > a[0] for a, b in zip(pts, pts[1:]))
