"""Catching pose from the URDF: reachable, pinch orientation, clear of joint limits, small corrections."""

import os
from pathlib import Path

import numpy as np
import pytest

from frisbee_interception.core.catch_pose import R_PINCH, CatchPoseConfig, CatchPosePlanner
from frisbee_interception.core.kinematics import ArmChain, rot_error

URDF = Path(os.environ.get("FRISBEE_TEST_URDF",
                           Path(__file__).resolve().parents[2] / "panther_sim" / "urdf" / "robot_description.urdf"))
pytestmark = pytest.mark.skipif(not URDF.is_file(), reason="panther_sim URDF not in this checkout")


@pytest.fixture(scope="module")
def planner():
    pl = CatchPosePlanner(URDF.read_text(), CatchPoseConfig(samples=6000))
    pl.solve_nominal(k=8)
    return pl


def test_zero_pose_hangs_down():
    ch = ArmChain(URDF.read_text(), "body_link", "link_lf")
    assert ch.n == 8 and ch.fk(np.zeros(8))[2, 3] == pytest.approx(-0.683, abs=0.01)


def test_ik_recovers_a_reachable_pose():
    ch = ArmChain(URDF.read_text(), "body_link", "link_rf")
    q_true = np.array([0.3, -0.8, -0.5, 0.4, -1.0, 0.2, 0.1, 0.3])
    T = ch.fk(q_true)
    q = ch.ik(T[:3, 3], T[:3, :3], q_true + 0.15)
    assert q is not None
    assert np.linalg.norm(ch.fk(q)[:3, 3] - T[:3, 3]) < 1e-3


def test_nominal_pose_puts_both_hands_in_front_of_the_chest(planner):
    for arm, q in planner.nominal.items():
        ch = planner.chains[arm]
        T = ch.fk(q)
        assert np.linalg.norm(T[:3, 3] - planner.hand_target(arm)) < 1e-3
        # fingers point at the thrower (+x) and close vertically, up to the gripper's 180 deg symmetry
        assert T[0, 2] == pytest.approx(-1.0, abs=1e-2)  # gripper z axis = -x (fingers along +x)
        assert abs(T[2, 1]) == pytest.approx(1.0, abs=1e-2)  # gripper y (closing axis) vertical
        assert ch.limit_margin(q) > 0.1  # room to correct


def test_corrections_reach_the_shifted_point_with_small_joint_motion(planner):
    for dy, dz in ((0.05, 0.0), (-0.05, 0.05), (0.08, -0.08)):
        targets, applied = planner.correction(dy, dz)
        assert applied == (dy, dz)
        for arm, q in targets.items():
            T = planner.chains[arm].fk(q)
            assert np.linalg.norm(T[:3, 3] - planner.hand_target(arm, dy, dz)) < 2e-3
            assert np.linalg.norm(rot_error(planner.rot[arm], T[:3, :3])) < 0.2  # still a pinch (~10 deg)
            assert np.abs(q - planner.nominal[arm]).max() < 0.35  # < 0.7 s at 0.5 rad/s


def test_corrections_are_clamped(planner):
    targets, applied = planner.correction(0.5, -0.5)
    assert applied == (planner.cfg.max_correction_m, -planner.cfg.max_correction_m)
    assert R_PINCH.shape == (3, 3)
