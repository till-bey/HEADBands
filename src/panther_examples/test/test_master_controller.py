import pytest
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from panther_examples.master_controller import split
from panther_examples.robot import LEFT_ARM, LEFT_TRAJ, RIGHT_ARM

CURRENT = {**{n: 0.1 for n in LEFT_ARM + RIGHT_ARM}, "joint_head0": 0.0, "joint_head1": 0.0,
           "body_joint": 0.4, "joint_lf": 0.0, "joint_rf": 0.0}


def traj(names, *points):
    return JointTrajectory(joint_names=names, points=[
        JointTrajectoryPoint(positions=list(q), time_from_start=Duration(sec=t)) for t, q in points])


def test_partial_arm_filled_and_others_stepped():
    arms, steps = split(traj(["joint_la1", "body_joint", "joint_head0"], (2, [0.5, 0.6, 0.3]), (4, [0.7, 0.6, 0.0])),
                        CURRENT)
    assert list(arms) == [LEFT_TRAJ]  # right arm untouched
    assert list(arms[LEFT_TRAJ].points[0].positions) == [0.1, 0.5] + [0.1] * 6
    assert [s[:3] for s in steps] == [(2.0, "joint_head0", 0.3), (2.0, "body_joint", 0.6),
                                      (4.0, "joint_head0", 0.0), (4.0, "body_joint", 0.6)]
    assert steps[1][3] == pytest.approx(0.1)  # 0.2 m in 2 s


def test_unknown_joint_refused():
    with pytest.raises(ValueError):
        split(traj(["joint_xx"], (1, [0.0])), CURRENT)
