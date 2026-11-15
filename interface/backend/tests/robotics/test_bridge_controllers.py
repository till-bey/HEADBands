"""Robot bridge without master_controller: arm trajectories straight to the trajectory controllers,
head and grippers streamed at their times (as panther_examples/master_gui.py does)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BRIDGE_SRC = Path(__file__).resolve().parents[3] / "robot_bridge" / "ros2_ws" / "src" / "swiss_robot_bridge"


@pytest.fixture()
def mods():
    sys.path.insert(0, str(BRIDGE_SRC))
    try:
        from swiss_robot_bridge import bridge_node, controllers, local_safety
        from swiss_robot_bridge.camera_manager import PantherCameraManager

        yield SimpleNamespace(bridge=bridge_node, controllers=controllers, safety=local_safety, Cam=PantherCameraManager)
    finally:
        sys.path.remove(str(BRIDGE_SRC))


class FakePub:
    def __init__(self) -> None:
        self.msgs = []

    def publish(self, msg) -> None:
        self.msgs.append(msg)


class Msg(SimpleNamespace):
    pass


def _point():
    return Msg(positions=[], time_from_start=Msg(sec=0, nanosec=0))


def _hardware(mods, head_order="pitch_yaw"):
    """A bridge adapter wired to fake publishers, as if rclpy were present."""
    hw = mods.bridge.ROS2HardwareAdapter(
        supervisor=mods.safety.LocalSafetySupervisor(head_order=head_order, head_full_range=True),
        camera_manager=mods.Cam(allow_sim_fallback=True),
    )
    hw._joint_traj_cls = lambda: Msg(joint_names=[], points=[])
    hw._joint_traj_point_cls = _point
    hw._f64_multi_cls = lambda: Msg(data=[])
    hw._arm_traj_pubs = {"left_arm_joint_trajectory_position_controller": FakePub(),
                         "right_arm_joint_trajectory_position_controller": FakePub()}
    hw._head_pub = FakePub()
    switched = []
    hw._switcher = SimpleNamespace(ensure_active=lambda needed: switched.append(list(needed)) or True)
    return hw, switched


def test_timed_stream_interpolates_and_finishes(mods) -> None:
    s = mods.controllers.TimedStream(points=[(0.0, (0.0, None)), (1.0, (1.0, 0.5))])
    assert s.sample(0.5) == (0.5, 0.5)  # None side takes the known value
    assert s.sample(2.0) == (1.0, 0.5) and not s.active()
    steps = mods.controllers.StepStream(events=[(1.0, "joint_rf", 0.67), (3.0, "joint_rf", 0.0)])
    assert steps.due(0.5) == [] and steps.due(1.0) == [("joint_rf", 0.67)] and steps.due(5) == [("joint_rf", 0.0)]


def test_gesture_goes_to_the_arm_trajectory_controller_after_switching_it_in(mods) -> None:
    hw, switched = _hardware(mods)
    names = tuple(f"joint_la{i}" for i in range(8)) + ("joint_head0", "joint_head1")
    pts = [(1.0, tuple([0.1] * 8 + [0.2, 0.3]), None), (2.0, tuple([0.2] * 8 + [0.0, 0.0]), None)]
    plan = mods.safety.PantherActuationPlan(command_id="g", action="generated_gesture",
                                             master_trajectory_joints=names, master_trajectory_points=pts)
    hw.publish_master_trajectory(plan)
    assert switched == [["left_arm_joint_trajectory_position_controller"]]
    left = hw._arm_traj_pubs["left_arm_joint_trajectory_position_controller"].msgs
    assert len(left) == 1 and left[0].joint_names == [f"joint_la{i}" for i in range(8)]
    assert [p.positions[0] for p in left[0].points] == [0.1, 0.2]
    assert hw._arm_traj_pubs["right_arm_joint_trajectory_position_controller"].msgs == []
    assert hw._head_stream.active()  # head streamed, not sent to a trajectory controller


def test_head_moves_are_streamed_in_controller_order(mods) -> None:
    hw, _ = _hardware(mods, head_order="pitch_yaw")
    hw.publish_head_targets({"joint_head1": 0.6})
    t_end = hw._head_stream.points[-1][0]
    hw.tick_streams(t_end + 0.01)
    assert hw._head_pub.msgs[-1].data == [pytest.approx(0.0), pytest.approx(0.6)]  # [pitch, yaw]
    hw2, _ = _hardware(mods, head_order="yaw_pitch")
    hw2.publish_head_targets({"joint_head1": 0.6})
    hw2.tick_streams(hw2._head_stream.points[-1][0] + 0.01)
    assert hw2._head_pub.msgs[-1].data == [pytest.approx(0.6), pytest.approx(0.0)]  # [yaw, pitch]


def test_head_ramp_respects_the_speed_cap(mods) -> None:
    hw, _ = _hardware(mods)
    hw.publish_head_targets({"joint_head0": 1.2})
    pts = hw._head_stream.points
    speeds = [abs(b[1][0] - a[1][0]) / (b[0] - a[0]) for a, b in zip(pts, pts[1:])]
    assert max(speeds) <= mods.bridge.HEAD_MAX_VEL_RPS + 1e-6


def test_stop_holds_a_moving_arm_and_stops_the_head(mods) -> None:
    hw, _ = _hardware(mods)
    names = tuple(f"joint_ra{i}" for i in range(8))
    plan = mods.safety.PantherActuationPlan(command_id="g", action="generated_gesture",
                                             master_trajectory_joints=names,
                                             master_trajectory_points=[(3.0, tuple([0.1] * 8), None)])
    hw.publish_master_trajectory(plan)
    hw.publish_head_targets({"joint_head1": 1.0})
    hw.joint_positions.update({n: 0.05 for n in names})
    hw.hold_arms_if_moving()
    right = hw._arm_traj_pubs["right_arm_joint_trajectory_position_controller"].msgs
    assert right[-1].points[0].positions == [0.05] * 8  # hold where it is
    assert not hw._head_stream.active()
