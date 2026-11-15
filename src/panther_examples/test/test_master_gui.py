from types import SimpleNamespace as C

import pytest

from panther_examples.master_gui import JOINTS, OURS, holders, move_time


def test_move_time_slowest_joint_sets_the_pace():
    assert move_time({"joint_la0": 0.0, "joint_lf": 0.0}, {"joint_la0": 1.0, "joint_lf": 1.0}) == pytest.approx(2.0)
    assert move_time({"joint_la0": 0.0}, {"joint_la0": 0.01}) == 0.3  # floor for tiny moves


def test_holders_only_active_controllers_on_our_joints():
    states = {c.name: c for c in [
        C(name="dual_arm_forward_position_controller", state="active", claimed_interfaces=["joint_la0/position"]),
        C(name="left_unix_gripper3_controller", state="active", claimed_interfaces=["joint_lf/position"]),
        C(name="lifting_controller", state="active", claimed_interfaces=["body_joint/position"]),
        C(name="head_joint_trajectory_controller", state="active", claimed_interfaces=["joint_head0/position"]),
        C(name="right_arm_joint_trajectory_position_controller", state="inactive", claimed_interfaces=[]),
    ]}
    assert holders(states, JOINTS, OURS) == ["dual_arm_forward_position_controller", "head_joint_trajectory_controller"]


LIMITS = {n: (-2.0, 2.0) for n in JOINTS} | {"joint_lf": (0.0, 1.0), "joint_rf": (0.0, 1.0), "body_joint": (0.0, 0.8)}


def test_csv_round_trip_and_plan(tmp_path):
    from panther_examples.master_gui import check, peak_speed, plan, play_frames, read_csv, write_csv
    from panther_examples.robot import LEFT_TRAJ, RIGHT_TRAJ
    path = tmp_path / "rec.csv"
    write_csv(path, ["joint_ra1", "joint_head0", "joint_rf", "body_joint"],
              [[0.0, 0.0, 0.0, 0.0, 0.5], [1.0, 1.0, 0.2, 0.5, 0.5]])
    names, rows = read_csv(path)
    assert names == ["joint_ra1", "joint_head0", "joint_rf", "body_joint"] and rows[1] == [1.0, 1.0, 0.2, 0.5, 0.5]
    check(names, rows, LIMITS)
    assert peak_speed(names, rows) == (1.0, "joint_ra1")
    now = {n: 0.3 for n in JOINTS} | {"joint_rf": 0.0}
    times, frames = play_frames(names, rows, now, LIMITS)
    assert times == pytest.approx([0.0, 0.6, 1.6])  # 0.3 rad at 0.5 rad/s to the first row, then the file's 1 s
    arms, head, grip = plan(times, frames)
    right = arms[RIGHT_TRAJ].points
    assert [p.positions[1] for p in right] == [0.0, 1.0]  # joint_ra1, frame 0 (now) not sent: the controller starts there
    assert all(p.positions[0] == 0.3 for p in arms[LEFT_TRAJ].points)  # not in the file: holds
    assert [q for _t, q in head] == [[0.3, 0.3], [0.0, 0.3], [0.2, 0.3]]  # interpolated from where it is now
    assert grip["joint_rf"] == [(pytest.approx(0.6), 0.5, pytest.approx(0.5))] and grip["joint_lf"] == []


def test_check_refuses_bad_files():
    from panther_examples.master_gui import check
    with pytest.raises(ValueError, match="strictly"):
        check(["joint_ra1"], [[0.0, 0.0], [0.0, 0.1]], LIMITS)
    with pytest.raises(ValueError, match="limits"):
        check(["joint_ra1"], [[0.0, 2.1]], LIMITS)
    with pytest.raises(ValueError, match="unknown"):
        check(["joint_xx"], [[0.0, 0.0]], LIMITS)


def test_reads_retarget_files():
    import os
    from panther_examples.master_gui import read_csv
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "motion", "retarget", "joints_trajectory_shoulder_rel_3_v5.csv")
    if not os.path.exists(path):
        pytest.skip("retarget output not in this checkout")
    names, rows = read_csv(path)
    assert names[0] == "joint_ra0" and names[-1] == "joint_rf" and len(rows[0]) == 10


def test_right_gripper_clamped_not_refused():
    from panther_examples.master_gui import check, play_frames
    limits = LIMITS | {"joint_rf": (0.0, 0.67)}
    check(["joint_rf"], [[0.0, 1.0]], limits)  # a file closing it fully still plays
    _times, frames = play_frames(["joint_rf"], [[0.0, 1.0]], {n: 0.0 for n in JOINTS}, limits)
    assert frames[-1]["joint_rf"] == 0.67
