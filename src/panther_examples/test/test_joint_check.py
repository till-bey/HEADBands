from panther_examples.joint_check import step_target


def test_step_target_goes_toward_more_room():
    assert step_target(0.0, -1.0, 0.5, 0.1) == -0.1
    assert step_target(0.0, -0.5, 1.0, 0.1) == 0.1
    assert abs(step_target(0.95, -1.0, 1.0, 0.1) - 0.85) < 1e-9


def test_step_target_none_without_room():
    assert step_target(0.0, -0.05, 0.05, 0.1) is None
