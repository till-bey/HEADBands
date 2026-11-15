from panther_examples.can_state import drive_state, mode_name


def test_drive_states():
    assert drive_state(0x0237) == "OPERATION ENABLED (torque)"
    assert drive_state(0x0233) == "switched on (no torque)"
    assert drive_state(0x0250) == "switch on disabled (no torque)"
    assert drive_state(0x0218) == "FAULT"


def test_modes():
    assert mode_name(8) == "cyclic sync position" and mode_name(7) == "interpolated position"
    assert mode_name(0xFF) == "mode -1"
