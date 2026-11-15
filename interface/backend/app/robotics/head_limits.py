"""Panther head joint limits and speed, shared by the parser, ER 2 sanitizer, gesture planner and validator.

URDF (src/panther_sim/urdf/robot_description.urdf) and the user guide:
  joint_head0 = pitch, axis +y: -0.348 (20 deg up) .. +1.5 (86 deg down); positive looks DOWN.
  joint_head1 = yaw,   axis +z: -1.57 .. +1.57 (+-90 deg);                positive looks LEFT.
We keep 5 deg inside each hardware limit. The robot bridge applies its own, tighter range until
PANTHER_HEAD_FULL_RANGE=1 is set there (after the head command order is checked on the robot).
"""

from __future__ import annotations

import math

HEAD_PITCH_JOINT = "joint_head0"
HEAD_YAW_JOINT = "joint_head1"
HEAD_JOINTS = (HEAD_PITCH_JOINT, HEAD_YAW_JOINT)

_MARGIN = math.radians(5.0)
HEAD_PITCH_MIN_RAD = round(-0.348 + _MARGIN, 4)  # about 15 deg up
HEAD_PITCH_MAX_RAD = round(1.5 - _MARGIN, 4)  # about 81 deg down
HEAD_YAW_MAX_RAD = round(1.57 - _MARGIN, 4)  # about +-85 deg

HEAD_LIMITS: dict[str, tuple[float, float]] = {
    HEAD_PITCH_JOINT: (HEAD_PITCH_MIN_RAD, HEAD_PITCH_MAX_RAD),
    HEAD_YAW_JOINT: (-HEAD_YAW_MAX_RAD, HEAD_YAW_MAX_RAD),
}

HEAD_MAX_VEL_RPS = 0.6  # planned head speed
HEAD_HARD_MAX_VEL_RPS = 0.7  # validator / bridge cap


def clamp_pitch(value: float) -> float:
    return max(HEAD_PITCH_MIN_RAD, min(HEAD_PITCH_MAX_RAD, value))


def clamp_yaw(value: float) -> float:
    return max(-HEAD_YAW_MAX_RAD, min(HEAD_YAW_MAX_RAD, value))
