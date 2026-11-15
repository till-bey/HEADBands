"""Combine saved camera mounting offsets with fresh robot feedback (no ROS)."""

import math


def camera_pose(settings, feedback=None):
    feedback = feedback or {}
    values = settings if isinstance(settings, dict) else vars(settings)
    base = values.get("camera_base_height", 0.83)
    height_offset = values.get("robot_z", 0.0)
    tilt_offset = values.get("tilt_rad", 0.0)

    def measured(value_key, fresh_key):
        value = feedback.get(value_key)
        fresh = bool(feedback.get(fresh_key) and isinstance(value, (int, float))
                     and math.isfinite(value))
        return (float(value) if fresh else 0.0), fresh

    body_height, lift_fresh = measured("lift_position", "lift_fresh")
    head_tilt, head_fresh = measured("head_pitch", "head_fresh")
    return {
        "base_height": base, "height_offset": height_offset,
        "body_height": body_height, "tilt_offset": tilt_offset, "head_tilt": head_tilt,
        "height": base + height_offset + body_height,
        "tilt": tilt_offset + head_tilt,
        "lift_fresh": lift_fresh, "head_fresh": head_fresh,
    }
