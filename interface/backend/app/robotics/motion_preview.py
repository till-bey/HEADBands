"""What a validated command will look like, for the Web UI simulation preview only.

The robot bridge never receives this (robot_gateway leaves it out): it plans the real motion itself from the
command. So the preview follows the bridge's rules (local_safety.validate_and_compute_panther_plan,
throw_points, retime_from_current_pose) closely but not exactly: the bridge's throw speed
(PANTHER_THROW_SPEED) and head range (PANTHER_HEAD_FULL_RANGE) are its own settings.
"""

from __future__ import annotations

import math
from pathlib import Path

from app.robotics.action_schema import (
    AllowedActionType,
    GestureTrajectoryPoint,
    MotionPreview,
    ValidatedRobotCommand,
)
from app.robotics.head_limits import HEAD_PITCH_JOINT, HEAD_YAW_JOINT, clamp_pitch, clamp_yaw

# Copies of motion/retarget/*.csv, the files the bridge plays (panther_topics.THROW_FILES); copy again if they change.
THROW_DIR = Path(__file__).resolve().parent / "data" / "throws"
THROW_FILES = {
    "standard": "joints_trajectory_shoulder_rel_3_v6.csv",
    "early": "joints_trajectory_shoulder_rel_3_v5_early.csv",
}
THROW_PAUSE_S = 1.0  # bridge: hold the first row 1 s, send its gripper value, hold 1 s more
THROW_JOINTS = [f"joint_ra{i}" for i in range(8)] + ["joint_rf"]
CAMERA_HEIGHT_M = 1.05  # bridge _head_pitch_yaw
ROTATE_SPEED_RPS = 0.4  # bridge rotate
BRIDGE_MAX_SPEED_MPS = 0.5  # bridge default max_linear_speed_mps
LIFT_RANGE_M = (0.0, 0.8)
RIGHT_WORDS = ("right", "rechts", "rächts", "droite", "derecha", "prav", "右")


def _throw_rows(variant: str | None) -> list[list[float]]:
    path = THROW_DIR / THROW_FILES.get(variant or "standard", THROW_FILES["standard"])
    rows = [[float(v) for v in line.split(",")] for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if not rows or any(len(r) != 10 for r in rows):
        raise ValueError(f"{path.name}: rows of t + 9 values expected")
    return rows


def throw_preview(variant: str | None) -> MotionPreview:
    """The throw as the bridge plays it at PANTHER_THROW_SPEED=1: two 1 s holds on the first row, then the file."""
    rows = _throw_rows(variant)
    first = rows[0][1:]
    pts = [(0.0, first), (THROW_PAUSE_S, first)] + [(r[0] + 2 * THROW_PAUSE_S, r[1:]) for r in rows]
    return MotionPreview(
        name=f"throw ({variant or 'standard'})",
        duration_s=round(pts[-1][0], 3),
        joint_names=THROW_JOINTS,
        points=[GestureTrajectoryPoint(t=round(t, 4), positions=list(q)) for t, q in pts],
    )


def _head(cmd: ValidatedRobotCommand) -> tuple[float, float]:
    """(pitch, yaw) as the bridge aims: explicit angles win, else toward the target, else a gentle look down."""
    pitch, yaw = 0.20, cmd.yaw_rad or 0.0
    if cmd.target is not None:
        if abs(cmd.target.x) > 1e-3 or abs(cmd.target.y) > 1e-3:
            yaw = math.atan2(cmd.target.y, max(cmd.target.x, 0.15))
        pitch = math.atan2(CAMERA_HEIGHT_M - cmd.target.z, max(math.hypot(cmd.target.x, cmd.target.y), 0.25))
    if cmd.head_pitch_rad is not None:
        pitch = cmd.head_pitch_rad
    if cmd.head_yaw_rad is not None:
        yaw = cmd.head_yaw_rad
    # shortcut: clamped to the cloud's head range; the bridge clamps tighter until PANTHER_HEAD_FULL_RANGE=1
    return clamp_pitch(pitch), clamp_yaw(yaw)


def _gripper_joint(cmd: ValidatedRobotCommand) -> str:
    text = f"{cmd.object_label or ''} {cmd.rationale or ''}".lower()
    right = any(w in text for w in RIGHT_WORDS) or (cmd.target is not None and cmd.target.y < -0.08)
    return "joint_rf" if right else "joint_lf"


def build_preview(cmd: ValidatedRobotCommand) -> MotionPreview | None:
    """Preview of one command, or None when nothing visible moves (stop)."""
    act = cmd.action
    if act == AllowedActionType.GENERATED_GESTURE and cmd.gesture is not None:
        g = cmd.gesture
        return MotionPreview(name=g.name, duration_s=g.duration_s, joint_names=g.joint_names, points=g.points)
    if act == AllowedActionType.THROW_OBJECT:
        p = throw_preview(cmd.throw_variant)
    elif act in (AllowedActionType.MOVE_TO, AllowedActionType.RETURN_HOME) and cmd.target is not None:
        dist = math.hypot(cmd.target.x, cmd.target.y)
        speed = min(cmd.speed, BRIDGE_MAX_SPEED_MPS)
        p = MotionPreview(name=f"{act.value.replace('_', ' ')} {dist:.2f} m", duration_s=0.0)
        if dist > 1e-3 and speed > 1e-4:
            p.base_path = [[0.0, 0.0, 0.0, 0.0], [round(dist / speed, 3), cmd.target.x, cmd.target.y, 0.0]]
    elif act == AllowedActionType.ROTATE and cmd.yaw_rad:
        p = MotionPreview(name=f"rotate {math.degrees(cmd.yaw_rad):.0f}°", duration_s=0.0,
                          base_path=[[0.0, 0.0, 0.0, 0.0], [round(abs(cmd.yaw_rad) / ROTATE_SPEED_RPS, 3), 0.0, 0.0, cmd.yaw_rad]])
    else:
        p = MotionPreview(name=act.value.replace("_", " "), duration_s=0.0)

    # head, gripper and lift targets: one end pose; the player eases there at a safe speed
    end: dict[str, float] = {}
    if act == AllowedActionType.MOVE_HEAD:
        if cmd.head_pitch_rad is not None:
            end[HEAD_PITCH_JOINT] = clamp_pitch(cmd.head_pitch_rad)
        if cmd.head_yaw_rad is not None:
            end[HEAD_YAW_JOINT] = clamp_yaw(cmd.head_yaw_rad)
    elif act in (AllowedActionType.LOOK_AT, AllowedActionType.INSPECT_OBJECT, AllowedActionType.PICK_OBJECT,
                 AllowedActionType.PLACE_OBJECT, AllowedActionType.THROW_OBJECT):
        end[HEAD_PITCH_JOINT], end[HEAD_YAW_JOINT] = _head(cmd)
    if act == AllowedActionType.PICK_OBJECT:
        end[_gripper_joint(cmd)] = 1.0
    elif act == AllowedActionType.PLACE_OBJECT:
        end[_gripper_joint(cmd)] = 0.0
    if cmd.target is not None and cmd.target.z > 0.01 and act in (
        AllowedActionType.MOVE_TO, AllowedActionType.RETURN_HOME, AllowedActionType.PICK_OBJECT,
        AllowedActionType.PLACE_OBJECT, AllowedActionType.THROW_OBJECT,
    ):
        end["body_joint"] = min(max(cmd.target.z, LIFT_RANGE_M[0]), LIFT_RANGE_M[1])
    end = {k: v for k, v in end.items() if k not in p.joint_names}  # the throw file already drives joint_rf
    if end:
        names = p.joint_names + list(end)
        if p.points:
            p.points = [GestureTrajectoryPoint(t=pt.t, positions=pt.positions + list(end.values())) for pt in p.points]
        else:
            p.points = [GestureTrajectoryPoint(t=0.0, positions=list(end.values()))]
        p.joint_names = names
    if not p.points and not p.base_path:
        return None
    p.duration_s = round(max([p.duration_s] + [pt.t for pt in p.points] + [b[0] for b in p.base_path]), 3)
    return p
