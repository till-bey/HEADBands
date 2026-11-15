"""Generate arm gestures ("salute", "wave", "point at the door") from language, safely.

The language model never produces joint angles. It describes the gesture as a few hand
keyframes (gripper base position in `body_link`, the torso frame that rides on the lift),
and this module turns them into a joint trajectory itself:

1. Gemini (or the mock) returns `GestureKeyframePlan`: times, arm, hand x/y/z, gripper.
2. Damped-least-squares IK on the Panther URDF arm chains, seeded sample to sample.
3. Checks, any failure refuses the gesture:
   - every sample reached within 3 cm, joints inside the URDF limits;
   - time is stretched until no joint exceeds MAX_JOINT_VEL (0.45 rad/s, under the 0.5 rad/s
     cartesian_follow cap);
   - hands, wrists and elbows keep CLEARANCE_M from approximate head, neck and torso boxes,
     and the two hands stay HAND_SEPARATION_M apart.
4. On failure the model gets the reasons and may try again (MAX_ATTEMPTS).

The result is a `GestureTrajectory` that the safety gateway validates again and only sends to
the robot after a human confirms it. The bridge re-checks limits and speed before playback.
Collision boxes are stand-ins sized from the RViz shapes, not measured geometry.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings
from app.core.logging import get_logger
from app.robotics.action_schema import GestureTrajectory, GestureTrajectoryPoint

logger = get_logger(__name__)

# Verbatim copy of HEADBands clean_up src/panther_sim/urdf/robot_description.urdf (the robot's own
# /robot_description). Only joints are read; meshes, ros2_control and gazebo blocks are ignored.
URDF_PATH = Path(__file__).resolve().parent / "data" / "robot_description.urdf"
ARM_JOINTS = {
    "left": tuple(f"joint_la{i}" for i in range(8)),
    "right": tuple(f"joint_ra{i}" for i in range(8)),
}
ARM_TIP_LINK = {"left": "link_lf", "right": "link_rf"}
MAX_JOINT_VEL = 0.45  # rad/s
SAMPLE_DT = 0.1  # s, Cartesian interpolation step between keyframes
REACH_TOL_M = 0.03
CLEARANCE_M = 0.03
HAND_SEPARATION_M = 0.10
MAX_DURATION_S = 30.0
MAX_KEYFRAMES = 16
MAX_ATTEMPTS = 3
RETURN_SECONDS = 2.0

# Approximate obstacle boxes in body_link (x forward, y left, z up), (min xyz, max xyz).
# Head: the RViz stand-in box on link_head1 (head joints at zero). Torso: the column under the shoulders.
OBSTACLES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "head": ((-0.01, -0.10, 0.17), (0.135, 0.10, 0.335)),
    "neck": ((-0.03, -0.05, 0.05), (0.07, 0.05, 0.17)),
    "torso": ((-0.13, -0.16, -0.80), (0.11, 0.16, 0.06)),
}


# --------------------------------------------------------------------------- model output schema
class HandKeyframe(BaseModel):
    """One pose of one hand: where the gripper base should be at time t."""

    model_config = ConfigDict(extra="ignore")

    t: float = Field(description="Seconds from the start of the gesture, > 0, increasing per arm.")
    arm: Literal["left", "right"]
    x: float = Field(description="Forward from the torso origin, meters.")
    y: float = Field(description="Left of the torso origin, meters (right arm is negative).")
    z: float = Field(description="Up from shoulder height, meters (arm hanging is about -0.68).")
    gripper: float | None = Field(default=None, description="0 open .. 1 closed, null to keep.")
    note: str = Field(default="", max_length=120)


class GestureKeyframePlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(max_length=60, description="Short gesture name, e.g. 'salute'.")
    summary: str = Field(default="", max_length=240)
    keyframes: list[HandKeyframe] = Field(default_factory=list)
    return_to_start: bool = True


GESTURE_SYSTEM_INSTRUCTION = """You design expressive arm gestures for the UniX AI Panther humanoid robot.
You NEVER output joint angles, code, or ROS topics. You only output hand keyframes; a planner solves
inverse kinematics, checks joint limits, speed and collisions, and a human confirms before anything moves.

Frame `body_link` (meters): origin between the shoulders at shoulder height; +x forward, +y to the robot's
LEFT, +z up. The hand point is the gripper base; fingertips are about 8 cm further along the forearm.
- Shoulders: left at (0, +0.27, 0), right at (0, -0.27, 0). Each hand reaches at most ~0.65 m from its shoulder.
- Rest: arms hang straight down, hands at (0.0, +/-0.275, -0.68).
- Head occupies x in [-0.01, 0.135], y in [-0.10, 0.10], z in [0.17, 0.335]. The face looks along +x;
  its front plane is x = 0.135 and the forehead is around z = 0.28. Keep hands at least 5 cm outside the head.
- Torso occupies x in [-0.13, 0.11], y in [-0.16, 0.16], z below 0.06. Keep hands at least 5 cm outside it.
- The right arm cannot cross far to the left of y = -0.05, and the left arm not right of y = +0.05.
- Hands must stay at least 10 cm apart from each other.
Rules:
- Use 2 to 6 keyframes per arm used; only include arms that move. Times in seconds from start, increasing,
  first keyframe at t >= 1.0; whole gesture under 12 s. Slow, smooth, natural.
- Repeated motions (waving) alternate between two hand positions about 8-12 cm apart, 3 to 4 times.
- Set return_to_start=true unless the user asks to hold the pose.
Examples:
- "salute" (right hand): right hand to the forehead, in front of and slightly right of it:
  t=2.0 (0.20, -0.08, 0.27), hold t=3.5 same pose. return_to_start=true.
- "wave" (right hand): raise the right hand beside the head: t=2.0 (0.15, -0.40, 0.25), then
  t=2.8 (0.15, -0.30, 0.27), t=3.6 (0.15, -0.40, 0.25), t=4.4 (0.15, -0.30, 0.27), t=5.2 (0.15, -0.40, 0.25).
- "raise both arms": t=3.0 left (0.10, 0.35, 0.45) and right (0.10, -0.35, 0.45), hold until t=4.5.
"""


# --------------------------------------------------------------------------- kinematics
def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle(axis: np.ndarray, a: float) -> np.ndarray:
    x, y, z = axis
    c, s, t = math.cos(a), math.sin(a), 1 - math.cos(a)
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def _tf(rot: np.ndarray | None = None, trans: np.ndarray | None = None) -> np.ndarray:
    out = np.eye(4)
    if rot is not None:
        out[:3, :3] = rot
    if trans is not None:
        out[:3, 3] = trans
    return out


@dataclass(frozen=True)
class _Joint:
    name: str
    revolute: bool
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


@dataclass
class ArmChain:
    """Joints from body_link to the gripper base of one arm."""

    arm: str
    joints: list[_Joint]
    names: tuple[str, ...] = field(init=False)
    lower: np.ndarray = field(init=False)
    upper: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rev = [j for j in self.joints if j.revolute]
        self.names = tuple(j.name for j in rev)
        self.lower = np.array([j.lower for j in rev])
        self.upper = np.array([j.upper for j in rev])

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Tip position, plus position and world axis of every revolute joint (in order)."""
        T = np.eye(4)
        positions, axes = [], []
        k = 0
        for j in self.joints:
            T = T @ j.origin
            if j.revolute:
                positions.append(T[:3, 3].copy())
                axes.append(T[:3, :3] @ j.axis)
                T = T @ _tf(_axis_angle(j.axis, float(q[k])))
                k += 1
        return T[:3, 3].copy(), positions, axes

    def body_points(self, q: np.ndarray) -> list[np.ndarray]:
        """Points along the forearm and hand used for collision checks (elbow onward)."""
        tip, pos, _ = self.fk(q)
        elbow, wrist = pos[4], pos[6]
        return [elbow, (elbow + wrist) / 2, wrist, (wrist + tip) / 2, tip]


@lru_cache(maxsize=1)
def load_arm_chains(urdf_path: str = str(URDF_PATH)) -> dict[str, ArmChain]:
    root = ET.parse(urdf_path).getroot()
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    chains: dict[str, ArmChain] = {}
    for arm, tip in ARM_TIP_LINK.items():
        seq = []
        link = tip
        while link in by_child and link != "body_link":
            j = by_child[link]
            seq.append(j)
            link = j.find("parent").get("link")
        if link != "body_link":
            raise RuntimeError(f"{tip} does not hang off body_link in {urdf_path}")
        joints: list[_Joint] = []
        for j in reversed(seq):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            a = j.find("axis")
            axis = np.array([float(v) for v in (a.get("xyz") if a is not None else "1 0 0").split()])
            axis = axis / (np.linalg.norm(axis) or 1.0)
            lim = j.find("limit")
            joints.append(_Joint(
                name=j.get("name"),
                revolute=j.get("type") == "revolute",
                origin=_tf(_rpy(*rpy), xyz),
                axis=axis,
                lower=float(lim.get("lower", 0)) if lim is not None else 0.0,
                upper=float(lim.get("upper", 0)) if lim is not None else 0.0,
            ))
        chain = ArmChain(arm, joints)
        if chain.names != ARM_JOINTS[arm]:
            raise RuntimeError(f"Unexpected {arm} arm joints {chain.names}")
        chains[arm] = chain
    return chains


def joint_limits() -> dict[str, tuple[float, float]]:
    """URDF limits of every arm joint, for the validator."""
    out: dict[str, tuple[float, float]] = {}
    for chain in load_arm_chains().values():
        for name, lo, hi in zip(chain.names, chain.lower, chain.upper, strict=True):
            out[name] = (float(lo), float(hi))
    return out


def _bent_seed(arm: str) -> np.ndarray:
    q = np.zeros(8)
    q[1], q[2], q[4] = -0.7, (-0.25 if arm == "right" else 0.25), -1.3
    return q


def solve_ik(chain: ArmChain, target: np.ndarray, seed: np.ndarray, iters: int = 150) -> tuple[np.ndarray, float]:
    """Damped least squares to a position target, pulled softly toward the seed for continuity."""
    q = np.clip(seed.astype(float).copy(), chain.lower, chain.upper)
    lam2, mu2 = 0.03**2, 0.02**2
    err = float("inf")
    for _ in range(iters):
        tip, pos, axes = chain.fk(q)
        e = target - tip
        err = float(np.linalg.norm(e))
        if err < 1e-3:
            break
        J = np.stack([np.cross(ax, tip - p) for ax, p in zip(axes, pos, strict=True)], axis=1)
        A = J.T @ J + (lam2 + mu2) * np.eye(8)
        b = J.T @ e + mu2 * (seed - q)
        dq = np.linalg.solve(A, b)
        step = np.max(np.abs(dq))
        if step > 0.25:
            dq *= 0.25 / step
        q = np.clip(q + dq, chain.lower, chain.upper)
    return q, err


def solve_ik_robust(chain: ArmChain, target: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, float]:
    """Seeded solve, then a bent-elbow and a few random restarts if it falls short."""
    best_q, best_err = solve_ik(chain, target, seed)
    if best_err <= REACH_TOL_M:
        return best_q, best_err
    rng = np.random.default_rng(7)
    seeds = [_bent_seed(chain.arm)] + [rng.uniform(chain.lower, chain.upper) for _ in range(12)]
    for s in seeds:
        q, err = solve_ik(chain, target, s)
        if err < best_err - 1e-4 or (abs(err - best_err) < 1e-4 and np.linalg.norm(q - seed) < np.linalg.norm(best_q - seed)):
            best_q, best_err = q, err
        if best_err <= REACH_TOL_M:
            break
    return best_q, best_err


def _box_distance(p: np.ndarray, box: tuple[tuple[float, float, float], tuple[float, float, float]]) -> float:
    lo, hi = np.array(box[0]), np.array(box[1])
    d = np.maximum(np.maximum(lo - p, 0.0), p - hi)
    outside = float(np.linalg.norm(d))
    if outside > 0:
        return outside
    return -float(np.min(np.minimum(p - lo, hi - p)))  # negative = inside


# --------------------------------------------------------------------------- planning
@dataclass
class GesturePlanResult:
    ok: bool
    reasons: list[str]
    trajectory: GestureTrajectory | None = None
    keyframe_plan: GestureKeyframePlan | None = None
    attempts: int = 1


def plan_from_keyframes(
    plan: GestureKeyframePlan,
    description: str,
    current_positions: dict[str, float] | None = None,
) -> GesturePlanResult:
    """IK + timing + collision checks for model keyframes. Pure function, no network."""
    chains = load_arm_chains()
    reasons: list[str] = []
    current_positions = current_positions or {}
    frames = sorted(plan.keyframes, key=lambda k: (k.arm, k.t))[:MAX_KEYFRAMES]
    arms = sorted({k.arm for k in frames})
    if not arms:
        return GesturePlanResult(False, ["The plan has no keyframes."], keyframe_plan=plan)
    for k in frames:
        if not all(math.isfinite(v) for v in (k.t, k.x, k.y, k.z)) or k.t <= 0:
            return GesturePlanResult(False, [f"Keyframe {k.arm} t={k.t} is not finite or not after 0."], keyframe_plan=plan)

    end_t = max(k.t for k in frames)
    if plan.return_to_start:
        end_t += RETURN_SECONDS
    times = np.round(np.arange(SAMPLE_DT, end_t + 1e-9, SAMPLE_DT), 4)
    if end_t > MAX_DURATION_S:
        return GesturePlanResult(False, [f"Gesture lasts {end_t:.1f} s, over {MAX_DURATION_S:.0f} s."], keyframe_plan=plan)

    q_start: dict[str, np.ndarray] = {}
    joint_tracks: dict[str, np.ndarray] = {}
    tip_tracks: dict[str, np.ndarray] = {}
    for arm in arms:
        chain = chains[arm]
        q0 = np.array([float(current_positions.get(n, 0.0)) for n in chain.names])
        q0 = np.clip(q0, chain.lower, chain.upper)
        q_start[arm] = q0
        kfs = [k for k in frames if k.arm == arm]
        # IK once per keyframe, each seeded from the previous solution (the first also tries a bent
        # elbow, since the hanging arm is a singular pose); then ease between them in joint space.
        # Joint-space segments never flip wrist solutions, and from rest the arm swings out and up
        # instead of dragging the hand across the torso. Collisions are checked along the result.
        key_t = [0.0]
        key_q = [q0]
        for k in kfs:
            target = np.array([k.x, k.y, k.z])
            q_k, err = solve_ik_robust(chain, target, key_q[-1] if len(key_q) > 1 else _bent_seed(arm))
            if err > REACH_TOL_M:
                reasons.append(
                    f"{arm} hand cannot reach keyframe t={k.t:g} s ({k.x:.2f}, {k.y:.2f}, {k.z:.2f}): "
                    f"{err * 100:.0f} cm short (each hand reaches about 0.65 m from its shoulder)."
                )
                break
            if k.t <= key_t[-1] + 1e-6:  # same time as the previous keyframe: treat as a hold
                key_q[-1] = q_k
                continue
            key_t.append(k.t)
            key_q.append(q_k)
        if reasons:
            break
        last_key_t = key_t[-1]
        qs, tips = [], []
        q_hold: np.ndarray | None = None  # pose at the last keyframe, start of return / hold
        for t in times:
            if t <= last_key_t + 1e-9:
                i = max(0, int(np.searchsorted(key_t, t, side="right")) - 1)
                i = min(i, len(key_t) - 2)
                t0, t1 = key_t[i], key_t[i + 1]
                a = 1.0 if t1 <= t0 else (t - t0) / (t1 - t0)
                a = 0.5 - 0.5 * math.cos(math.pi * min(max(a, 0.0), 1.0))  # ease in/out
                qs.append(key_q[i] + a * (key_q[i + 1] - key_q[i]))
            else:
                # After this arm's last keyframe: ease back to the start pose in joint space, or hold.
                if q_hold is None:
                    q_hold = qs[-1].copy() if qs else q0.copy()
                if plan.return_to_start:
                    a = min(1.0, (t - last_key_t) / RETURN_SECONDS)
                    a = 0.5 - 0.5 * math.cos(math.pi * a)
                    qs.append(q_hold + a * (q0 - q_hold))
                else:
                    qs.append(q_hold.copy())
            tips.append(chain.fk(qs[-1])[0])
        if reasons:
            break
        joint_tracks[arm] = np.array(qs)
        tip_tracks[arm] = np.array(tips)
    if reasons:
        return GesturePlanResult(False, reasons, keyframe_plan=plan)

    # Collisions: forearm and hand points vs head / neck / torso, and hand-to-hand distance.
    min_clear = float("inf")
    for arm in arms:
        chain = chains[arm]
        for idx, q in enumerate(joint_tracks[arm]):
            for p in chain.body_points(q):
                for name, box in OBSTACLES.items():
                    d = _box_distance(p, box)
                    min_clear = min(min_clear, d)
                    if d < CLEARANCE_M:
                        reasons.append(
                            f"{arm} arm comes {max(d, 0) * 100:.0f} cm from the {name} at t={times[idx]:.1f} s "
                            f"(hand at {tip_tracks[arm][idx][0]:.2f}, {tip_tracks[arm][idx][1]:.2f}, {tip_tracks[arm][idx][2]:.2f}); "
                            f"needs {CLEARANCE_M * 100:.0f} cm."
                        )
                        break
                if reasons:
                    break
            if reasons:
                break
    if not reasons and len(arms) == 2:
        sep = np.linalg.norm(tip_tracks["left"] - tip_tracks["right"], axis=1)
        if float(sep.min()) < HAND_SEPARATION_M:
            i = int(sep.argmin())
            reasons.append(f"hands come {sep[i] * 100:.0f} cm apart at t={times[i]:.1f} s; need {HAND_SEPARATION_M * 100:.0f} cm.")
    if reasons:
        return GesturePlanResult(False, reasons[:3], keyframe_plan=plan)

    # Timing: stretch until the fastest joint stays under MAX_JOINT_VEL, from the start pose on.
    names: list[str] = []
    cols: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    for arm in arms:
        names.extend(chains[arm].names)
        cols.append(joint_tracks[arm])
        starts.append(q_start[arm])
    Q = np.concatenate(cols, axis=1)
    Q_full = np.vstack([np.concatenate(starts), Q])
    T_full = np.concatenate([[0.0], times])
    # Stretch only the intervals that are too fast (raising an arm from rest), not the whole gesture,
    # so a wave keeps its rhythm once the hand is up.
    dq = np.abs(np.diff(Q_full, axis=0)).max(axis=1)
    dt = np.maximum(np.diff(T_full), dq / MAX_JOINT_VEL)
    out_times = np.cumsum(dt)
    peak = float((dq / dt).max()) if dq.size else 0.0
    scale = float(out_times[-1] / times[-1]) if times[-1] > 0 else 1.0
    if out_times[-1] > MAX_DURATION_S:
        return GesturePlanResult(
            False,
            [f"Slowed to {MAX_JOINT_VEL} rad/s the gesture would take {out_times[-1]:.0f} s; make the motion smaller."],
            keyframe_plan=plan,
        )

    traj = GestureTrajectory(
        name=plan.name[:60] or "gesture",
        description=description[:200],
        summary=plan.summary[:240],
        arms=list(arms),
        joint_names=names,
        points=[
            GestureTrajectoryPoint(t=round(float(t), 3), positions=[round(float(v), 5) for v in row])
            for t, row in zip(out_times, Q, strict=True)
        ],
        start_positions=[round(float(v), 5) for v in np.concatenate(starts)],
        keyframes=[k.model_dump() for k in frames],
        duration_s=round(float(out_times[-1]), 3),
        peak_joint_vel_rps=round(peak, 4),
        min_clearance_m=round(min_clear, 4) if math.isfinite(min_clear) else None,
        time_scale=round(scale, 3),
    )
    return GesturePlanResult(True, [], trajectory=traj, keyframe_plan=plan)


# --------------------------------------------------------------------------- providers
def _mock_keyframes(description: str) -> GestureKeyframePlan:
    d = description.lower()
    if any(w in d for w in ("wave", "wink", "salü", "hallo", "hello", "coucou", "saluda", "mávn")):
        xs = [(-0.40, 0.25), (-0.30, 0.27)] * 2 + [(-0.40, 0.25)]
        return GestureKeyframePlan(
            name="wave",
            summary="Right hand raised beside the head, waving.",
            keyframes=[HandKeyframe(t=2.0 + 0.8 * i, arm="right", x=0.15, y=y, z=z) for i, (y, z) in enumerate(xs)],
        )
    if any(w in d for w in ("both arms", "beide arme", "deux bras", "ambos brazos", "raise arms", "hands up")):
        kf = []
        for t in (3.0, 4.5):
            kf += [HandKeyframe(t=t, arm="left", x=0.10, y=0.35, z=0.45),
                   HandKeyframe(t=t, arm="right", x=0.10, y=-0.35, z=0.45)]
        return GestureKeyframePlan(name="raise both arms", summary="Both arms raised.", keyframes=kf)
    if any(w in d for w in ("point", "zeig", "montre", "señala")):
        return GestureKeyframePlan(
            name="point",
            summary="Right arm points forward.",
            keyframes=[HandKeyframe(t=2.5, arm="right", x=0.50, y=-0.25, z=0.05),
                       HandKeyframe(t=4.0, arm="right", x=0.50, y=-0.25, z=0.05)],
        )
    return GestureKeyframePlan(
        name="salute",
        summary="Right hand to the forehead, then back down.",
        keyframes=[HandKeyframe(t=2.0, arm="right", x=0.20, y=-0.08, z=0.27),
                   HandKeyframe(t=3.5, arm="right", x=0.20, y=-0.08, z=0.27)],
    )


class GesturePlanner:
    """Language -> hand keyframes (Gemini or mock) -> checked joint trajectory."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None
        self._use_mock = settings.robotics_provider == "mock" or settings.app_env == "test"

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(
                vertexai=True,
                project=self._settings.gcp_project_id,
                location=self._settings.gcp_robotics_location,
            )
        return self._client

    async def _ask_model(self, description: str, feedback: list[str]) -> GestureKeyframePlan:
        if self._use_mock:
            return _mock_keyframes(description)
        from google.genai import types

        prompt = f"Gesture request: {description}"
        if feedback:
            prompt += (
                "\nYour previous keyframes were refused by the planner:\n- "
                + "\n- ".join(feedback)
                + "\nAdjust the keyframes to fix every problem."
            )
        config = types.GenerateContentConfig(
            system_instruction=GESTURE_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=GestureKeyframePlan,
            temperature=0.2,
        )
        response = await self._get_client().aio.models.generate_content(
            model=self._settings.gemini_robotics_model,
            contents=[prompt],
            config=config,
        )
        if response.parsed and isinstance(response.parsed, GestureKeyframePlan):
            return response.parsed
        return GestureKeyframePlan.model_validate_json(response.text or "{}")

    async def plan(self, description: str, current_positions: dict[str, float] | None = None) -> GesturePlanResult:
        feedback: list[str] = []
        result = GesturePlanResult(False, ["No attempt made."])
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                kf = await self._ask_model(description, feedback)
            except Exception as exc:  # model or network failure
                logger.warning("Gesture keyframe generation failed", extra={"error": str(exc)})
                return GesturePlanResult(False, [f"Gesture model unavailable: {type(exc).__name__}"], attempts=attempt)
            result = plan_from_keyframes(kf, description, current_positions)
            result.attempts = attempt
            logger.info(
                "Gesture plan attempt",
                extra={"attempt": attempt, "ok": result.ok, "gesture": kf.name, "reasons": result.reasons},
            )
            if result.ok or self._use_mock:
                return result
            feedback = result.reasons
        return result
