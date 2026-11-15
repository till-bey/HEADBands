"""Catching pose computed from the URDF, and small corrections toward the predicted catch point.

Pose: both grippers in front of the chest, on a vertical catching plane x = center[0] (body_link), at
y = +/- half_spacing around the center, fingers pointing at the thrower (gripper -z along body +x) and closing
vertically (gripper y along body z), so a disc flying flat is pinched at its front rim by both hands.
The two-finger gripper is symmetric, so a hand turned 180 deg about its finger axis grips the same way; the
mirrored right arm needs that.

Correction: both hands shift together by (dy, dz) in the catching plane, clamped to max_correction_m, so the
center between them meets the predicted crossing point. Solved by IK seeded from the current joints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from frisbee_interception.core.kinematics import ArmChain, rot_error

ARMS = {"left": ("link_lf", +1.0), "right": ("link_rf", -1.0)}
# gripper axes (x, y, z) expressed in body_link: fingers forward, closing axis vertical
R_PINCH = np.column_stack([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
R_PINCH_FLIPPED = R_PINCH @ np.diag([-1.0, -1.0, 1.0])


@dataclass
class CatchPoseConfig:
    base: str = "body_link"
    center: list = field(default_factory=lambda: [0.45, 0.0, -0.20])  # between the hands, body_link (m)
    half_spacing: float = 0.13  # hand y = center_y +/- this (disc radius ~0.135)
    max_correction_m: float = 0.10  # largest in-plane shift toward the predicted point
    rot_tol_rad: float = 0.17  # corrected hands may tilt this much (~10 deg) and still pinch the rim
    samples: int = 20000  # FK samples used to seed IK
    seed: int = 3

    @classmethod
    def from_dict(cls, d: dict) -> CatchPoseConfig:
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                cur = getattr(cfg, k)
                setattr(cfg, k, [float(x) for x in v] if isinstance(cur, list) else type(cur)(v))
        return cfg


class CatchPosePlanner:
    def __init__(self, urdf: str, cfg: CatchPoseConfig | None = None) -> None:
        self.cfg = cfg or CatchPoseConfig()
        self.chains = {arm: ArmChain(urdf, self.cfg.base, tip) for arm, (tip, _) in ARMS.items()}
        self.rot = {}
        self.nominal: dict[str, np.ndarray] = {}

    @property
    def joint_names(self) -> dict[str, list[str]]:
        return {arm: ch.names for arm, ch in self.chains.items()}

    def hand_target(self, arm: str, dy: float = 0.0, dz: float = 0.0) -> np.ndarray:
        c = np.array(self.cfg.center, float)
        return np.array([c[0], c[1] + ARMS[arm][1] * self.cfg.half_spacing + dy, c[2] + dz])

    def _seeds(self, arm: str, p: np.ndarray, R: np.ndarray, k: int) -> list[np.ndarray]:
        ch = self.chains[arm]
        rng = np.random.default_rng(self.cfg.seed)
        Q = rng.uniform(ch.lo, ch.hi, (self.cfg.samples, ch.n))
        T = [ch.fk(q) for q in Q]
        cost = np.array([np.linalg.norm(t[:3, 3] - p) + 0.15 * np.linalg.norm(rot_error(R, t[:3, :3])) for t in T])
        return [Q[i] for i in np.argsort(cost)[:k]]

    def solve_nominal(self, k: int = 12) -> dict[str, np.ndarray]:
        """Nominal pose for both arms; among IK solutions, the one whose corrections need the least motion."""
        m = self.cfg.max_correction_m
        grid = [(dy, dz) for dy in (-m, 0.0, m) for dz in (-m, 0.0, m) if (dy, dz) != (0.0, 0.0)]
        for arm, ch in self.chains.items():
            p = self.hand_target(arm)
            best = None
            for R in (R_PINCH, R_PINCH_FLIPPED):
                for s in self._seeds(arm, p, R, k):
                    q = ch.ik(p, R, s, iters=400)
                    if q is None:
                        continue
                    worst, ok = 0.0, True
                    for dy, dz in grid:
                        qc = ch.ik(self.hand_target(arm, dy, dz), R, q, tol_r=self.cfg.rot_tol_rad, rest=q)
                        if qc is None:
                            ok = False
                            break
                        worst = max(worst, float(np.abs(qc - q).max()))
                    if not ok:
                        continue
                    score = (worst, -ch.limit_margin(q))
                    if best is None or score < best[0]:
                        best = (score, q, R)
            if best is None:
                raise RuntimeError(f"{arm} arm: no catching pose at {p.round(3).tolist()} with +/-{m} m corrections")
            self.nominal[arm], self.rot[arm] = best[1], best[2]
        return dict(self.nominal)

    def correction(self, dy: float, dz: float, q_now: dict[str, np.ndarray] | None = None):
        """Joint targets for both arms shifted by (dy, dz) (clamped); None if an arm cannot reach it."""
        if not self.nominal:
            raise RuntimeError("solve_nominal() first")
        m = self.cfg.max_correction_m
        dy, dz = float(np.clip(dy, -m, m)), float(np.clip(dz, -m, m))
        out = {}
        for arm, ch in self.chains.items():
            seed = (q_now or {}).get(arm, self.nominal[arm])
            target = self.hand_target(arm, dy, dz)
            # stay close to the catching pose: least joint motion, same arm posture throw after throw
            q = ch.ik(target, self.rot[arm], seed, tol_r=self.cfg.rot_tol_rad, rest=self.nominal[arm])
            if q is None:
                q = ch.ik(target, self.rot[arm], self.nominal[arm], tol_r=self.cfg.rot_tol_rad, rest=self.nominal[arm])
            if q is None:
                return None
            out[arm] = q
        return out, (dy, dz)
