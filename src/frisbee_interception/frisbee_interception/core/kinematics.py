"""Arm kinematics from the URDF (numpy only): forward kinematics and damped least-squares IK with joint limits.

Same model as panther_examples/cartesian_follow.py (PyKDL Chain): a chain of URDF joints from `base` to `tip`,
each with its fixed origin (xyz, rpy) followed by a rotation about its axis. CI checks both agree.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np


def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_angle(axis: np.ndarray, a: float) -> np.ndarray:
    x, y, z = axis / np.linalg.norm(axis)
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def rot_error(R_goal: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotation vector taking R to R_goal (base frame)."""
    E = R_goal @ R.T
    c = max(-1.0, min(1.0, (np.trace(E) - 1) / 2))
    a = math.acos(c)
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    if math.sin(a) < 1e-6:  # ~180 deg
        w, V = np.linalg.eigh((E + E.T) / 2)
        return V[:, int(np.argmax(w))] * a
    return v * a / (2 * math.sin(a))


class ArmChain:
    def __init__(self, urdf: str, base: str, tip: str) -> None:
        joints_by_child = {j.find("child").get("link"): j for j in ET.fromstring(urdf).findall("joint")}
        path, link = [], tip
        while link != base:
            if link not in joints_by_child:
                raise ValueError(f"no joint chain from {base} to {tip}")
            path.append(joints_by_child[link])
            link = path[-1].find("parent").get("link")
        self.segments = []  # (origin 4x4, axis or None)
        self.names, lo, hi = [], [], []
        for j in reversed(path):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = rpy_matrix(*rpy), xyz
            if j.get("type") == "revolute":
                axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
                self.segments.append((T, axis))
                self.names.append(j.get("name"))
                lo.append(float(j.find("limit").get("lower")))
                hi.append(float(j.find("limit").get("upper")))
            elif j.get("type") == "fixed":
                self.segments.append((T, None))
            else:
                raise ValueError(f"{j.get('name')}: {j.get('type')} joints are not handled")
        self.lo, self.hi = np.array(lo), np.array(hi)

    @property
    def n(self) -> int:
        return len(self.names)

    def fk(self, q) -> np.ndarray:
        return self._fk(q)[0]

    def _fk(self, q):
        T = np.eye(4)
        axes, origins = [], []
        k = 0
        for origin, axis in self.segments:
            T = T @ origin
            if axis is not None:
                R = T[:3, :3]
                axes.append(R @ (axis / np.linalg.norm(axis)))
                origins.append(T[:3, 3].copy())
                Tj = np.eye(4)
                Tj[:3, :3] = axis_angle(axis, float(q[k]))
                T = T @ Tj
                k += 1
        return T, axes, origins

    def jacobian(self, q) -> tuple[np.ndarray, np.ndarray]:
        T, axes, origins = self._fk(q)
        p = T[:3, 3]
        J = np.zeros((6, self.n))
        for i, (a, o) in enumerate(zip(axes, origins)):
            J[:3, i] = np.cross(a, p - o)
            J[3:, i] = a
        return T, J

    def ik(self, p_goal, R_goal, seed, iters: int = 300, tol_p: float = 1e-4, tol_r: float = 1e-3,
           rot_weight: float = 0.3, damping: float = 0.02, rest=None):
        """Joint angles reaching (p_goal, R_goal) within the limits, or None. Seeded: stays near `seed`.

        The arm is redundant (8 joints, 6 constraints): the free motion is pulled toward `rest` (default: the middle
        of the joint ranges, clear of limits). Corrections pass the catching pose, so they move the joints least.
        """
        q = np.clip(np.array(seed, float), self.lo, self.hi)
        for _ in range(iters):
            T, J = self.jacobian(q)
            ep = np.asarray(p_goal) - T[:3, 3]
            er = rot_error(np.asarray(R_goal), T[:3, :3])
            if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
                return q
            e = np.r_[ep, rot_weight * er]
            Jw = J.copy()
            Jw[3:] *= rot_weight
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + damping ** 2 * np.eye(6), e)
            # null-space pull toward the rest posture (does not change the hand pose)
            N = np.eye(self.n) - np.linalg.pinv(Jw) @ Jw
            if rest is None:
                dq += 0.05 * N @ ((self.lo + self.hi) / 2 - q) / np.maximum(self.hi - self.lo, 1e-6)
            else:
                dq += 0.3 * N @ (np.asarray(rest) - q)
            step = np.max(np.abs(dq))
            if step > 0.2:
                dq *= 0.2 / step
            q = np.clip(q + dq, self.lo, self.hi)
        return None

    def limit_margin(self, q) -> float:
        """Smallest distance (rad) of any joint to its limits."""
        return float(np.min(np.minimum(q - self.lo, self.hi - q)))
