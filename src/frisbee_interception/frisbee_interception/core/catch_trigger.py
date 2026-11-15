"""When to close the grippers for a catch at the chest (pure Python, no ROS).

The robot already holds a catching pose (taught with panther_examples `teach`). The thrower aims at its chest.
The only decision left is *when* to close the grippers: at the predicted arrival time minus the gripper's own
closing latency, and only for throws that will cross the calibrated catch window (the hands, in image pixels).

Each valid prediction updates the planned closing time. The plan is cancelled if the newest valid prediction
leaves the window or loses confidence. Once fired, the trigger waits `refractory_s` before it can fire again.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CatchConfig:
    gripper_latency_s: float = -1.0  # measured time from command to closed grip; < 0 = not measured -> never arm
    catch_window_px: list = field(default_factory=lambda: [0.0])  # [u0, v0, u1, v1] where catchable throws cross
    min_confidence: float = 0.5
    min_lead_s: float = 0.0  # do not fire later than this before arrival (0: fire even at the last moment)
    max_late_s: float = 0.03  # a closing time this far in the past is still sent (frames arrive late)
    refractory_s: float = 2.0  # one catch per throw

    @classmethod
    def from_dict(cls, d: dict) -> CatchConfig:
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                cur = getattr(cfg, k)
                setattr(cfg, k, [float(x) for x in v] if isinstance(cur, list) else type(cur)(v))
        return cfg

    @property
    def window(self):
        w = [float(x) for x in self.catch_window_px]
        return w if len(w) == 4 and w[2] > w[0] and w[3] > w[1] else None

    def calibration_problems(self) -> list[str]:
        out = []
        if self.gripper_latency_s < 0:
            out.append("gripper_latency_s not measured")
        if self.window is None:
            out.append("catch_window_px not set")
        return out


@dataclass
class CatchPlan:
    close_at: float  # absolute time (measurement clock) to send the close command
    intercept_time: float
    pixel: tuple
    confidence: float


class CatchScheduler:
    def __init__(self, cfg: CatchConfig) -> None:
        self.cfg = cfg
        self.plan: CatchPlan | None = None
        self.last_fire: float | None = None
        self.last_reason = "idle"

    def in_window(self, pixel) -> bool:
        w = self.cfg.window
        return w is not None and pixel is not None and w[0] <= pixel[0] <= w[2] and w[1] <= pixel[1] <= w[3]

    def update(self, t_now: float, valid: bool, intercept_time: float, pixel, confidence: float) -> CatchPlan | None:
        """Feed one prediction. Returns the current plan (or None)."""
        if self.last_fire is not None and t_now - self.last_fire < self.cfg.refractory_s:
            self.last_reason = "refractory after a catch"
            return None
        if not valid:
            # keep the plan: the disc is usually lost in the last frames (too close, blurred)
            self.last_reason = "keeping plan (no new valid prediction)" if self.plan else "no valid prediction"
            return self.plan
        if confidence < self.cfg.min_confidence:
            self.plan, self.last_reason = None, f"confidence {confidence:.2f} too low"
            return None
        if not self.in_window(pixel):
            self.plan, self.last_reason = None, "throw will miss the catch window"
            return None
        close_at = intercept_time - max(self.cfg.gripper_latency_s, 0.0)
        if intercept_time - t_now < self.cfg.min_lead_s:
            self.last_reason = "too late for a new plan"
            return self.plan
        self.plan = CatchPlan(close_at, intercept_time, (float(pixel[0]), float(pixel[1])), confidence)
        self.last_reason = "planned"
        return self.plan

    def due(self, t_now: float) -> CatchPlan | None:
        """Return the plan if it must fire now (and forget it), else None."""
        p = self.plan
        if p is None or t_now < p.close_at:
            return None
        self.plan = None
        if t_now - p.close_at > self.cfg.max_late_s and t_now > p.intercept_time:
            self.last_reason = "missed: closing time passed"
            return None
        self.last_fire = t_now
        self.last_reason = "fired"
        return p
