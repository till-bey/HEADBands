"""Frisbee tracking in image space: a few constant-velocity Kalman tracks on (u, v, d).

State x = [u, v, d, du/dt, dv/dt, dd/dt]  (d = apparent diameter in pixels, the ellipse major axis).
Time steps come from message timestamps, never from an assumed frame rate.

Up to `max_tracks` candidate tracks run at once, so one static distractor cannot block the real frisbee.
Each track goes through:
  TENTATIVE  - started from one detection; needs `min_hits` consistent detections AND visible motion
               (a frisbee in flight moves or grows; a lamp, a wall marking or an isolated false detection
               does not get confirmed);
  CONFIRMED  - publishes velocity and size rate; keeps a short history of measurements;
  COASTING   - no detection for a while: the filter predicts, the history is kept;
  LOST       - no detection for `max_coast_s`: the track is dropped.
The primary track is the confirmed (or coasting) track that moves or grows fastest; it switches only to a
clearly faster one (`primary_switch_ratio`), so walking people or heads do not hide a frisbee in flight.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

TENTATIVE, CONFIRMED, COASTING, LOST = "tentative", "confirmed", "coasting", "lost"


@dataclass
class TrackerConfig:
    min_hits: int = 3  # consistent detections before a track is confirmed
    min_motion_px_s: float = 15.0  # confirmed only if speed or size-rate exceeds this (rejects static objects)
    max_coast_s: float = 0.25  # drop a confirmed track after this long without a detection
    max_tracks: int = 8  # crowded scenes: people moving must not use up every slot
    primary_switch_ratio: float = 1.5  # switch the primary to a track this much faster (px/s)
    gate_base_px: float = 40.0  # association gate radius at zero velocity / uncertainty
    gate_sigma: float = 4.0  # plus this many predicted standard deviations
    max_size_ratio: float = 1.8  # reject a detection whose diameter jumps by more than this factor
    min_confidence: float = 0.3  # detections below this are ignored
    accel_noise_px: float = 1000.0  # process noise: px/s^2 of unmodeled acceleration (u, v)
    size_accel_noise_px: float = 800.0  # px/s^2 for the diameter (it grows fast on approach)
    meas_noise_px: float = 4.0  # center measurement noise, px
    size_meas_noise_frac: float = 0.08  # diameter measurement noise, fraction of the diameter
    history_len: int = 20  # measurements kept for trajectory fitting
    max_dt_s: float = 0.5  # larger gaps restart tracking

    @classmethod
    def from_dict(cls, d: dict) -> TrackerConfig:
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, type(getattr(cfg, k))(v))
        return cfg


@dataclass
class Measurement:
    t: float  # seconds (message stamp)
    u: float
    v: float
    d: float  # apparent diameter, px
    confidence: float = 1.0


class KalmanCV:
    """Three independent constant-velocity axes (u, v, d), written as one 6-state filter."""

    def __init__(self, z: Measurement, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self.x = np.array([z.u, z.v, z.d, 0.0, 0.0, 0.0])
        r = self._R(z.d)
        self.P = np.diag([r[0, 0], r[1, 1], r[2, 2], 800.0**2, 800.0**2, 400.0**2])
        self.t = z.t

    def _R(self, d: float) -> np.ndarray:
        s = max(1.0, self.cfg.size_meas_noise_frac * d)
        m = self.cfg.meas_noise_px
        return np.diag([m * m, m * m, s * s])

    def _FQ(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        Q = np.zeros((6, 6))
        for i, q in ((0, self.cfg.accel_noise_px), (1, self.cfg.accel_noise_px), (2, self.cfg.size_accel_noise_px)):
            q2 = q * q
            Q[i, i] = q2 * dt**4 / 4
            Q[i, i + 3] = Q[i + 3, i] = q2 * dt**3 / 2
            Q[i + 3, i + 3] = q2 * dt**2
        return F, Q

    def predicted(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        dt = max(0.0, t - self.t)
        F, Q = self._FQ(dt)
        return F @ self.x, F @ self.P @ F.T + Q

    def update(self, z: Measurement) -> None:
        x, P = self.predicted(z.t)
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        y = np.array([z.u, z.v, z.d]) - H @ x
        S = H @ P @ H.T + self._R(z.d)
        K = P @ H.T @ np.linalg.inv(S)
        self.x = x + K @ y
        self.P = (np.eye(6) - K @ H) @ P
        self.t = z.t


class Track:
    _next_id = 1

    def __init__(self, z: Measurement, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self.kf = KalmanCV(z, cfg)
        self.state = TENTATIVE
        self.hits = 1
        self.last_meas_t = z.t
        self.history: deque[Measurement] = deque([z], maxlen=cfg.history_len)
        self.id = Track._next_id
        Track._next_id += 1

    def gate(self, z: Measurement, t: float) -> float | None:
        """Normalized distance of a detection to the prediction, or None if outside the gate."""
        x, P = self.kf.predicted(t)
        dist = math.hypot(z.u - x[0], z.v - x[1])
        sigma = math.sqrt(max(P[0, 0], 0.0) + max(P[1, 1], 0.0))
        gate = self.cfg.gate_base_px + self.cfg.gate_sigma * sigma + 0.5 * max(x[2], z.d)
        if dist > gate:
            return None
        if x[2] > 0 and z.d > 0 and max(z.d / x[2], x[2] / z.d) > self.cfg.max_size_ratio:
            return None
        return dist / gate

    def motion_rate(self, n: int = 5) -> float:
        """Image speed or size rate (px/s, whichever is larger) over the last n measurements."""
        if len(self.history) < 2:
            return 0.0
        h = list(self.history)[-n:]
        a, b = h[0], h[-1]
        dt = b.t - a.t
        if dt <= 0:
            return 0.0
        return max(math.hypot(b.u - a.u, b.v - a.v), abs(b.d - a.d)) / dt

    def moving(self) -> bool:
        """Has the track moved or grown enough to be a frisbee in flight (measured, not filtered)?"""
        return self.motion_rate(n=len(self.history)) >= self.cfg.min_motion_px_s

    def update(self, z: Measurement) -> None:
        self.kf.update(z)
        self.history.append(z)
        self.hits += 1
        self.last_meas_t = z.t
        if self.state == TENTATIVE and self.hits >= self.cfg.min_hits and self.moving():
            self.state = CONFIRMED
        elif self.state == COASTING:
            self.state = CONFIRMED

    def miss(self, t: float) -> None:
        if self.state == TENTATIVE:
            self.state = LOST  # a tentative track that misses is most likely a false detection
        elif t - self.last_meas_t > self.cfg.max_coast_s:
            self.state = LOST
        else:
            self.state = COASTING


class FrisbeeTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.cfg = config or TrackerConfig()
        self.tracks: list[Track] = []
        self.rejected = 0  # detections not used by any track (diagnostics)
        self._primary: Track | None = None

    def reset(self) -> None:
        self.tracks = []
        self._primary = None

    # Single-track view used by the nodes and tests --------------------------
    @property
    def primary(self) -> Track | None:
        return self._primary

    @property
    def state(self) -> str:
        p = self._primary
        if p is not None:
            return p.state
        return TENTATIVE if self.tracks else LOST

    @property
    def history(self) -> deque:
        return self._primary.history if self._primary is not None else deque()

    @property
    def hits(self) -> int:
        return self._primary.hits if self._primary is not None else 0

    @property
    def last_meas_t(self) -> float | None:
        return self._primary.last_meas_t if self._primary is not None else None

    def step(self, t: float, detections: list[Measurement]) -> str:
        """Advance to time t with this frame's detections (possibly empty). Returns the primary state."""
        dets = [d for d in detections if d.confidence >= self.cfg.min_confidence and d.d > 0]
        self.tracks = [tr for tr in self.tracks if t - tr.last_meas_t <= self.cfg.max_dt_s]

        # Greedy association: best (track, detection) pairs first; confirmed tracks get priority.
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di, d in enumerate(dets):
                g = tr.gate(d, t)
                if g is not None:
                    prio = 0.0 if tr.state in (CONFIRMED, COASTING) else 1.0
                    pairs.append((prio, g - 0.2 * d.confidence, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            self.tracks[ti].update(dets[di])
            used_t.add(ti)
            used_d.add(di)
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.miss(t)
        self.tracks = [tr for tr in self.tracks if tr.state != LOST]

        # Unused detections seed new tentative tracks (most confident first).
        free = sorted((d for i, d in enumerate(dets) if i not in used_d), key=lambda d: -d.confidence)
        self.rejected += len(free)
        for d in free:
            if len(self.tracks) >= self.cfg.max_tracks:
                break
            self.tracks.append(Track(d, self.cfg))

        # Primary = the confirmed track that moves or grows fastest (a frisbee in flight is much faster than
        # walking people or heads). Switch only to a clearly faster track, so the primary does not flap.
        confirmed = [tr for tr in self.tracks if tr.state in (CONFIRMED, COASTING)]
        if not confirmed:
            self._primary = None
        else:
            fastest = max(confirmed, key=lambda tr: tr.motion_rate())
            if self._primary not in confirmed or (
                fastest is not self._primary
                and fastest.motion_rate() > self.cfg.primary_switch_ratio * self._primary.motion_rate()
            ):
                self._primary = fastest
        return self.state

    # ----------------------------------------------------------------- outputs
    def estimate(self, t: float | None = None) -> dict | None:
        """Primary track's filtered state (predicted to t if given): position, velocity, size, size rate."""
        tr = self._primary
        if tr is None:
            return None
        x, P = tr.kf.predicted(t) if t is not None else (tr.kf.x, tr.kf.P)
        return {
            "u": float(x[0]), "v": float(x[1]), "d": float(x[2]),
            "du": float(x[3]), "dv": float(x[4]), "dd": float(x[5]),
            "sigma_uv": float(math.sqrt(max(P[0, 0] + P[1, 1], 0.0))),
            "sigma_d": float(math.sqrt(max(P[2, 2], 0.0))),
            "state": tr.state, "hits": tr.hits, "track_id": tr.id,
        }
