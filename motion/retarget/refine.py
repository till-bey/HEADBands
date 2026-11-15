#!/usr/bin/env python3
"""Improved copy of a retarget.py output; the original file is left as it is.

- The throw ends at CUT_T (s in SRC), before the elbow twist (ra3) swings back.
- The disc is held flat: ra5 (forearm roll) is set at every row so the gripper's opening axis (its y) is vertical,
  on the FLAT_SIDE branch (+1: gripper +y up, -1: down); ra6 is 0. The recording alone held it nearly on edge.
- The wrist flick runs on ra7, whose axis is the gripper's y, so with the disc flat it turns the hand about vertical:
  FLICK_DEG[0] (fingers swung back) to FLICK_DEG[1] over the last FLICK_PART, speed rising from zero to the release.
- After the release every joint keeps its release speed, easing to zero over SLOW_DOWN_S with a cosine profile:
  no jump in acceleration at the release or at the stop.
- A last column joint_rf: 1 (closed), 0 (open) from OPEN_LEAD_S before the release on (a row is inserted there);
  cartesian_follow sends it at full gripper speed.

  python3 refine.py   # joints_trajectory_shoulder_rel_3.csv -> joints_trajectory_shoulder_rel_3_v2.csv
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src", "panther_examples"))
from panther_examples.cartesian_follow import Chain, spline_peak, timed  # noqa: E402

SRC = os.path.join(HERE, "joints_trajectory_shoulder_rel_3.csv")
OUT = os.path.join(HERE, "joints_trajectory_shoulder_rel_3_v2.csv")
CUT_T = 3.97               # s in SRC: release here; ra3 turns back right after
FLAT_SIDE = -1             # which gripper side faces up when the disc is flat (both branches fit ra5's limits)
FLICK_DEG = (20.0, -20.0)  # hand angle about vertical at the start and at the release (+ = fingers swung back)
FLICK_PART = 0.55          # share of the throw, at its end, during which the flick happens
SLOW_DOWN_S = 0.8          # s after release (file time) for every joint to ease to a stop
OPEN_LEAD_S = 0.05         # s (file time) the gripper opens before the release, for its lag

chain = Chain(open(os.path.join(HERE, "..", "..", "src", "panther_sim", "urdf", "robot_description.urdf")).read(),
              "body_link", "link_rf")
d = np.loadtxt(SRC, delimiter=",", comments="#")
d = d[d[:, 0] <= CUT_T + 1e-9]
t, q = d[:, 0], d[:, 1:]

# Flat disc: ra5 maximising FLAT_SIDE * (gripper y)_z, with ra6 = ra7 = 0 (ra7 turns about gripper y: no effect).
q[:, 6] = q[:, 7] = 0.0
grid = np.radians(np.arange(-160.0, 160.5, 0.5))
for row in q:
    tilt = []
    for a in grid:
        row[5] = a
        tilt.append(FLAT_SIDE * chain.fk(row).M.UnitY()[2])
    row[5] = grid[int(np.argmax(tilt))]
    assert max(tilt) > 0.9, f"disc can't be levelled within 25 deg ({np.degrees(np.arccos(max(tilt))):.0f} deg)"
assert np.abs(np.diff(q[:, 5])).max() < np.radians(30), "ra5 jumps: the flat branch flipped"

# Flick sign: increasing ra7 must swing the fingers (gripper -z) backward, against the hand's motion at release.
fingers = lambda qq: -np.array(list(chain.fk(qq).M.UnitZ()))
hand_v = np.array(list(chain.fk(q[-1]).p - chain.fk(q[-2]).p))
dq = np.zeros(8)
dq[7] = 1e-3
sign = -1.0 if (fingers(q[-1] + dq) - fingers(q[-1])) @ hand_v > 0 else 1.0
s = np.clip((t / t[-1] - (1 - FLICK_PART)) / FLICK_PART, 0.0, 1.0)
q[:, 7] = sign * np.radians(FLICK_DEG[0] + (FLICK_DEG[1] - FLICK_DEG[0]) * s**2)

# Speed v0 * (1 + cos(pi u / T)) / 2 after the release; its integral is the displacement below.
dt = t[-1] - t[-2]
n = int(np.ceil(SLOW_DOWN_S / dt))
u = SLOW_DOWN_S * np.arange(1, n + 1) / n
g = u / 2 + SLOW_DOWN_S / (2 * np.pi) * np.sin(np.pi * u / SLOW_DOWN_S)
q = np.clip(np.vstack([q, q[-1] + g[:, None] * (q[-1] - q[-2]) / dt]), chain.lo, chain.hi)
ts = np.r_[t, t[-1] + u]

# Gripper opens OPEN_LEAD_S before the release, at an inserted row (joints interpolated there).
t_open = t[-1] - OPEN_LEAD_S
i = int(np.searchsorted(ts, t_open))
assert ts[i - 1] < t_open < ts[i], "OPEN_LEAD_S lands on a row: change it"
q = np.insert(q, i, [np.interp(t_open, ts, q[:, k]) for k in range(q.shape[1])], axis=0)
ts = np.insert(ts, i, t_open)
grip = (ts < t_open - 1e-9).astype(float)

peak = spline_peak(*timed(chain, list(q), ts, q[0], np.inf)).max()
vel = np.ceil(peak * 100) / 100
np.savetxt(OUT, np.column_stack([ts, q, grip]), delimiter=",", fmt="%.6f", header=(
    f"t,{','.join(chain.names)},joint_rf  (s, rad, 1 closed / 0 open) for cartesian_follow.py --arm right "
    f"--joints FILE --max-joint-vel {vel}\n"
    f"{os.path.basename(SRC)} cut at t={CUT_T}s (release), gripper opens {OPEN_LEAD_S}s before, "
    f"then {SLOW_DOWN_S}s cosine slow-down\n"
    f"disc flat via ra5 (side {FLAT_SIDE:+d} up), flick on {chain.names[7]} {FLICK_DEG[0]:+.0f} to "
    f"{FLICK_DEG[1]:+.0f} deg over the last {FLICK_PART:.0%}"))
print(f"{OUT}: release at t={t[-1]:.3f}s, gripper opens t={t_open:.3f}s, ends t={ts[-1]:.3f}s, "
      f"ra5 {np.degrees(q[:, 5].min()):.0f}..{np.degrees(q[:, 5].max()):.0f} deg, "
      f"peak {peak:.2f} rad/s (--max-joint-vel {vel})")
