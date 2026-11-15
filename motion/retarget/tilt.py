#!/usr/bin/env python3
"""Rotate a whole joints file about a body_link axis through the shoulder (x = z = 0 for y, x = y = 0 for z);
the input file is left as it is. Optionally re-time the gripper: fully open LEAD s before the release, after
opening linearly over OPEN_S (times in the file; F=0.5 playback halves them).

Every row's hand frame (link_rf, position and orientation) is rotated by DEG and IK is re-solved, seeded from
the input row shifted by the previous row's change, so the arm keeps nearly the same shape. The gripper column is kept.

  python3 tilt.py joints_trajectory_shoulder_rel_3_v2.csv joints_trajectory_shoulder_rel_3_v3.csv y -15
  python3 tilt.py joints_trajectory_shoulder_rel_3_v3.csv joints_trajectory_shoulder_rel_3_v5.csv z -15 0.2
"""
import os
import sys

import numpy as np
import PyKDL as kdl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src", "panther_examples"))
from panther_examples.cartesian_follow import Chain, spline_peak, timed  # noqa: E402

if len(sys.argv) not in (5, 6) or sys.argv[3] not in ("x", "y", "z"):
    sys.exit(__doc__)
src, out = (os.path.join(HERE, a) for a in sys.argv[1:3])
axis, deg = sys.argv[3], float(sys.argv[4])
lead = float(sys.argv[5]) if len(sys.argv) == 6 else None
OPEN_S = 0.2  # s the gripper takes from closed to open (file time); its speed is 1 / OPEN_S strokes/s

chain = Chain(open(os.path.join(HERE, "..", "..", "src", "panther_sim", "urdf", "robot_description.urdf")).read(),
              "body_link", "link_rf")
d = np.loadtxt(src, delimiter=",", comments="#")
header = [line[2:].rstrip("\n") for line in open(src) if line.startswith("# ")]
ts, q, grip = d[:, 0], d[:, 1:9], d[:, 9]
opened = int(np.flatnonzero(grip == 0)[0])  # the row refine.py inserted where the gripper opens
release = opened + 1

rot = kdl.Frame({"x": kdl.Rotation.RotX, "y": kdl.Rotation.RotY, "z": kdl.Rotation.RotZ}[axis](np.radians(deg)))
qt, prev = [], None
for i, r in enumerate(q):
    target = rot * chain.fk(r)
    sol = chain.ik(target, r if prev is None else r + (prev - q[i - 1]))
    if sol is None:
        sol = chain.ik(target, r)
    if sol is None:
        sys.exit(f"{deg:+.0f} deg about {axis}: row {i} (t={ts[i]:.2f}s) out of reach within the joint limits")
    qt.append(sol)
    prev = sol
qt = np.array(qt)
jump = np.degrees(np.abs(np.diff(qt, axis=0)).max(axis=0))
assert jump.max() < 30, f"IK switched arm configuration (step up to {jump.max():.0f} deg on {chain.names[int(jump.argmax())]})"
change = np.degrees(np.abs(qt - q).max())  # vs the input, before any gripper re-timing

note = f"{os.path.basename(src)} rotated {deg:+.0f} deg about body {axis} (hand path rotated, IK re-solved)"
if lead is not None:  # drop the opening row, add rows where the opening starts and ends (joints interpolated)
    ts, qt, grip = np.delete(ts, opened), np.delete(qt, opened, axis=0), np.delete(grip, opened)
    release -= 1
    t_done = ts[release] - lead
    for t_new in (t_done - OPEN_S, t_done):
        i = int(np.searchsorted(ts, t_new))
        assert 0 < i and ts[i - 1] < t_new - 1e-4 and t_new + 1e-4 < ts[i], f"t={t_new:.3f}s lands on a row: change LEAD"
        qt = np.insert(qt, i, [np.interp(t_new, ts, qt[:, k]) for k in range(8)], axis=0)
        ts = np.insert(ts, i, t_new)
        release += 1
    grip = np.interp(ts, [t_done - OPEN_S, t_done], [1.0, 0.0])
    note += f", gripper opens over {OPEN_S}s and is fully open {lead}s before the release"

peak = spline_peak(*timed(chain, list(qt), ts, qt[0], np.inf)).max()
vel = np.ceil(peak * 100) / 100
header[0] = header[0].split("--max-joint-vel")[0] + f"--max-joint-vel {vel}"
np.savetxt(out, np.column_stack([ts, qt, grip]), delimiter=",", fmt="%.6f", header="\n".join(header + [note]))
z = lambda qq, k: chain.fk(qq[k]).p[2]
print(f"{out}: {note}; release {100 * (z(qt, release) - z(qt, 0)):.1f} cm above the start, largest joint change "
      f"vs input {change:.0f} deg, "
      f"peak {peak:.2f} rad/s (--max-joint-vel {vel})")
