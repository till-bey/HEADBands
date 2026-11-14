#!/usr/bin/env python3
"""Turn a phone recording (Trajectory.csv + Orientation.csv, a frisbee throw) into path.csv for cartesian_follow.py.

Position and orientation are replayed relative to the first sample, shrunk by SCALE and slowed by SLOWDOWN, with the
hand starting at START. The start orientation is free: hand orientations at START are tried, and the one from which
the whole path stays furthest inside the joint limits (at least MIN_MARGIN) and under MAX_JOINT_VEL is kept. Its
joint angles go into path.csv as "# start_joints:" so cartesian_follow.py uses that same arm configuration.
Phone world axes (z up, x = throw direction) are taken as body_link axes.

  python3 to_path.py          # writes path.csv next to this script
  python3 ../cartesian_follow.py --arm right --path path.csv --dry-run
"""
import os
import sys

import numpy as np
import PyKDL as kdl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from cartesian_follow import MAX_JOINT_VEL, Chain, timed  # noqa: E402

ARM = "right"
# First hand position (m) in base_footprint: floor level, robot centre, x forward. Converted to body_link with
# the lift at its URDF zero; on the robot a raised lift raises the whole path with it.
START = (0.35, 0.0, 1.00)
SCALE = 0.1        # the throw is metres across; the arm reaches ~0.6 m
SLOWDOWN = 10.0    # the phone turns at up to 15.8 rad/s; each joint is limited to MAX_JOINT_VEL
MIN_MARGIN = 0.02  # rad every joint keeps from its limits along the whole path
CANDIDATES = 2000  # start orientations tried
URDF = os.path.join(HERE, "..", "..", "mock-sim", "panther.urdf")  # the robot's kinematics, from PR #521

pos = np.genfromtxt(os.path.join(HERE, "Trajectory.csv"), delimiter=",", names=True)
ori = np.genfromtxt(os.path.join(HERE, "Orientation.csv"), delimiter=",", names=True)
assert np.array_equal(pos["seconds_elapsed"], ori["seconds_elapsed"]), "the two files are not sampled together"

ts = SLOWDOWN * (pos["seconds_elapsed"] - pos["seconds_elapsed"][0])
R = [kdl.Rotation.Quaternion(o["qx"], o["qy"], o["qz"], o["qw"]) for o in ori]
dR = [r * R[0].Inverse() for r in R]  # phone rotation since start, in the phone's world frame
p = np.column_stack([pos["x"], pos["y"], pos["z"]])
dp = SCALE * (p - p[0])

urdf = open(URDF).read()
chain = Chain(urdf, "body_link", f"link_{ARM[0]}f")
body = Chain(urdf, "base_footprint", "body_link").fk([])  # fixed joints only
target = body.Inverse() * kdl.Vector(*START)


def poses(start):
    return [kdl.Frame(dR[i] * start.M, start.p + kdl.Vector(*dp[i])) for i in range(len(ts))]


def follow(q0):
    """Joint path from start joints q0, or None if a waypoint or a joint speed is out of limits."""
    qs = [q0]
    for f in poses(chain.fk(q0))[1:]:
        q = chain.ik(f, qs[-1])
        if q is None:
            return None
        qs.append(q)
    try:
        timed(chain, qs, ts, q0, MAX_JOINT_VEL)
    except RuntimeError:
        return None
    return np.array(qs)


rng = np.random.default_rng(0)
best = None
for _ in range(CANDIDATES):
    seed = rng.uniform(chain.lo, chain.hi)
    q0 = chain.ik(kdl.Frame(chain.fk(seed).M, target), seed)  # START with the hand orientation of a random arm pose
    if q0 is None:
        continue
    qs = follow(q0)
    if qs is None:
        continue
    margin = np.minimum(qs - chain.lo, chain.hi - qs).min()
    if margin >= MIN_MARGIN and (best is None or margin > best[0]):
        best = (margin, q0)
if best is None:
    sys.exit(f"no orientation at START {START} works: move START, raise SLOWDOWN, lower SCALE or MIN_MARGIN")

margin, q0 = best
start = chain.fk(q0)
rows = [[t, *f.p, *f.M.GetRPY()] for t, f in zip(ts, poses(start))]
np.savetxt(os.path.join(HERE, "path.csv"), rows, delimiter=",", fmt="%.5f", header=(
    f"t,x,y,z,roll,pitch,yaw  (s, m, rad) of link_{ARM[0]}f in body_link, for --arm {ARM}: frisbee throw from "
    f"the phone, x{SCALE} in size, x{SLOWDOWN} slower, starting at {START} m in base_footprint\n"
    "start_joints: " + " ".join(f"{v:.6f}" for v in q0)))
print(f"start: hand {np.round(list(start.p), 3)} in body_link = {START} in base_footprint, "
      f"joint margin {margin:.3f} rad; wrote path.csv")
