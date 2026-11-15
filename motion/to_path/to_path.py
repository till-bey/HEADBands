#!/usr/bin/env python3
"""Turn a phone recording (Trajectory.csv of a frisbee throw) into a path for cartesian_follow.py.

The phone's position is replayed relative to the first sample, shrunk by SCALE and slowed by SLOWDOWN, with the hand
starting at START. The phone's rotation is ignored: the gripper keeps one orientation, fingers pointing along body +y
and opening along z, so the gripper's plane is the body's y-z plane (HANDS: its x axis forward or backward).
On top of that, a wrist flick turns the hand about body z: held at FLICK_DEG[0] (fingers swung back), then turned
to FLICK_DEG[1] over the last FLICK_PART of the throw, its speed ramping up linearly from zero to the release.
After the release, hand motion and flick decelerate to a stop over SLOW_DOWN_S, travelling a little further.
Arm configurations reaching the first pose are tried, and among those from which the whole path stays MIN_MARGIN
inside every joint limit, the one needing the lowest peak joint speed (at most VEL_CAP) is kept. Its joint angles go
into the file as "# start_joints:" so cartesian_follow.py uses that same arm configuration, and its peak speed as the
--max-joint-vel to run it with.
Phone world axes (z up, x = throw direction) are taken as body_link axes.

  python3 to_path.py                        # this folder's recording -> path.csv here
  python3 to_path.py RECORDING_DIR OUT.csv  # any recording folder
"""
import os
import sys

import numpy as np
import PyKDL as kdl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src", "panther_examples", "panther_examples"))
from cartesian_follow import Chain, plan, spline_peak, timed  # noqa: E402

ARM = "right"
# First hand position (m) in base_footprint: floor level, robot centre, x forward. Converted to body_link with
# the lift at its URDF zero; on the robot a raised lift raises the whole path with it.
START = (0.35, 0.0, 1.00)
SCALE = 0.1        # the throw is metres across; the arm reaches ~0.6 m
SLOWDOWN = 1.4     # recordings last 0.6-0.8 s, so throws take ~1 s
FLICK_DEG = (20.0, -20.0)  # hand angle about body z at the start and at the end (+ = fingers swung backward)
FLICK_PART = 0.55          # share of the throw, at its end, during which the flick happens (0.4 needs >3 rad/s)
SLOW_DOWN_S = 0.2          # after release, hand motion and flick keep going, decelerating linearly to a stop
VEL_CAP = 3.0     # rad/s: no path needing a faster joint is written. No vendor limit is known; PR #521 uses 0.35
MIN_MARGIN = 0.02  # rad every joint keeps from its limits along the whole path
CANDIDATES = 2000  # IK seeds tried for the start configuration
URDF = os.path.join(HERE, "..", "..", "src", "panther_sim", "urdf", "panther.urdf")  # the robot's kinematics, from PR #521

src = sys.argv[1] if len(sys.argv) > 1 else HERE
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "path.csv")
pos = np.genfromtxt(os.path.join(src, "Trajectory.csv"), delimiter=",", names=True)

ts = SLOWDOWN * (pos["seconds_elapsed"] - pos["seconds_elapsed"][0])
p = np.column_stack([pos["x"], pos["y"], pos["z"]])
# Gripper axes in body_link (columns x, y, z). The fingers extend along the gripper's -z and open along its y, so
# z = -Y points them along +Y, and x = +-X puts the gripper's y-z plane in the body's Y-Z plane.
HANDS = [kdl.Rotation(kdl.Vector(s, 0, 0), kdl.Vector(0, 0, s), kdl.Vector(0, -1, 0)) for s in (1, -1)]
dp = SCALE * (p - p[0])
# Flick angle per waypoint: constant, then s^2 over the last FLICK_PART (speed rising linearly to the release).
s = np.clip(((ts - ts[0]) / (ts[-1] - ts[0]) - (1 - FLICK_PART)) / FLICK_PART, 0.0, 1.0)
flick = np.radians(FLICK_DEG[0] + (FLICK_DEG[1] - FLICK_DEG[0]) * s**2)
# Slow-down after release: carry on at the release velocity, decaying linearly to zero over SLOW_DOWN_S, so the
# arm is not stopped dead at full speed. Displacement after time u is v * (u - u^2 / (2 * SLOW_DOWN_S)).
dt = ts[-1] - ts[-2]
u = SLOW_DOWN_S * np.arange(1, int(np.ceil(SLOW_DOWN_S / dt)) + 1) / np.ceil(SLOW_DOWN_S / dt)
g = u - u**2 / (2 * SLOW_DOWN_S)
release = len(ts) - 1
dp = np.vstack([dp, dp[-1] + g[:, None] * (dp[-1] - dp[-2]) / dt])
flick = np.r_[flick, flick[-1] + g * (flick[-1] - flick[-2]) / dt]
ts = np.r_[ts, ts[-1] + u]

urdf = open(URDF).read()
chain = Chain(urdf, "body_link", f"link_{ARM[0]}f")
body = Chain(urdf, "base_footprint", "body_link").fk([])  # fixed joints only
target = body.Inverse() * kdl.Vector(*START)


def poses(start):
    # RotZ on the left: about body z, relative to the start pose (which already has flick[0]).
    return [kdl.Frame(kdl.Rotation.RotZ(flick[i] - flick[0]) * start.M, start.p + kdl.Vector(*dp[i]))
            for i in range(len(ts))]


def follow(q0):
    """(joint path, peak joint speed) from start joints q0, or None if out of limits or over VEL_CAP."""
    qs = [q0]
    for f in poses(chain.fk(q0))[1:]:
        q = chain.ik(f, qs[-1])
        if q is None:
            return None
        qs.append(q)
    try:
        planned = timed(chain, qs, ts, q0, VEL_CAP)
    except RuntimeError:
        return None
    return np.array(qs), spline_peak(*planned).max()


rng = np.random.default_rng(0)
best = None
for i in range(CANDIDATES):
    seed = rng.uniform(chain.lo, chain.hi)
    hand = kdl.Rotation.RotZ(flick[0]) * HANDS[i % 2]
    q0 = chain.ik(kdl.Frame(hand, target), seed)  # random seeds find the arm's different configurations
    if q0 is None or (res := follow(q0)) is None:
        continue
    qs, peak = res
    margin = np.minimum(qs - chain.lo, chain.hi - qs).min()
    if margin >= MIN_MARGIN and (best is None or peak < best[0]):
        best = (peak, margin, q0)
if best is None:
    sys.exit(f"{src}: no arm configuration at START {START} works under {VEL_CAP} rad/s: "
             "move START, raise SLOWDOWN or VEL_CAP, lower SCALE or MIN_MARGIN")

q0 = best[2]
start = chain.fk(q0)
rows = [[t, *f.p, *f.M.GetRPY()] for t, f in zip(ts, poses(start))]
# Speed and margin as cartesian_follow.py will see them: it re-solves IK from the rounded CSV values, and near a
# joint limit that can take a slightly different joint path. Plan exactly that (start_joints as the hint).
rounded = np.round(rows, 5)
try:
    qs_r, t_r, v_r = plan(chain, [kdl.Frame(kdl.Rotation.RPY(*r[4:7]), kdl.Vector(*r[1:4])) for r in rounded],
                          rounded[:, 0], np.round(q0, 6), VEL_CAP, np.round(q0, 6))
except RuntimeError as e:
    sys.exit(f"{src}: the written path fails to plan: {e}")
peak = spline_peak(qs_r, t_r, v_r).max()
margin = np.minimum(qs_r - chain.lo, chain.hi - qs_r).min()
vel = np.ceil(peak * 100) / 100  # the --max-joint-vel this path needs, rounded up
np.savetxt(out, rows, delimiter=",", fmt="%.5f", header=(
    f"t,x,y,z,roll,pitch,yaw  (s, m, rad) of link_{ARM[0]}f in body_link, for --arm {ARM} --max-joint-vel {vel}: "
    f"{os.path.basename(os.path.normpath(src))}, x{SCALE} in size, x{SLOWDOWN} slower, "
    f"starting at {START} m in base_footprint, joint margin {margin:.3f} rad\n"
    f"flick {FLICK_DEG[0]:+.0f} to {FLICK_DEG[1]:+.0f} deg about body z over the last {FLICK_PART:.0%}, release at "
    f"t={ts[release]:.3f}s, then {SLOW_DOWN_S}s slow-down to {np.degrees(flick[-1]):+.0f} deg\n"
    "start_joints: " + " ".join(f"{v:.6f}" for v in q0)))
after = 100 * np.linalg.norm(dp[-1] - dp[release])
print(f"{out}: peak joint speed {peak:.2f} rad/s (run with --max-joint-vel {vel}), margin {margin:.3f} rad, "
      f"after release: {after:.1f} cm further, flick ends at {np.degrees(flick[-1]):+.0f} deg")
