#!/usr/bin/env python3
"""Retarget the cropped human arm recordings (selections.csv, made with visualize.py) to right-arm joint angles.

Joint angles are copied, not hand positions, so the robot's arm takes the human's elbow pose whatever the limb lengths:
- ra1, ra2, ra3 (spherical shoulder) and ra4 (elbow): solved in closed form from the upper-arm and forearm directions.
  ra2 only swings the arm outward, so when the arm crosses in front of the body ra0 (yaw about z) turns it inward.
  With the elbow nearly straight the upper-arm twist (ra3) is undefined and the neighbouring value is held.
- ra5 = 0. Wrist flick like example_1/to_path.py: FLICK_DEG[0] (fingers back), then to FLICK_DEG[1] over the last
  FLICK_PART of the crop, its speed rising from zero to the release. It runs on ra6 or ra7, whichever axis is closer
  to body z at the release (example_1 turns the hand about body z), signed so the fingers swing in the hand's direction.
After the release, every joint keeps its release speed, decaying linearly to zero over SLOW_DOWN_S.
Angles outside the joint limits are clipped (reported). Time is stretched so no joint exceeds VEL_CAP.
Output: joints_<recording>.csv here, rows t,q_ra0..q_ra7, for cartesian_follow.py --arm right --joints FILE.

  python3 retarget.py
"""
import csv
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from cartesian_follow import Chain, spline_peak, timed  # noqa: E402

URDF = os.path.join(HERE, "..", "..", "mock-sim", "panther.urdf")
# Recording axes -> body_link axes (x forward, y left, z up). The recording is believed to be the same, but not
# certain: if the arm moves rotated or mirrored compared to visualize.py, fix this matrix.
TO_BODY = np.eye(3)
SMOOTH = 7                 # samples (60 Hz) of the Savitzky-Golay window on the positions; noise inflates joint speed
YAW_SMOOTH = 15            # samples of the moving average on ra0 (see arm_angles); wider = slower ra0, more ra2 clip
STRAIGHT = np.radians(10) # elbow bend under which the upper-arm twist is taken as undefined
FLICK_DEG = (20.0, -20.0)  # wrist angle at the start and at the release (+ = fingers swung back)
FLICK_PART = 0.55          # share of the crop, at its end, during which the flick happens
SLOW_DOWN_S = 0.2          # s after release (robot time) for every joint to decelerate to a stop
VEL_CAP = 3.0              # rad/s: time is stretched until the peak joint speed is this. 0.5 is cartesian_follow's default

urdf = open(URDF).read()
chain = Chain(urdf, "body_link", "link_rf")
upto = {link: Chain(urdf, "body_link", link) for link in ("link_ra2", "link_ra4", "link_ra6", "link_ra7")}


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s, o, i = np.cos(a), np.sin(a), np.zeros_like(a), np.ones_like(a)
    return np.moveaxis(np.array([[c, -s, o], [s, c, o], [o, o, i]]), [0, 1], [-2, -1])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def arm_angles(u, f):
    """ra0..ra4 (n x 5) from upper-arm directions u and forearm directions f (n x 3, body frame), unclipped.

    In the frame after ra0 (about z), the upper arm is Ry(q1) Rx(q2) Rz(q3) (0,0,-1) =
    (-sin q1 cos q2, sin q2, -cos q1 cos q2), as the URDF chains them. ra2 (-1.57..0) only swings the arm outward
    (-y), so ra0 stays 0 unless the arm crosses in front of the body (u_y > 0): then it yaws inward just enough.
    In the frame after q0..q2 the forearm is Rz(q3) Ry(q4) (0,0,-1) = (-sin q4 cos q3, -sin q4 sin q3, -cos q4).
    """
    # Yaw bringing the arm to y = 0, always turning it forward (+x): choosing -x for a backward arm would flip
    # ra0 by ~pi whenever a hanging arm crosses x = 0. A backward arm beyond ra0's limit gets an ra2 clip instead.
    need = np.where(u[:, 1] > 0, np.arctan2(u[:, 1], u[:, 0]), 0.0)
    # With the arm near vertical the needed yaw jumps; smoothed, the rest becomes a small ra2 clip.
    q0 = uniform_filter1d(np.clip(need, chain.lo[0], chain.hi[0]), YAW_SMOOTH, mode="nearest")
    u0 = np.einsum("nji,nj->ni", rot_z(q0), u)  # u in the frame after ra0
    q2 = np.arcsin(np.clip(u0[:, 1], -1, 1))  # cos q2 >= 0 inside ra2's limits
    q1 = np.unwrap(np.arctan2(-u0[:, 0], -u0[:, 2]))
    q3, q4 = np.full(len(u), np.nan), np.zeros(len(u))
    for i in range(len(u)):
        fa = (rot_z(q0[i]) @ rot_y(q1[i]) @ rot_x(q2[i])).T @ f[i]
        q4[i] = -np.arccos(np.clip(-fa[2], -1, 1))  # ra4 bends one way only (-2.23..0)
        if -q4[i] > STRAIGHT:
            q3[i] = np.arctan2(fa[1], fa[0])
    ok = ~np.isnan(q3)
    if not ok.any():
        q3[:] = 0.0
    else:  # hold the nearest defined twist across straight-arm samples
        q3 = np.interp(np.arange(len(u)), np.flatnonzero(ok), np.unwrap(q3[ok]))
    return np.column_stack([q0, q1, q2, q3, q4])


def direction_error(q, u, f):
    """Largest angle (deg) between the robot's upper arm / forearm (URDF forward kinematics) and u / f."""
    err = 0.0
    for k in range(len(q)):
        s, e = upto["link_ra2"].fk(q[k, :3]).p, upto["link_ra4"].fk(q[k, :5]).p
        w = upto["link_ra6"].fk(q[k, :7]).p
        for got, want in ((e - s, u[k]), (w - e, f[k])):
            got = unit(np.array(list(got)))
            err = max(err, np.degrees(np.arccos(np.clip(got @ want, -1, 1))))
    return err


def retarget(name, start, release):
    d = np.genfromtxt(os.path.join(HERE, name), delimiter=",", names=True)
    keep = (d["timestamp"] >= start - 1e-9) & (d["timestamp"] <= release + 1e-9)
    t = d["timestamp"][keep] - start
    pos = lambda j: savgol_filter(np.column_stack([d[f"{j}_rel_{a}"] for a in "xyz"]), SMOOTH, 3, axis=0)[keep]
    el, wr = pos("elbow") @ TO_BODY.T, pos("wrist") @ TO_BODY.T
    u, f = unit(el), unit(wr - el)

    q = np.zeros((len(t), 8))
    q[:, :5] = arm_angles(u, f)
    # Self-check of the closed form: before clipping, the robot's arm points exactly like the human's.
    assert (e := direction_error(q, u, f)) < 2.0, f"{name}: off by {e:.1f} deg: sign or frame error in arm_angles"

    # Flick joint: of ra6 (x axis) and ra7 (y axis), the one whose axis is closer to body z at the release.
    axes = {6: upto["link_ra6"].fk(q[-1, :7]).M.UnitX(), 7: upto["link_ra7"].fk(q[-1]).M.UnitY()}
    k = max(axes, key=lambda j: abs(axes[j][2]))
    # Sign: increasing the joint must swing the fingers (gripper -z) backward, against the hand's motion.
    fingers = lambda qq: -np.array(list(chain.fk(qq).M.UnitZ()))
    hand_v = np.array(list(chain.fk(q[-1]).p - chain.fk(q[-2]).p))
    dq = np.zeros(8)
    dq[k] = 1e-3
    sign = -1.0 if (fingers(q[-1] + dq) - fingers(q[-1])) @ hand_v > 0 else 1.0
    s = np.clip((t / t[-1] - (1 - FLICK_PART)) / FLICK_PART, 0.0, 1.0)
    q[:, k] = sign * np.radians(FLICK_DEG[0] + (FLICK_DEG[1] - FLICK_DEG[0]) * s**2)

    raw = q.copy()
    q = np.clip(q, chain.lo, chain.hi)
    off = direction_error(q, u, f)  # what clipping costs
    # Stretch time so the controller's spline peaks at VEL_CAP (peak speed scales exactly with 1/stretch).
    peak = spline_peak(*timed(chain, list(q), t, q[0], np.inf)).max()
    stretch = max(1.0, peak / VEL_CAP * 1.001)
    ts = t * stretch
    # After release: keep the release speed, decaying linearly to zero over SLOW_DOWN_S, then clip again.
    dt = ts[-1] - ts[-2]
    n = int(np.ceil(SLOW_DOWN_S / dt))
    uu = SLOW_DOWN_S * np.arange(1, n + 1) / n
    g = uu - uu**2 / (2 * SLOW_DOWN_S)
    q = np.clip(np.vstack([q, q[-1] + g[:, None] * (q[-1] - q[-2]) / dt]), chain.lo, chain.hi)
    ts = np.r_[ts, ts[-1] + uu]
    peak = spline_peak(*timed(chain, list(q), ts, q[0], np.inf)).max()  # recheck after the slow-down

    clipped = np.degrees(np.abs(raw - np.clip(raw, chain.lo, chain.hi)).max(axis=0))
    vel = np.ceil(peak * 100) / 100
    out = os.path.join(HERE, "joints_" + name)
    np.savetxt(out, np.column_stack([ts, q]), delimiter=",", fmt="%.6f", header=(
        f"t,{','.join(chain.names)}  (s, rad) for cartesian_follow.py --arm right --joints FILE "
        f"--max-joint-vel {vel}\n"
        f"{name} from t={start:.3f}s to {release:.3f}s, played {stretch:.2f}x slower: release at t={ts[len(t) - 1]:.3f}s here, "
        f"then {SLOW_DOWN_S}s slow-down\n"
        f"flick on {chain.names[k]} {FLICK_DEG[0]:+.0f} to {FLICK_DEG[1]:+.0f} deg over the last {FLICK_PART:.0%}\n"
        "max clipped to the joint limits (deg): " + " ".join(f"{c:.0f}" for c in clipped) +
        f", arm direction off by up to {off:.0f} deg"))
    worst = int(clipped.argmax())
    print(f"{out}: {stretch:.2f}x slower, peak {peak:.2f} rad/s (--max-joint-vel {vel}), flick on {chain.names[k]}, "
          f"worst clip {clipped[worst]:.0f} deg on {chain.names[worst]}, arm direction off by up to {off:.0f} deg")


with open(os.path.join(HERE, "selections.csv")) as fh:
    for r in csv.DictReader(fh):
        retarget(r["file"], float(r["start"]), float(r["release"]))
