"""Render a synthetic straight-line frisbee throw toward the camera, with exact ground truth (Phase 6 test).

  python -m frisbee_interception.synthetic --background frame.jpg --out throw.mp4 --truth throw.json

A shaded, slightly tilted disc of diameter D flies along a straight line at constant velocity in the camera
optical frame and is rendered with a pinhole model (intrinsics from the horizontal FOV, square pixels).
Ground truth: for each catching distance z_c the crossing time, pixel and the apparent diameter there.
"""

from __future__ import annotations

import argparse
import json
import math

import cv2
import numpy as np


def render(background: np.ndarray, out: str, truth_path: str, fps: float = 30.0, hfov_deg: float = 77.32,
           start=(0.5, -0.4, 9.0), vel=(-0.35, 0.25, -9.0), diameter=0.27, tilt_deg=35.0,
           lead_s=3.0, tail_s=0.3, z_catch=(1.0, 1.5), noise=2.0, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    h, w = background.shape[:2]
    fx = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    fy, cx, cy = fx, w / 2, h / 2
    start, vel = np.array(start, float), np.array(vel, float)
    t_end = (0.6 - start[2]) / vel[2]  # stop at 0.6 m from the camera
    n = int((lead_s + t_end + tail_s) * fps)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frames = []
    for i in range(n):
        t = i / fps - lead_s  # flight starts at t = 0 (video time lead_s)
        img = background.copy()
        if 0 <= t <= t_end:
            p = start + vel * t
            u, v = cx + fx * p[0] / p[2], cy + fy * p[1] / p[2]
            d = fx * diameter / p[2]
            minor = d * math.cos(math.radians(tilt_deg + 10 * math.sin(6 * t)))  # tilt wobbles a little
            ang = 15 + 20 * t
            cv2.ellipse(img, ((u, v), (d, minor), ang), (60, 210, 90), -1, cv2.LINE_AA)
            cv2.ellipse(img, ((u, v), (d * 0.7, minor * 0.7), ang), (30, 150, 50), max(1, int(d / 30)), cv2.LINE_AA)
            frames.append({"frame": i, "t_video": i / fps, "u": u, "v": v, "d": d, "z": p[2]})
        if noise > 0:
            img = np.clip(img.astype(np.int16) + rng.normal(0, noise, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
        writer.write(img)
    writer.release()
    truth = {"fps": fps, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "diameter_m": diameter, "lead_s": lead_s,
             "start": start.tolist(), "vel": vel.tolist(), "frames": frames, "catch": {}}
    for zc in z_catch:
        tc = (zc - start[2]) / vel[2]
        pc = start + vel * tc
        truth["catch"][str(zc)] = {
            "t_video": lead_s + tc, "point_cam": pc.tolist(),
            "pixel": [cx + fx * pc[0] / pc[2], cy + fy * pc[1] / pc[2]], "diameter_px": fx * diameter / zc,
        }
    with open(truth_path, "w") as f:
        json.dump(truth, f, indent=1)
    return truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--background", required=True, help="image, or video (first frame is used)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()
    bg = cv2.imread(a.background)
    if bg is None:
        ok, bg = cv2.VideoCapture(a.background).read()
        if not ok:
            raise SystemExit(f"cannot read background {a.background}")
    t = render(bg, a.out, a.truth, a.fps)
    print(json.dumps({k: t[k] for k in ("fx", "cx", "cy")} | {"catch": t["catch"]}, indent=1))


if __name__ == "__main__":
    main()
