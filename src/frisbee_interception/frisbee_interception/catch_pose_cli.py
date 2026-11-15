"""Compute the frisbee catching pose from the URDF and write it for the existing arm tools.

  ros2 run frisbee_interception frisbee_catch_pose --urdf src/panther_sim/urdf/robot_description.urdf --out catch_pose
  (or without --urdf: reads /robot_description from the running robot)

Writes:
  catch_pose/catch_pose.json        nominal joints of both arms + the pose parameters (loaded by frisbee_catch_trigger)
  catch_pose/catch_pose_left.csv    t,q0..q7,gripper for  cartesian_follow --arm left  --joints ...
  catch_pose/catch_pose_right.csv   t,q0..q7,gripper for  cartesian_follow --arm right --joints ...
Moving into the pose is done with the existing, checked tool (dry run first):
  ros2 run panther_examples cartesian_follow --arm left  --joints catch_pose/catch_pose_left.csv --dry-run
  ros2 run panther_examples cartesian_follow --arm left  --joints catch_pose/catch_pose_left.csv
  (same for --arm right)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from frisbee_interception.core.catch_pose import CatchPoseConfig, CatchPosePlanner


def urdf_hash(urdf: str) -> str:
    return hashlib.sha256(urdf.encode()).hexdigest()[:16]


def compute(urdf: str, cfg: CatchPoseConfig) -> dict:
    pl = CatchPosePlanner(urdf, cfg)
    t0 = time.time()
    nominal = pl.solve_nominal()
    m = cfg.max_correction_m
    worst = {}
    for arm in nominal:
        w = 0.0
        for dy in (-m, 0.0, m):
            for dz in (-m, 0.0, m):
                r = pl.correction(dy, dz)
                if r is not None:
                    w = max(w, float(np.abs(r[0][arm] - nominal[arm]).max()))
        worst[arm] = round(w, 3)
    hands = {arm: pl.chains[arm].fk(q)[:3, 3].round(4).tolist() for arm, q in nominal.items()}
    return {
        "urdf_hash": urdf_hash(urdf), "config": cfg.__dict__, "joint_names": pl.joint_names,
        "nominal": {arm: q.tolist() for arm, q in nominal.items()},
        "hand_positions": hands,
        "rotation": {arm: pl.rot[arm].tolist() for arm in nominal},
        "limit_margin_rad": {arm: round(pl.chains[arm].limit_margin(q), 3) for arm, q in nominal.items()},
        "worst_joint_change_for_max_correction_rad": worst,
        "solve_time_s": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", help="URDF file (default: /robot_description from the running robot)")
    ap.add_argument("--out", default="catch_pose")
    ap.add_argument("--center", type=float, nargs=3, default=CatchPoseConfig().center, metavar=("X", "Y", "Z"))
    ap.add_argument("--half-spacing", type=float, default=CatchPoseConfig().half_spacing)
    ap.add_argument("--max-correction", type=float, default=CatchPoseConfig().max_correction_m)
    a = ap.parse_args()
    if a.urdf:
        urdf = Path(a.urdf).read_text()
    else:
        import rclpy  # noqa: PLC0415
        from rclpy.qos import DurabilityPolicy, QoSProfile  # noqa: PLC0415
        from std_msgs.msg import String  # noqa: PLC0415

        rclpy.init()
        node = rclpy.create_node("frisbee_catch_pose")
        got = {}
        node.create_subscription(String, "/robot_description", lambda m: got.setdefault("urdf", m.data),
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        for _ in range(50):
            if "urdf" in got:
                break
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_node()
        rclpy.shutdown()
        if "urdf" not in got:
            raise SystemExit("no /robot_description: pass --urdf FILE")
        urdf = got["urdf"]
    cfg = CatchPoseConfig(center=list(a.center), half_spacing=a.half_spacing, max_correction_m=a.max_correction)
    try:
        res = compute(urdf, cfg)
    except RuntimeError as e:
        raise SystemExit(f"no catching pose: {e}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "catch_pose.json").write_text(json.dumps(res, indent=1))
    for arm, q in res["nominal"].items():
        names = res["joint_names"][arm]
        (out / f"catch_pose_{arm}.csv").write_text(
            f"# frisbee catching pose ({arm} arm), from frisbee_catch_pose; gripper column 0 = open\n"
            f"# t,{','.join(names)},gripper\n"
            "0.0," + ",".join(f"{v:.6f}" for v in q) + ",0.0\n")
    print(json.dumps({k: res[k] for k in ("hand_positions", "limit_margin_rad",
                                          "worst_joint_change_for_max_correction_rad", "solve_time_s")}, indent=1))
    print(f"wrote {out}/catch_pose.json, catch_pose_left.csv, catch_pose_right.csv")


if __name__ == "__main__":
    main()
