"""ROS-level smoke test: listen to the running frisbee pipeline and check its outputs and safety behaviour.

  python3 ros_smoke_check.py --mode image  --truth throw_truth.json --duration 20 --report out.json
  python3 ros_smoke_check.py --mode metric --truth throw_truth.json --duration 20 --report out.json

image : disarmed. Arming must be refused (no workspace), no catch target may ever be published, valid
        predictions must hit the true crossing pixel.
metric: arms via ~/enable_interception, expects catch targets near the true 3D crossing point, then publishes
        /frisbee/disable and expects no further catch targets.
"""

import argparse
import json
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import Empty
from std_srvs.srv import SetBool

from frisbee_msgs.msg import FrisbeeDetectionArray, FrisbeeIntercept, FrisbeeTrack


class Checker(Node):
    def __init__(self):
        super().__init__("frisbee_smoke_check")
        self.n_det = self.n_det_nonempty = self.n_track = self.n_confirmed = 0
        self.intercepts, self.targets = [], []
        self.disabled_at = None
        self.create_subscription(FrisbeeDetectionArray, "/frisbee/detections", self._det, 10)
        self.create_subscription(FrisbeeTrack, "/frisbee/track", self._trk, 10)
        self.create_subscription(FrisbeeIntercept, "/frisbee/intercept", self._int, 50)
        self.create_subscription(PointStamped, "/frisbee/catch_target", self._tgt, 50)
        self.disable_pub = self.create_publisher(Empty, "/frisbee/disable", 10)
        self.enable_cli = self.create_client(SetBool, "/frisbee_interceptor/enable_interception")

    def _det(self, m):
        self.n_det += 1
        self.n_det_nonempty += bool(m.detections)

    def _trk(self, m):
        self.n_track += 1
        self.n_confirmed += bool(m.confirmed)

    def _int(self, m):
        self.intercepts.append({
            "valid": m.valid, "status": m.status, "mode": m.mode, "tti": m.time_to_intercept,
            "pixel": [m.pixel_u, m.pixel_v], "point": [m.catch_point.x, m.catch_point.y, m.catch_point.z],
            "confidence": m.confidence, "in_workspace": m.in_workspace, "enabled": m.interception_enabled,
        })

    def _tgt(self, m):
        self.targets.append({"wall": time.monotonic(), "point": [m.point.x, m.point.y, m.point.z]})

    def call_enable(self, value: bool):
        if not self.enable_cli.wait_for_service(timeout_sec=15.0):
            return None
        fut = self.enable_cli.call_async(SetBool.Request(data=value))
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        r = fut.result()
        return None if r is None else {"success": r.success, "message": r.message}


def spin_for(node, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["image", "metric"], required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    truth = json.load(open(a.truth))["catch"]["1.0"]

    rclpy.init()
    n = Checker()
    checks, report = {}, {"mode": a.mode}
    if a.mode == "image":
        res = n.call_enable(True)
        report["enable_response"] = res
        checks["arming refused without workspace"] = res is not None and not res["success"]
        spin_for(n, a.duration)
    else:
        res = n.call_enable(True)
        report["enable_response"] = res
        checks["arming accepted with workspace"] = res is not None and res["success"]
        spin_for(n, a.duration * 0.7)
        n.disable_pub.publish(Empty())
        n.disabled_at = time.monotonic()
        spin_for(n, a.duration * 0.3)

    valid = [i for i in n.intercepts if i["valid"]]
    report.update({
        "detection_msgs": n.n_det, "detection_msgs_nonempty": n.n_det_nonempty,
        "track_msgs": n.n_track, "confirmed_track_msgs": n.n_confirmed,
        "intercept_msgs": len(n.intercepts), "valid_intercepts": len(valid), "catch_targets": len(n.targets),
        "statuses": sorted({i["status"].split(" (")[0] for i in n.intercepts})[:20],
        "truth": truth,
    })
    checks["detections published"] = n.n_det_nonempty >= 10
    checks["track confirmed"] = n.n_confirmed >= 3
    checks["valid predictions"] = len(valid) >= 3
    if a.mode == "image":
        px_err = [math.hypot(i["pixel"][0] - truth["pixel"][0], i["pixel"][1] - truth["pixel"][1]) for i in valid]
        report["pixel_errors_with_tti"] = [[round(e, 1), round(i["tti"], 3)] for e, i in zip(px_err, valid)]
        # same budget as test_synthetic_throw: tight near contact, coarser 0.3-0.65 s out (80 px = 18 cm at 1 m)
        late = [e for e, i in zip(px_err, valid) if i["tti"] <= 0.3]
        early = [e for e, i in zip(px_err, valid) if i["tti"] > 0.3]
        checks["predicted pixel within 20 px in the last 0.3 s"] = bool(late) and max(late) < 20
        checks["predicted pixel within 80 px earlier"] = max(early, default=0.0) < 80
        checks["no catch target while disarmed"] = len(n.targets) == 0
    else:
        errs = [math.dist(i["point"], truth["point_cam"]) for i in valid]
        report["point_errors_m"] = [round(e, 3) for e in errs]
        checks["metric mode used"] = all(i["mode"] == "metric" for i in valid) and bool(valid)
        checks["catch point within 0.12 m"] = bool(errs) and max(errs) < 0.12
        before = [t for t in n.targets if t["wall"] < n.disabled_at]
        after = [t for t in n.targets if t["wall"] > n.disabled_at + 0.2]
        report["targets_before_disable"], report["targets_after_disable"] = len(before), len(after)
        checks["catch targets sent while armed"] = len(before) >= 1
        checks["no catch target after /frisbee/disable"] = len(after) == 0
    report["checks"] = checks
    report["passed"] = all(checks.values())
    json.dump(report, open(a.report, "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "truth"}, indent=1))
    n.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
