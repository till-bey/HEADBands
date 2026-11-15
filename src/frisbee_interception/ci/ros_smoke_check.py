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
from std_msgs.msg import Empty, String
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
        self.catch_cli = self.create_client(SetBool, "/frisbee_catch_trigger/enable")
        self.grips, self.catch_events = [], []
        from unix_msgs.msg import GripperCommand  # noqa: PLC0415
        for topic in ("/left_gripper/gripper_command", "/right_gripper/gripper_command"):
            self.create_subscription(GripperCommand, topic,
                                     lambda m, tp=topic: self.grips.append({"wall": time.monotonic(), "topic": tp,
                                                                            "position": list(m.command.position)}), 10)
        self.create_subscription(String, "/frisbee/catch_event", lambda m: self.catch_events.append(json.loads(m.data)), 50)
        self.mock = None  # last /mock_robot/status (metric run); its times are CLOCK_MONOTONIC like ours
        self.create_subscription(String, "/mock_robot/status", lambda m: setattr(self, "mock", json.loads(m.data)), 10)

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

    def call_enable(self, value: bool, cli=None):
        cli = cli or self.enable_cli
        if not cli.wait_for_service(timeout_sec=15.0):
            return None
        fut = cli.call_async(SetBool.Request(data=value))
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
    ap.add_argument("--truth-body", help="metric: true crossing point in body_link")
    ap.add_argument("--urdf", help="metric: URDF, to check where the corrected hands go (FK)")
    a = ap.parse_args()
    truth = json.load(open(a.truth))["catch"]["1.0"]
    truth_body = json.load(open(a.truth_body))["point_body"] if a.truth_body else None

    rclpy.init()
    n = Checker()
    checks, report = {}, {"mode": a.mode}
    if a.mode == "image":
        res = n.call_enable(True)
        report["enable_response"] = res
        checks["arming refused without workspace"] = res is not None and not res["success"]
        cres = n.call_enable(True, n.catch_cli)
        report["catch_enable_response"] = cres
        checks["catch trigger arming refused without calibration"] = cres is not None and not cres["success"]
        spin_for(n, a.duration)
    else:
        res = n.call_enable(True)
        report["enable_response"] = res
        checks["arming accepted with workspace"] = res is not None and res["success"]
        # the catch node needs the robot description + catching pose first: retry for up to 30 s
        t_end, cres = time.monotonic() + 30.0, None
        while time.monotonic() < t_end:
            cres = n.call_enable(True, n.catch_cli)
            if cres and cres["success"]:
                break
            spin_for(n, 1.0)
        report["catch_enable_response"] = cres
        checks["catch armed (calibration, catching pose, controller switch)"] = cres is not None and cres["success"]
        spin_for(n, a.duration * 0.7)
        n.disable_pub.publish(Empty())
        n.disabled_at = time.monotonic()
        spin_for(n, a.duration * 0.3)

    valid = [i for i in n.intercepts if i["valid"]]
    report.update({
        "detection_msgs": n.n_det, "detection_msgs_nonempty": n.n_det_nonempty,
        "track_msgs": n.n_track, "confirmed_track_msgs": n.n_confirmed,
        "intercept_msgs": len(n.intercepts), "valid_intercepts": len(valid), "catch_targets": len(n.targets),
        "catch_events": sorted({e["event"] for e in n.catch_events}),
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
        checks["grippers never commanded"] = len(n.grips) == 0
    else:
        errs = [math.dist(i["point"], truth_body) for i in valid]
        report["point_errors_m"] = [round(e, 3) for e in errs]
        checks["metric mode used"] = all(i["mode"] == "metric" for i in valid) and bool(valid)
        checks["catch point within 0.12 m"] = bool(errs) and max(errs) < 0.12
        before = [t for t in n.targets if t["wall"] < n.disabled_at]
        after = [t for t in n.targets if t["wall"] > n.disabled_at + 0.2]
        report["targets_before_disable"], report["targets_after_disable"] = len(before), len(after)
        checks["catch targets sent while armed"] = len(before) >= 1
        checks["no catch target after /frisbee/disable"] = len(after) == 0
        g_before = [g for g in n.grips if g["wall"] < n.disabled_at]
        g_after = [g for g in n.grips if g["wall"] > n.disabled_at + 0.2]
        fired = [e for e in n.catch_events if e["event"] == "fired"]
        report["gripper_commands_before_disable"], report["gripper_commands_after_disable"] = len(g_before), len(g_after)
        report["catch_fired_events"] = fired
        checks["grippers closed while armed (both topics)"] = {g["topic"] for g in g_before} == {
            "/left_gripper/gripper_command", "/right_gripper/gripper_command"}
        checks["one close per throw (refractory)"] = 1 <= len(fired) <= 1 + int(a.duration * 0.7 / 2.0)
        checks["no gripper command after /frisbee/disable"] = len(g_after) == 0
        # ---- hand corrections (mock robot: controller switches and trajectory goals it received)
        corr = [e for e in n.catch_events if e["event"] == "correction"]
        report["corrections"] = corr[:10]
        mock = n.mock or {"switches": [], "goals": []}
        report["mock_switches"] = mock["switches"]
        report["mock_goals"] = len(mock["goals"])
        sw = mock["switches"]
        traj = {"left_arm_joint_trajectory_position_controller", "right_arm_joint_trajectory_position_controller"}
        fwd = "dual_arm_forward_position_controller"
        checks["arming switched to the trajectory controllers"] = bool(sw) and set(sw[0]["activate"]) == traj and \
            sw[0]["deactivate"] == [fwd]
        checks["disarming restored the forward controller"] = len(sw) >= 2 and sw[-1]["activate"] == [fwd]
        checks["corrections sent while armed"] = len(corr) >= 1
        goals_before = [g for g in mock["goals"] if g["t"] < n.disabled_at]
        goals_after = [g for g in mock["goals"] if g["t"] > n.disabled_at + 0.3]
        checks["both arms received accepted goals"] = {g["arm"] for g in goals_before if g["accepted"]} == {"left", "right"}
        checks["no arm goal after /frisbee/disable"] = len(goals_after) == 0
        checks["corrections within the joint speed cap"] = all(
            c["max_joint_change"] / c["duration_s"] <= 0.5 / 1.5 * 1.0001 for c in corr)
        if a.urdf and goals_before:
            from frisbee_interception.core.kinematics import ArmChain  # noqa: PLC0415

            urdf = open(a.urdf).read()
            center = [0.45, 0.0, -0.20]
            true_off = [max(-0.1, min(0.1, truth_body[1] - center[1])), max(-0.1, min(0.1, truth_body[2] - center[2]))]
            # 1) the node aims at the predicted point: compare its offsets with the true crossing (prediction error)
            aim_err = [math.dist(c["offset"], true_off) for c in corr]
            report["true_offset_m"] = [round(v, 3) for v in true_off]
            report["aimed_offset_error_m"] = [round(e, 3) for e in aim_err]
            checks["aimed correction within 5 cm of the true crossing"] = bool(aim_err) and max(aim_err) < 0.05
            # 2) the hands go where the node aimed (IK + goal): FK of the last goal vs the last correction's offset
            off = corr[-1]["offset"] if corr else [0.0, 0.0]
            hand_err = {}
            for arm, tip, sgn in (("left", "link_lf", 1.0), ("right", "link_rf", -1.0)):
                last = [g for g in goals_before if g["arm"] == arm and g["accepted"]][-1]
                ch = ArmChain(urdf, "body_link", tip)
                q = [last["positions"][last["joint_names"].index(nm)] for nm in ch.names]
                p = ch.fk(q)[:3, 3]
                want = [center[0], center[1] + sgn * 0.13 + off[0], center[2] + off[1]]
                hand_err[arm] = round(math.dist(p.tolist(), want), 4)
            report["corrected_hand_error_m"] = hand_err
            checks["corrected hands within 1 cm of the aimed point (FK of the sent goals)"] = max(hand_err.values()) < 0.01
    report["checks"] = checks
    report["passed"] = all(checks.values())
    json.dump(report, open(a.report, "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "truth"}, indent=1))
    n.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
