"""Move every joint a small step, check the measured position follows, and move it back. Robot or sim.

  ros2 run panther_examples joint_check                         # plan only: current position and target per joint
  ros2 run panther_examples joint_check --yes                   # arms, head, grippers, lift
  ros2 run panther_examples joint_check --yes --groups head lift --delta 0.05

Each joint steps toward the side with more room inside its URDF limits (lift: toward mid travel), one at a time.
Arms: the forward controller is swapped for the trajectory controllers (as in PR #521) and restored at the end.
Head: forward controller. Grippers and lift: their robot topics. The driving base is not checked.
Exit code 0 = every joint reached its target, 1 = at least one did not.
"""
import argparse
import sys
import time
import xml.etree.ElementTree as ET

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from controller_manager_msgs.srv import ListControllers, SwitchController
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from unix_msgs.msg import GripperCommand

from panther_examples.robot import (
    FORWARD, HEAD, HEAD_FORWARD, LEFT_ARM, LEFT_GRIPPER_CMD_TOPIC, LEFT_GRIPPER_STATE_TOPIC, LEFT_TRAJ, LIFT_CMD_TOPIC,
    LIFT_STATE_TOPIC, LIFT_VEL, MAX_LIFT, MIN_LIFT, RIGHT_ARM, RIGHT_GRIPPER_CMD_TOPIC, RIGHT_GRIPPER_STATE_TOPIC,
    RIGHT_TRAJ,
)

ARM_VEL = 0.2  # rad/s for the step, under cartesian_follow's 0.5 default
TOL = {"arms": 0.02, "head": 0.02, "grippers": 0.05, "lift": 0.01}  # rad, rad, 0..1, m
GRIPPER_STEP, LIFT_STEP = 0.3, 0.05


def step_target(q, lower, upper, delta):
    """q moved by delta toward the side with more room, or None if neither side has delta of room."""
    if upper - q >= q - lower:
        return q + delta if upper - q >= delta else None
    return q - delta if q - lower >= delta else None


class Check:
    def __init__(self, node):
        self.node = node
        self.pos = {}  # joint name -> latest position; grippers as joint_lf / joint_rf, lift as body_joint
        node.create_subscription(JointState, "/joint_states", self.on_states, 10)
        for topic, name in ((LEFT_GRIPPER_STATE_TOPIC, "joint_lf"), (RIGHT_GRIPPER_STATE_TOPIC, "joint_rf"),
                            (LIFT_STATE_TOPIC, "body_joint")):
            node.create_subscription(JointState, topic, lambda m, n=name: self.on_single(n, m), 10)
        self.arm = {LEFT_TRAJ: node.create_publisher(JointTrajectory, f"/{LEFT_TRAJ}/joint_trajectory", 10),
                    RIGHT_TRAJ: node.create_publisher(JointTrajectory, f"/{RIGHT_TRAJ}/joint_trajectory", 10)}
        self.head = node.create_publisher(Float64MultiArray, f"/{HEAD_FORWARD}/commands", 10)
        self.gripper = {"joint_lf": node.create_publisher(GripperCommand, LEFT_GRIPPER_CMD_TOPIC, 10),
                        "joint_rf": node.create_publisher(GripperCommand, RIGHT_GRIPPER_CMD_TOPIC, 10)}
        self.lift = node.create_publisher(JointState, LIFT_CMD_TOPIC, 10)
        self.urdf = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String, "/robot_description", lambda m: setattr(self, "urdf", m.data), latched)

    def on_states(self, msg):
        self.pos.update(zip(msg.name, msg.position))

    def on_single(self, name, msg):  # gripper and lift state topics carry one unnamed position
        if msg.position:
            self.pos[name] = msg.position[0]

    def spin_until(self, done, timeout):
        end = time.monotonic() + timeout
        while not done() and time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return done()

    def limits(self):
        root = ET.fromstring(self.urdf)
        return {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
                for j in root.findall("joint") if j.find("limit") is not None}

    def call(self, srv, name, req):
        cli = self.node.create_client(srv, name)
        if not cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"{name} not found")
        f = cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, f, timeout_sec=10.0)
        return f.result()

    def switch(self, on, off):
        if not (on or off):
            return
        req = SwitchController.Request(activate_controllers=on, deactivate_controllers=off,
                                       strictness=SwitchController.Request.STRICT)
        req.timeout.sec = 5
        res = self.call(SwitchController, "/controller_manager/switch_controller", req)
        if not (res and res.ok):
            raise RuntimeError(f"switch_controller failed (activate {on}, deactivate {off})")

    def command(self, group, joint, value, start):
        """Send one joint to value; the rest of its group stays at start (dict joint -> start position)."""
        if group == "arms":
            names, jtc = (LEFT_ARM, LEFT_TRAJ) if joint in LEFT_ARM else (RIGHT_ARM, RIGHT_TRAJ)
            t = max(1.0, abs(value - self.pos[joint]) / ARM_VEL)
            point = JointTrajectoryPoint(positions=[value if n == joint else start[n] for n in names])
            point.time_from_start.sec, point.time_from_start.nanosec = int(t), int(t % 1 * 1e9)
            msg, pub = JointTrajectory(joint_names=names, points=[point]), self.arm[jtc]
        elif group == "head":
            msg, pub = Float64MultiArray(data=[value if n == joint else start[n] for n in HEAD]), self.head
        elif group == "grippers":
            msg, pub = GripperCommand(mode="parallel"), self.gripper[joint]
            msg.command.name, msg.command.position, msg.command.velocity, msg.command.effort = [""], [value], [0.5], [0.5]
        else:
            msg, pub = JointState(name=[""], position=[value]), self.lift
        self.spin_until(lambda: pub.get_subscription_count() > 0, 1.0)
        for _ in range(3):  # topics without a handshake: repeat so a late-matched subscriber still gets it
            pub.publish(msg)
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def step(self, group, joint, target, start):
        """Move to target, wait until within tolerance, move back. Returns (reached position, ok)."""
        timeout = (abs(target - start[joint]) / (LIFT_VEL if group == "lift" else ARM_VEL)) + 3.0
        near = lambda v: abs(self.pos.get(joint, float("nan")) - v) <= TOL[group]
        self.command(group, joint, target, start)
        ok = self.spin_until(lambda: near(target), timeout)
        reached = self.pos.get(joint)
        self.command(group, joint, start[joint], start)
        back = self.spin_until(lambda: near(start[joint]), timeout)
        return reached, ok and back


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="move the joints; without it, print the plan only")
    ap.add_argument("--groups", nargs="+", default=["arms", "head", "grippers", "lift"],
                    choices=["arms", "head", "grippers", "lift"])
    ap.add_argument("--delta", type=float, default=0.1, help="arm and head step, rad (default 0.1)")
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("joint_check")
    c = Check(node)
    joints = {"arms": LEFT_ARM + RIGHT_ARM, "head": HEAD, "grippers": ["joint_lf", "joint_rf"], "lift": ["body_joint"]}
    wanted = [(g, j) for g in args.groups for j in joints[g]]
    if not c.spin_until(lambda: c.urdf and all(j in c.pos for _, j in wanted), 5.0):
        missing = [j for _, j in wanted if j not in c.pos] + ([] if c.urdf else ["/robot_description"])
        sys.exit(f"no state for: {' '.join(missing)}")
    start = dict(c.pos)
    lim = c.limits()

    plan = []
    for g, j in wanted:
        if g == "grippers":
            target = start[j] + GRIPPER_STEP if start[j] < 0.5 else start[j] - GRIPPER_STEP
        elif g == "lift":
            target = start[j] + LIFT_STEP if start[j] < (MIN_LIFT + MAX_LIFT) / 2 else start[j] - LIFT_STEP
        else:
            target = step_target(start[j], *lim[j], args.delta)
        plan.append((g, j, target))
    print(f"{'joint':12} {'now':>8} {'target':>8}")
    for g, j, target in plan:
        print(f"{j:12} {start[j]:8.3f} {'no room' if target is None else f'{target:8.3f}':>8}")
    if not args.yes:
        print("Plan only. Add --yes to move.")
        return 0

    restore = ([], [])
    failed, skipped = [], []
    try:
        if "arms" in args.groups:
            states = {s.name: s.state for s in c.call(ListControllers, "/controller_manager/list_controllers",
                                                      ListControllers.Request()).controller}
            on = [t for t in (LEFT_TRAJ, RIGHT_TRAJ) if states.get(t) != "active"]
            off = [FORWARD] if states.get(FORWARD) == "active" else []
            c.switch(on, off)
            restore = (off, on)
        print(f"\n{'joint':12} {'target':>8} {'reached':>8}  result")
        for g, j, target in plan:
            if target is None:
                print(f"{j:12} {'':>8} {'':>8}  SKIP (no room inside limits)")
                skipped.append(j)
                continue
            reached, ok = c.step(g, j, target, start)
            print(f"{j:12} {target:8.3f} {'none' if reached is None else f'{reached:8.3f}':>8}  {'PASS' if ok else 'FAIL'}")
            if not ok:
                failed.append(j)
    finally:
        c.switch(*restore)
    print(f"\n{len(plan) - len(failed) - len(skipped)}/{len(plan)} joints passed"
          + (f"; failed: {' '.join(failed)}" if failed else "") + (f"; skipped: {' '.join(skipped)}" if skipped else ""))
    node.destroy_node()
    rclpy.shutdown()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
