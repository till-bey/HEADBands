#!/usr/bin/env python3
"""One trajectory topic for the whole upper body: both arms, head, lift and grippers.

  ros2 run panther_examples master_controller

Send a trajectory_msgs/JointTrajectory to /master_controller/joint_trajectory with any mix of
joint_la0..7, joint_ra0..7, joint_head0 (pitch), joint_head1 (yaw), body_joint (lift, m, 0 = bottom),
joint_lf, joint_rf (gripper, 0 open .. 1 closed). Joints left out hold their current position.
It is split onto the robot's own interfaces, so it runs on the robot and in the sim:

  arms      -> /left|right_arm_joint_trajectory_position_controller/joint_trajectory (the controller interpolates)
  head      -> /head_forward_position_controller/commands, each point sent at its time
  lift      -> /lifting_controller/commands, each point sent at its time, speed from the point spacing
  grippers  -> /left|right_gripper/gripper_command, same

On start it switches both arm trajectory controllers in for dual_arm_forward_position_controller
(as cartesian_follow does), and switches back on Ctrl-C. A new trajectory replaces the pending
points of the joints it names, as a trajectory controller does.

  ros2 topic pub --once /master_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \\
    "{joint_names: [joint_head0, body_joint, joint_lf, joint_la1],
      points: [{positions: [0.3, 0.6, 1.0, 0.5], time_from_start: {sec: 4}}]}"
"""
import sys

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from controller_manager_msgs.srv import ListControllers, SwitchController
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from unix_msgs.msg import GripperCommand

from panther_examples.robot import (FORWARD, HEAD, HEAD_FORWARD, LEFT_ARM, LEFT_GRIPPER_CMD_TOPIC, LEFT_GRIPPER_STATE_TOPIC,
                                    LEFT_TRAJ, LIFT_CMD_TOPIC, LIFT_STATE_TOPIC, LIFT_VEL, MAX_LIFT, MIN_LIFT, RIGHT_ARM,
                                    RIGHT_GRIPPER_CMD_TOPIC, RIGHT_GRIPPER_STATE_TOPIC, RIGHT_TRAJ, single_position)

ARM_GROUPS = {LEFT_TRAJ: LEFT_ARM, RIGHT_TRAJ: RIGHT_ARM}
STEPPED = HEAD + ["body_joint", "joint_lf", "joint_rf"]  # no trajectory controller on the robot: sent point by point
# shortcut: gripper speed (strokes/s) when the timing gives none; the robot's real gripper speed is unmeasured.
GRIPPER_VEL = 0.5


def seconds(duration):
    return duration.sec + duration.nanosec * 1e-9


def split(traj, current):
    """({arm controller: JointTrajectory}, [(t, joint, position, speed or None)]) for one master trajectory.
    Joints not in traj hold their position in `current`. Raises ValueError on an unknown joint, KeyError on an
    unknown current position."""
    unknown = set(traj.joint_names) - set(LEFT_ARM + RIGHT_ARM + STEPPED)
    if unknown:
        raise ValueError(f"unknown joints {sorted(unknown)}")
    idx = {n: i for i, n in enumerate(traj.joint_names)}
    arms = {}
    for ctrl, names in ARM_GROUPS.items():
        if not idx.keys() & set(names):
            continue
        # the arm controllers reject partial goals, so the missing joints are filled in
        arms[ctrl] = JointTrajectory(joint_names=names, points=[JointTrajectoryPoint(
            positions=[p.positions[idx[n]] if n in idx else current[n] for n in names],
            velocities=[p.velocities[idx[n]] if n in idx else 0.0 for n in names] if p.velocities else [],
            time_from_start=p.time_from_start) for p in traj.points])
    steps = []
    for n in STEPPED:
        if n not in idx:
            continue
        prev, t_prev = current[n], 0.0
        for p in traj.points:
            t, q = seconds(p.time_from_start), p.positions[idx[n]]
            steps.append((t, n, q, abs(q - prev) / (t - t_prev) if t > t_prev else None))
            prev, t_prev = q, t
    return arms, sorted(steps, key=lambda s: s[0])


class MasterController(Node):
    def __init__(self):
        super().__init__("master_controller")
        self.current, self.steps, self.head = {}, [], None
        self.arm_pub = {c: self.create_publisher(JointTrajectory, f"/{c}/joint_trajectory", 10) for c in ARM_GROUPS}
        self.head_pub = self.create_publisher(Float64MultiArray, f"/{HEAD_FORWARD}/commands", 10)
        self.lift_pub = self.create_publisher(JointState, LIFT_CMD_TOPIC, 10)
        self.grip_pub = {"joint_lf": self.create_publisher(GripperCommand, LEFT_GRIPPER_CMD_TOPIC, 10),
                         "joint_rf": self.create_publisher(GripperCommand, RIGHT_GRIPPER_CMD_TOPIC, 10)}
        self.create_subscription(JointState, "/joint_states", lambda m: self.current.update(zip(m.name, m.position)), 10)
        # the robot's /joint_states has no lift or gripper; they have their own state topics
        for topic, joint in ((LIFT_STATE_TOPIC, "body_joint"), (LEFT_GRIPPER_STATE_TOPIC, "joint_lf"),
                             (RIGHT_GRIPPER_STATE_TOPIC, "joint_rf")):
            self.create_subscription(JointState, topic, lambda m, j=joint: self.on_single(j, m), 10)
        self.create_subscription(JointTrajectory, "~/joint_trajectory", self.on_trajectory, 10)
        self.create_timer(0.02, self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_single(self, joint, msg):
        q = single_position(msg)
        if q is not None:
            self.current[joint] = q

    def on_trajectory(self, msg):
        try:
            arms, steps = split(msg, self.current)
            head = [self.current[n] for n in HEAD] if set(HEAD) & set(msg.joint_names) else self.head
        except (KeyError, ValueError) as e:
            self.get_logger().error(f"trajectory dropped, nothing sent: {e} (no state received yet?)")
            return
        for ctrl, traj in arms.items():
            self.arm_pub[ctrl].publish(traj)
        t0, names = self.now(), set(msg.joint_names)
        self.head = head
        self.steps = sorted([s for s in self.steps if s[1] not in names] + [(t0 + t, *rest) for t, *rest in steps],
                            key=lambda s: s[0])

    def tick(self):
        now = self.now()
        while self.steps and self.steps[0][0] <= now:
            _, joint, q, speed = self.steps.pop(0)
            if joint in HEAD:
                self.head[HEAD.index(joint)] = q
                self.head_pub.publish(Float64MultiArray(data=[float(v) for v in self.head]))
            elif joint == "body_joint":
                q = min(max(q, MIN_LIFT), MAX_LIFT)
                self.lift_pub.publish(JointState(name=[""], position=[float(q)], velocity=[min(speed or LIFT_VEL, LIFT_VEL)]))
            else:
                cmd = GripperCommand()
                cmd.command.name, cmd.command.effort = [""], [0.5]
                cmd.command.position = [float(min(max(q, 0.0), 1.0))]
                cmd.command.velocity = [float(min(speed or GRIPPER_VEL, 1.0))]
                self.grip_pub[joint].publish(cmd)


def call(node, cli, req):
    if not cli.wait_for_service(timeout_sec=5.0):
        raise RuntimeError(f"{cli.srv_name} not found")
    f = cli.call_async(req)
    rclpy.spin_until_future_complete(node, f, timeout_sec=10.0)
    return f.result()


def switch(node, activate, deactivate):
    if not (activate or deactivate):
        return
    req = SwitchController.Request(activate_controllers=activate, deactivate_controllers=deactivate,
                                   strictness=SwitchController.Request.STRICT, activate_asap=True)
    req.timeout.sec = 5
    res = call(node, node.create_client(SwitchController, "/controller_manager/switch_controller"), req)
    if not (res and res.ok):
        raise RuntimeError(f"switch_controller failed (activate {activate}, deactivate {deactivate})")


def main():
    # keep the context alive on Ctrl-C so the controllers can be switched back
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = MasterController()
    states = {c.name: c.state for c in call(node, node.create_client(
        ListControllers, "/controller_manager/list_controllers"), ListControllers.Request()).controller}
    missing = [c for c in ARM_GROUPS if c not in states]
    if missing:
        sys.exit(f"{missing} not loaded")
    # same switch as cartesian_follow (PR #521): both arm trajectory controllers in, the forward controller out
    activate = [c for c in ARM_GROUPS if states[c] != "active"]
    deactivate = [FORWARD] if states.get(FORWARD) == "active" else []
    switch(node, activate, deactivate)
    try:
        node.get_logger().info("listening on /master_controller/joint_trajectory")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        switch(node, deactivate, activate)  # put the controllers back as they were
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
