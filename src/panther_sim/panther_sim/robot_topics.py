"""Sim only: the robot's lift and gripper topics, served on top of the sim's trajectory controllers.

On the robot these are vendor plugins (lifting_controller3, omni_picker_controller) that only exist there.
This node gives the same topics and message types, so code written for the robot runs here unchanged:

  /lifting_controller/commands      sensor_msgs/JointState      position (m) or velocity (m/s)
  /left_gripper/gripper_command     unix_msgs/GripperCommand    command.position 0 open .. 1 closed
  /right_gripper/gripper_command    unix_msgs/GripperCommand
  /lifting_controller/joint_states, /left_gripper/joint_states, /right_gripper/joint_states (state out)

Started by mock_sim.launch.py. The gripper topics need unix_msgs sourced before the launch.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from panther_examples.robot import LIFT_VEL, MAX_LIFT as LIFT_MAX, MIN_LIFT as LIFT_MIN

# shortcut: gripper speed when a command gives none; the real gripper speed is unmeasured, time it on the robot.
GRIPPER_VEL = 1.0  # full stroke per second
# joint: (sim controller, robot state topic)
JOINTS = {
    "body_joint": ("lifting_joint_trajectory_controller", "/lifting_controller/joint_states"),
    "joint_lf": ("left_gripper_joint_trajectory_controller", "/left_gripper/joint_states"),
    "joint_rf": ("right_gripper_joint_trajectory_controller", "/right_gripper/joint_states"),
}


def clamp(v, lo, hi):
    return min(max(v, lo), hi)


def lift_goal(cmd, now):
    """(target, speed) for a /lifting_controller/commands message, None to ignore it."""
    if cmd.position:
        v = abs(cmd.velocity[0]) if cmd.velocity and cmd.velocity[0] else LIFT_VEL
        return clamp(cmd.position[0], LIFT_MIN, LIFT_MAX), min(v, LIFT_VEL)
    if cmd.velocity:
        # shortcut: velocity mode runs to the end stop or the next command; whether the robot also stops
        # when commands stop arriving is unknown.
        v = clamp(cmd.velocity[0], -LIFT_VEL, LIFT_VEL)
        return (now if v == 0 else LIFT_MAX if v > 0 else LIFT_MIN), abs(v) or LIFT_VEL
    return None


def gripper_goal(cmd):
    if not cmd.command.position:
        return None
    v = cmd.command.velocity[0] if cmd.command.velocity and cmd.command.velocity[0] > 0 else GRIPPER_VEL
    return clamp(cmd.command.position[0], 0.0, 1.0), v


class RobotTopics(Node):
    def __init__(self):
        super().__init__("robot_topics")
        self.pos = {}
        self.traj = {j: self.create_publisher(JointTrajectory, f"/{c}/joint_trajectory", 10) for j, (c, _) in JOINTS.items()}
        self.state = {j: self.create_publisher(JointState, t, 10) for j, (_, t) in JOINTS.items()}
        self.create_subscription(JointState, "/joint_states", self.on_states, 10)
        self.create_subscription(JointState, "/lifting_controller/commands",
                                 lambda m: self.move("body_joint", lift_goal(m, self.pos.get("body_joint", 0.0))), 10)
        try:
            from unix_msgs.msg import GripperCommand
        except ImportError:
            self.get_logger().error("unix_msgs not found: gripper command topics are off. Source it before the launch.")
            return
        for side in "lr":
            self.create_subscription(GripperCommand, f"/{'left' if side == 'l' else 'right'}_gripper/gripper_command",
                                     lambda m, j=f"joint_{side}f": self.move(j, gripper_goal(m)), 10)

    def move(self, joint, goal):
        if goal is None:
            return
        target, speed = goal
        t = max(abs(target - self.pos.get(joint, target)) / speed, 0.01)
        point = JointTrajectoryPoint(positions=[target])
        point.time_from_start.sec, point.time_from_start.nanosec = int(t), int(t % 1 * 1e9)
        self.traj[joint].publish(JointTrajectory(joint_names=[joint], points=[point]))

    def on_states(self, msg):
        for i, name in enumerate(msg.name):
            if name in JOINTS:
                self.pos[name] = msg.position[i]
                self.state[name].publish(JointState(header=msg.header, name=[name], position=[msg.position[i]],
                                                    velocity=[msg.velocity[i]] if msg.velocity else []))


def main():
    rclpy.init()
    rclpy.spin(RobotTopics())


if __name__ == "__main__":
    main()
