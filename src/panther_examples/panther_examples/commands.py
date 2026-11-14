"""Command publishers for the Panther.

Gripper position: 0.0 open, 1.0 closed.
The drive controller listens on /cmd_vel. teleop_twist_keyboard publishes that
same topic; /cmd_vel_teleop is only the vendor teleop input.
"""

import subprocess
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from unix_msgs.msg import GripperCommand

LEFT_ARM = [
    "joint_la0", "joint_la1", "joint_la2", "joint_la3",
    "joint_la4", "joint_la5", "joint_la6", "joint_la7",
]
RIGHT_ARM = [
    "joint_ra0", "joint_ra1", "joint_ra2", "joint_ra3",
    "joint_ra4", "joint_ra5", "joint_ra6", "joint_ra7",
]


class PantherCommands(Node):
    def __init__(self):
        super().__init__("panther_commands")
        self.left_gripper = self.create_publisher(
            GripperCommand, "/left_gripper/gripper_command", 10)
        self.right_gripper = self.create_publisher(
            GripperCommand, "/right_gripper/gripper_command", 10)
        self.base = self.create_publisher(Twist, "/cmd_vel", 10)
        self.left_arm = self.create_publisher(
            JointTrajectory,
            "/left_arm_joint_trajectory_position_controller/joint_trajectory",
            10)
        self.right_arm = self.create_publisher(
            JointTrajectory,
            "/right_arm_joint_trajectory_position_controller/joint_trajectory",
            10)
        self.head = self.create_publisher(
            Float64MultiArray, "/head_forward_position_controller/commands", 10)

    def _send(self, publisher, message):
        deadline = time.monotonic() + 1.0
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        publisher.publish(message)
        rclpy.spin_once(self, timeout_sec=0.05)

    def gripper(self, side, position, velocity=0.1):
        message = GripperCommand()
        message.command.name = [""]
        message.command.position = [float(position)]
        message.command.velocity = [float(velocity)]
        publisher = self.left_gripper if side == "left" else self.right_gripper
        self._send(publisher, message)

    def _stream(self, publisher, message, seconds):
        end = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end:
            publisher.publish(message)
            rclpy.spin_once(self, timeout_sec=0.1)

    def base_teleop(self, linear_x=0.0, linear_y=0.0, angular_z=0.0, seconds=0.0):
        """Publish /cmd_vel at 10 Hz for `seconds`, then publish zeros."""
        message = Twist()
        message.linear.x = float(linear_x)
        message.linear.y = float(linear_y)
        message.angular.z = float(angular_z)
        if seconds <= 0.0:
            self._send(self.base, message)
            return
        self._stream(self.base, message, seconds)
        if linear_x or linear_y or angular_z:
            self._stream(self.base, Twist(), 0.5)

    def move_arm(self, side, positions, seconds=6):
        """Point-to-point. Those controllers start inactive; switch them on the robot first."""
        if side == "left":
            publisher, names = self.left_arm, LEFT_ARM
        else:
            publisher, names = self.right_arm, RIGHT_ARM
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        point.time_from_start.sec = int(seconds)
        message = JointTrajectory()
        message.joint_names = list(names)
        message.points = [point]
        self._send(publisher, message)

    def move_head(self, joint_head0, joint_head1, seconds=2.0):
        """Hold a head position. joint_head0 is yaw, joint_head1 is pitch."""
        message = Float64MultiArray()
        message.data = [float(joint_head0), float(joint_head1)]
        self._stream(self.head, message, seconds)


def main():
    rclpy.init()
    node = PantherCommands()

    # node.gripper("left", position=1.0, velocity=0.1)
    # node.gripper("left", position=0.0, velocity=0.1)
    node.base_teleop(angular_z=-0.5, seconds=0.5)
    node.base_teleop()
    # node.move_arm("left", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    node.move_head(0.5, 0.2)

    node.get_logger().info("publishers ready")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
