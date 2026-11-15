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

from panther_examples.robot import (
    FORWARD, HEAD_FORWARD, LEFT_GRIPPER_CMD_TOPIC, LEFT_TRAJ, RIGHT_GRIPPER_CMD_TOPIC, RIGHT_TRAJ,
)


class PantherCommands(Node):
    def __init__(self):
        super().__init__("panther_commands")
        self.left_gripper = self.create_publisher(
            GripperCommand, LEFT_GRIPPER_CMD_TOPIC, 10)
        self.right_gripper = self.create_publisher(
            GripperCommand, RIGHT_GRIPPER_CMD_TOPIC, 10)
        self.base = self.create_publisher(Twist, "/cmd_vel", 10)
        self.left_arm = self.create_publisher(
            JointTrajectory,
            f"/{LEFT_TRAJ}/joint_trajectory",
            10)
        self.right_arm = self.create_publisher(
            JointTrajectory,
            f"/{RIGHT_TRAJ}/joint_trajectory",
            10)
        self.head = self.create_publisher(
            Float64MultiArray, f"/{HEAD_FORWARD}/commands", 10)
        self.dual_arm = self.create_publisher(
            Float64MultiArray, f"/{FORWARD}/commands", 10)

    def _send(self, publisher, message):
        deadline = time.monotonic() + 1.0
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        publisher.publish(message)
        rclpy.spin_once(self, timeout_sec=0.05)

    def gripper(self, side, position, velocity=0.1, effort=0.5):
        message = GripperCommand()
        message.command.name = [""]
        message.command.position = [float(position)]
        message.command.velocity = [float(velocity)]
        message.command.effort = [float(effort)]
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

    # NOTE: This controller is inactive on boot
    # def move_arm(self, side, positions, seconds=6):
    #     """Point-to-point. Those controllers start inactive; switch them on the robot first."""
    #     if side == "left":
    #         publisher, names = self.left_arm, robot.LEFT_ARM
    #     else:
    #         publisher, names = self.right_arm, robot.RIGHT_ARM
    #     point = JointTrajectoryPoint()
    #     point.positions = [float(value) for value in positions]
    #     point.time_from_start.sec = int(seconds)
    #     message = JointTrajectory()
    #     message.joint_names = list(names)
    #     message.points = [point]
    #     self._send(publisher, message)

    def move_dual_arm(self, positions, seconds=2.0):
        """Hold 16 positions: joint_la0..la7, then joint_ra0..ra7. No time scaling."""
        if len(positions) != 16:
            raise ValueError("dual arm command needs 16 positions")
        message = Float64MultiArray()
        message.data = [float(value) for value in positions]
        self._stream(self.dual_arm, message, seconds)

    def move_head(self, joint_head0, joint_head1, seconds=2.0):
        """Hold a head position. joint_head0 is pitch, joint_head1 is yaw."""
        message = Float64MultiArray()
        message.data = [float(joint_head0), float(joint_head1)]
        self._stream(self.head, message, seconds)


def main():
    rclpy.init()
    node = PantherCommands()

    # node.gripper("right", position=0.2, velocity=0.5)
    # time.sleep(1)
    # node.gripper("right", position=0.78, velocity=0.5)
    # node.base_teleop(linear_y=0.2, seconds=1.0)
    # node.base_teleop()
    # node.move_dual_arm([...16 positions...])  # read the pose first: this holds exactly what you send
    # node.move_head(0.0, 1.0)

    node.get_logger().info("publishers ready")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
