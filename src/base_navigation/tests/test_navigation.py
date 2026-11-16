"""Check the ROS velocity adapter with mocked nodes; publish no commands."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import navigation
except ImportError as exc:
    raise unittest.SkipTest(f"ROS environment required: {exc}")


class NavigationTests(unittest.TestCase):
    def test_velocities_and_disconnected_stop(self):
        node = Mock()
        with patch.object(navigation, "BaseController", return_value=node), \
             patch.object(navigation, "rclpy"), \
             patch.object(navigation.time, "sleep"):
            with navigation.RobotNavigation() as robot:
                node.publisher.get_subscription_count.return_value = 1
                self.assertTrue(robot.set_velocity(0.1, -0.2))
                node.move_base.assert_called_once_with(0.1, -0.2, duration=0.15)
                self.assertTrue(robot.set_velocity(0, 0))
                node.stop.assert_called_once()
                node.publisher.get_subscription_count.return_value = 0
                self.assertFalse(robot.set_velocity(0.1, 0))
                self.assertEqual(node.move_base.call_count, 1)
                self.assertEqual(node.stop.call_count, 2)

    def test_exception_stops_and_cleans_up(self):
        node = Mock()
        with patch.object(navigation, "BaseController", return_value=node), \
             patch.object(navigation, "rclpy") as ros, \
             patch.object(navigation.time, "sleep"):
            with self.assertRaises(RuntimeError):
                with navigation.RobotNavigation():
                    raise RuntimeError("application failed")
            node._timer.cancel.assert_called_once()
            node.stop.assert_called_once()
            node.destroy_node.assert_called_once()
            ros.shutdown.assert_called_once()

    def test_wait_processes_ros_for_control_period(self):
        with patch.object(navigation, "BaseController"), \
             patch.object(navigation, "rclpy") as ros, \
             patch.object(navigation.time, "sleep"), \
             patch.object(navigation.time, "monotonic", side_effect=[100, 100.01, 100.06]):
            with navigation.RobotNavigation() as robot:
                robot.wait(0.05)
                ros.spin_once.assert_called_once()
                self.assertAlmostEqual(ros.spin_once.call_args.kwargs['timeout_sec'], 0.04)


class HeadControlTests(unittest.TestCase):
    def setUp(self):
        # No node construction or DDS: exercise the real command logic with a
        # fake publisher and steady clock.
        self.node = navigation.BaseController.__new__(navigation.BaseController)
        self.node.head_publisher = Mock()
        self.node.head_publisher.get_subscription_count.return_value = 1
        self.now = 0
        self.node._clock = Mock()
        self.node._clock.now.side_effect = lambda: SimpleNamespace(nanoseconds=self.now)
        self.node._head_positions = self.node._head_feedback_at = None
        self.node._head_target = self.node._head_command = self.node._head_tick_at = None
        self.node._head_deadline = None
        self.ros = patch.object(navigation.rclpy, 'ok', return_value=True)
        self.ros.start()
        self.addCleanup(self.ros.stop)
        self.feedback(0.0, 0.3)

    def feedback(self, pitch, yaw):
        # Feedback order need not match the command array's pitch/yaw order.
        self.node._on_head_state(SimpleNamespace(
            name=['joint_head1', 'joint_head0'], position=[pitch, yaw]))

    def test_smooth_up_and_down_preserve_yaw_and_stop_at_measured_pose(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                self.feedback(0.0, 0.3)
                self.node.move_head(sign * math.radians(5))
                self.now += int(0.05e9)
                self.node._tick_head()
                sent = self.node.head_publisher.publish.call_args.args[0].data
                self.assertAlmostEqual(sent[0], 0.3)  # Left/right stays fixed.
                self.assertAlmostEqual(sent[1], sign * 0.01)  # Up/down changes.
                self.feedback(sign * 0.005, 0.3)
                self.node.stop_head()
                held = self.node.head_publisher.publish.call_args.args[0].data
                self.assertAlmostEqual(held[0], 0.3)
                self.assertAlmostEqual(held[1], sign * 0.005)
                self.assertFalse(self.node.head_status['moving'])

    def test_limits_missing_controller_and_stale_or_invalid_feedback(self):
        self.feedback(1.4, 0.3)
        with self.assertRaises(ValueError):
            self.node.move_head(math.radians(5))
        self.feedback(-0.25, 0.3)
        with self.assertRaises(ValueError):
            self.node.move_head(-math.radians(5))
        self.feedback(0, 0.3)
        self.node.head_publisher.get_subscription_count.return_value = 0
        with self.assertRaises(RuntimeError):
            self.node.move_head(0.05)
        self.node.head_publisher.get_subscription_count.return_value = 1
        self.node.move_head(0.05)
        self.now += int(0.05e9)
        self.node._tick_head()
        self.now += int(1.1e9)
        self.node._tick_head()
        self.assertFalse(self.node.head_status['moving'])
        self.assertAlmostEqual(self.node.head_publisher.publish.call_args.args[0].data[1], 0.01)
        with self.assertRaises(RuntimeError):
            self.node.move_head(0.05)
        self.feedback(float('nan'), 0.3)
        self.assertFalse(self.node.head_status['fresh'])
