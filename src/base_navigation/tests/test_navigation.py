"""Check the ROS velocity adapter with mocked nodes; publish no commands."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

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
