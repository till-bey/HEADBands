"""Control direction, sequencing, and loss handling without ROS or hardware."""

import math
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller import PController


class ControllerTests(unittest.TestCase):
    def test_turns_before_driving_and_realigns(self):
        controller = PController()
        for x, sign in ((1.0, -1), (-1.0, 1)):
            command = controller.compute({"x": x, "y": 2.0})
            self.assertEqual(command.state, "aligning")
            self.assertEqual(command.linear_speed, 0)
            self.assertGreater(command.angular_speed * sign, 0)
            self.assertLessEqual(abs(command.angular_speed), controller.max_angular_speed)
        self.assertEqual(controller.compute({"x": 0, "y": 2}).state, "approaching")
        command = controller.compute({"x": 0.5, "y": 2})
        self.assertEqual(command.linear_speed, 0)
        self.assertEqual(command.state, "aligning")

    def test_speed_is_proportional_limited_and_zero_at_goal(self):
        controller = PController()
        self.assertAlmostEqual(controller.compute({"x": 0, "y": 0.2}).linear_speed, 0.08)
        self.assertEqual(controller.compute({"x": 0, "y": 10}).linear_speed, 0.15)
        for x, y in ((0, 0), (0.03, 0.02), (0, 0.05)):
            command = controller.compute({"x": x, "y": y})
            self.assertEqual(command.state, "reached")
            self.assertEqual((command.linear_speed, command.angular_speed), (0, 0))
        stand_off = PController(target_distance=0.5)
        self.assertEqual(stand_off.compute({"x": 0, "y": 0.5}).state, "reached")
        self.assertAlmostEqual(stand_off.compute({"x": 0, "y": 0.7}).linear_speed, 0.08)
        self.assertEqual(controller.compute({"x": 0, "y": -2}).linear_speed, 0)

    def test_target_loss_or_invalid_measurement_stops(self):
        positions = [None, {}, {"x": 0}, {"x": None, "y": 1},
                     {"x": float("nan"), "y": 1}, {"x": 0, "y": float("inf")}]
        for position in positions:
            with self.subTest(position=position):
                controller = PController()
                self.assertGreater(controller.compute({"x": 0, "y": 2}).linear_speed, 0)
                command = controller.compute(position)
                self.assertEqual(command.state, "lost")
                self.assertEqual((command.linear_speed, command.angular_speed), (0, 0))

    def test_invalid_configuration(self):
        options = [dict(linear_kp=-1), dict(max_linear_speed=0),
                   dict(angular_kp=float("nan")), dict(target_distance=-1),
                   dict(angle_tolerance=math.pi), dict(distance_tolerance=float("inf"))]
        for values in options:
            with self.subTest(values=values), self.assertRaises(ValueError):
                PController(**values)

    def test_closed_loop_converges_from_both_sides(self):
        # Transform a static ground target into the robot frame after each command.
        for target_x in (-1.0, 1.0):
            x, y = target_x, 2.0
            controller = PController()
            for _ in range(3000):
                command = controller.compute({"x": x, "y": y})
                if command.state == "reached":
                    break
                dt = 0.05
                yaw = command.angular_speed * dt
                x, y = (math.cos(yaw) * x + math.sin(yaw) * y,
                        -math.sin(yaw) * x + math.cos(yaw) * y - command.linear_speed * dt)
            else:
                self.fail("Controller failed to reach the target")
            self.assertLessEqual(math.hypot(x, y), controller.distance_tolerance)
