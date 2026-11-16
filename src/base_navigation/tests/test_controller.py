"""Control direction, sequencing, and loss handling without ROS or hardware."""

import math
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller import PController, ControllerHistory


class ControllerTests(unittest.TestCase):
    def test_close_target_wraps_rear_bearing_to_small_error(self):
        controller = PController(target_distance=0.5, angle_tolerance=0.001)
        for degrees in (179, -179, 180, -180, 10, -10):
            with self.subTest(degrees=degrees):
                bearing = math.radians(degrees)
                position = {'x': 0.2 * math.sin(bearing), 'y': 0.2 * math.cos(bearing)}
                command = controller.compute(position)
                expected = math.remainder(bearing, math.pi)
                self.assertAlmostEqual(command.angle_error, expected)
                self.assertEqual(command.angular_speed, 0)
                # Translating must increase separation for a too-close target.
                self.assertLess(command.linear_speed * math.cos(bearing), 0)
        # Far rear targets still require turning toward the target.
        self.assertAlmostEqual(controller.compute({'x': 0.01, 'y': -1}).angle_error,
                               math.atan2(0.01, -1))

    def test_close_angle_derivative_uses_wrapped_error_and_resets_at_goal_boundary(self):
        controller = PController(target_distance=0.5, angular_kd=0.2, angle_tolerance=0.001)
        history = ControllerHistory()
        history.compute(controller, {'x': 0.01, 'y': -0.6}, 1.0)
        near = {'x': 0.01, 'y': -0.2}
        self.assertEqual(history.compute(controller, near, 1.1), controller.compute(near))
        crossed = history.compute(controller, {'x': -0.01, 'y': -0.2}, 1.2)
        # Crossing +pi/-pi is a small change in the close-target angular error.
        self.assertLess(abs(crossed.angular_speed), 0.2)
        self.assertAlmostEqual(crossed.angle_error, math.atan(0.05))
        far = {'x': 0.01, 'y': -0.6}
        self.assertEqual(history.compute(controller, far, 1.3), controller.compute(far))

    def test_too_close_target_backs_up_without_steering(self):
        controller = PController(target_distance=0.5, angular_kd=1)
        for x in (-0.1, 0.1):
            position = {'x': x, 'y': 0.2, 'image_angle_rad': math.copysign(0.4, x)}
            command = controller.compute(position, bearing_rate=10)
            self.assertEqual(command.state, 'backing_up')
            self.assertLess(command.linear_speed, 0)
            self.assertEqual(command.angular_speed, 0)
            self.assertEqual(command.angle_error, position['image_angle_rad'])

    def test_steering_resumes_without_derivative_kick_after_backing_up(self):
        controller = PController(target_distance=0.5, angular_kd=1, linear_kd=0.1)
        history = ControllerHistory()
        for timestamp, angle in ((1.0, -0.4), (1.1, 0.2), (1.2, -0.1)):
            command = history.compute(controller, {'x': 0, 'y': 0.2, 'image_angle_rad': angle}, timestamp)
            self.assertEqual(command.angular_speed, 0)
            self.assertLess(command.linear_speed, 0)
        position = {'x': 0, 'y': 0.46, 'image_angle_rad': 0.2}
        command = history.compute(controller, position, 1.3)
        self.assertEqual(command.linear_speed, 0)
        self.assertAlmostEqual(command.angular_speed, -0.2)

    def test_reports_signed_errors_before_deadbands_and_speed_limits(self):
        controller = PController(target_distance=0.5)
        for x, y in ((0.3, 0.4), (-0.1, 0.2), (0.01, 0.52), (3, 4)):
            with self.subTest(x=x, y=y):
                command = controller.compute({'x': x, 'y': y})
                self.assertAlmostEqual(command.distance_error, math.hypot(x, y) - 0.5)
                self.assertAlmostEqual(command.angle_error, math.atan2(x, y))
        origin = controller.compute({'x': 0, 'y': 0})
        self.assertEqual(origin.distance_error, -0.5)
        self.assertIsNone(origin.angle_error)
        lost = controller.compute(None)
        self.assertIsNone(lost.distance_error)
        self.assertIsNone(lost.angle_error)

    def test_steers_and_drives_simultaneously_from_both_sides(self):
        controller = PController()
        for x, sign in ((1.0, -1), (-1.0, 1)):
            command = controller.compute({"x": x, "y": 2.0})
            self.assertEqual(command.state, "tracking")
            self.assertGreater(command.linear_speed, 0)
            self.assertLessEqual(command.linear_speed, controller.max_linear_speed)
            self.assertGreater(command.angular_speed * sign, 0)
            self.assertLessEqual(abs(command.angular_speed), controller.max_angular_speed)
        self.assertEqual(controller.compute({"x": 0, "y": 2}).state, "approaching")
        command = controller.compute({"x": 0.5, "y": 2})
        self.assertGreater(command.linear_speed, 0)
        self.assertEqual(command.state, "tracking")

    def test_approach_slows_off_center_and_does_not_drive_toward_a_rear_target(self):
        controller = PController()
        speeds = [controller.compute({'x': 2 * math.sin(angle), 'y': 2 * math.cos(angle)}).linear_speed
                  for angle in (0, 0.3, 1.0)]
        self.assertGreater(speeds[0], speeds[1])
        self.assertGreater(speeds[1], speeds[2])
        for x, y in ((1, 0), (1, -1), (0, -2)):
            command = controller.compute({'x': x, 'y': y})
            self.assertEqual(command.linear_speed, 0)
            self.assertNotEqual(command.angular_speed, 0)

    def test_corrects_bearing_at_tracking_distance_without_advancing(self):
        controller = PController()
        command = controller.compute({'x': 0.15, 'y': 0.3 * math.cos(math.pi / 6)})
        self.assertEqual(command.state, 'aligning')
        self.assertEqual(command.linear_speed, 0)
        self.assertLess(command.angular_speed, 0)

    def test_speed_is_proportional_limited_and_zero_at_goal(self):
        controller = PController(target_distance=0)
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

    def test_default_tracking_distance_is_thirty_centimeters(self):
        controller = PController()
        self.assertEqual(controller.target_distance, 0.3)
        for distance in (0.25, 0.3, 0.34, 0.35):
            command = controller.compute({"x": 0, "y": distance})
            self.assertEqual(command.state, 'reached')
            self.assertEqual((command.linear_speed, command.angular_speed), (0, 0))
        self.assertAlmostEqual(controller.compute({"x": 0, "y": 0.5}).linear_speed, 0.08)
        for distance in (0.0, 0.2):
            command = controller.compute({'x': 0, 'y': distance})
            self.assertLess(command.linear_speed, 0)
            self.assertEqual(command.state, 'backing_up')

    def test_signed_distance_tolerance_and_reverse_speed_limit(self):
        controller = PController(target_distance=0.5)
        for distance in (0.45, 0.5, 0.55):
            command = controller.compute({'x': 0, 'y': distance}, distance_rate=10)
            self.assertEqual(command.state, 'reached')
            self.assertEqual(command.linear_speed, 0)
        self.assertLess(controller.compute({'x': 0, 'y': 0.449}).linear_speed, 0)
        self.assertGreater(controller.compute({'x': 0, 'y': 0.551}).linear_speed, 0)
        fast = PController(target_distance=2)
        self.assertEqual(fast.compute({'x': 0, 'y': 0.1}).linear_speed, -fast.max_linear_speed)
        for y in (-0.2, -0.02):
            command = controller.compute({'x': 0, 'y': y})
            self.assertGreater(command.linear_speed, 0)
            self.assertNotEqual(command.state, 'reached')

    def test_derivative_brakes_reverse_motion_without_pushing_toward_close_target(self):
        controller = PController(target_distance=0.5, linear_kd=0.2)
        position = {'x': 0, 'y': 0.3}
        reversing = controller.compute(position)
        damped = controller.compute(position, distance_rate=0.2)
        self.assertLess(reversing.linear_speed, damped.linear_speed)
        self.assertLess(damped.linear_speed, 0)
        self.assertEqual(controller.compute(position, distance_rate=10).linear_speed, 0)

    def test_closed_loop_recovers_from_too_close_with_signed_motion(self):
        for kd in (0.0, 0.2):
            for initial in ((0, 0), (0.1, 0.15), (-0.1, 0.15), (0.1, -0.15), (0, -0.02)):
                with self.subTest(kd=kd, initial=initial):
                    controller = PController(linear_kd=kd)
                    history = ControllerHistory()
                    x, y = initial
                    for step in range(2000):
                        command = history.compute(controller, {'x': x, 'y': y}, step * 0.05)
                        if command.state == 'reached':
                            break
                        yaw = command.angular_speed * 0.05
                        x, y = (math.cos(yaw) * x + math.sin(yaw) * y,
                                -math.sin(yaw) * x + math.cos(yaw) * y - command.linear_speed * 0.05)
                    else:
                        self.fail('Failed to recover from a too-close target')
                    self.assertLessEqual(abs(math.hypot(x, y) - controller.target_distance),
                                         controller.distance_tolerance)

    def test_target_loss_or_invalid_measurement_stops(self):
        positions = [None, {}, {"x": 0}, {"x": None, "y": 1},
                     {"x": float("nan"), "y": 1}, {"x": 0, "y": float("inf")},
                     {"x": 0, "y": 1, "image_angle_rad": float('nan')}]
        for position in positions:
            with self.subTest(position=position):
                controller = PController()
                self.assertGreater(controller.compute({"x": 0, "y": 2}).linear_speed, 0)
                command = controller.compute(position)
                self.assertEqual(command.state, "lost")
                self.assertEqual((command.linear_speed, command.angular_speed), (0, 0))

    def test_invalid_configuration(self):
        options = [dict(linear_kp=-1), dict(max_linear_speed=0),
                   dict(angular_kd=-1), dict(linear_kd=float('nan')),
                   dict(angular_kp=float("nan")), dict(target_distance=-1),
                   dict(angle_tolerance=math.pi), dict(distance_tolerance=float("inf"))]
        for values in options:
            with self.subTest(values=values), self.assertRaises(ValueError):
                PController(**values)

    def test_derivative_brakes_approach_and_turn_without_reverse_translation(self):
        controller = PController(linear_kd=0.5, angular_kd=0.2)
        position = {'x': 0.1, 'y': 0.5}
        proportional = controller.compute(position)
        damped = controller.compute(position, distance_rate=-0.1, bearing_rate=-0.3)
        self.assertLess(damped.linear_speed, proportional.linear_speed)
        self.assertLess(abs(damped.angular_speed), abs(proportional.angular_speed))
        self.assertEqual(controller.compute(position, distance_rate=-10).linear_speed, 0)
        self.assertEqual(controller.compute({'x': 0, 'y': 0.5}, distance_rate=-10).state, 'braking')

    def test_history_filters_new_observations_and_holds_repeated_frames(self):
        controller = PController(linear_kd=0.2)
        history = ControllerHistory()
        history.compute(controller, {'x': 0, 'y': 0.7}, 1.0)
        position = {'x': 0, 'y': 0.6}
        command = history.compute(controller, position, 1.1)
        # Raw rate -1 m/s, alpha=0.1/(0.15+0.1)=0.4: filtered rate -0.4.
        self.assertAlmostEqual(command.linear_speed, 0.4 * 0.3 - 0.2 * 0.4)
        for _ in range(5):
            self.assertEqual(history.compute(controller, position, 1.1), command)

    def test_history_resets_after_loss_gaps_time_reversal_and_gain_changes(self):
        controller = PController(linear_kd=0.2)
        position = {'x': 0, 'y': 0.6}
        for event in ('missing', 'invalid', 'gap', 'backward', 'settings'):
            with self.subTest(event=event):
                history = ControllerHistory()
                history.compute(controller, {'x': 0, 'y': 0.7}, 1.0)
                history.compute(controller, position, 1.1)
                timestamp = 1.2
                current = controller
                if event == 'missing':
                    history.compute(controller, None, 1.15)
                elif event == 'invalid':
                    history.compute(controller, {'x': float('nan'), 'y': 1}, 1.15)
                elif event == 'gap':
                    timestamp = 2.0
                elif event == 'backward':
                    timestamp = 0.9
                else:
                    current = PController(linear_kd=0.3)
                self.assertEqual(history.compute(current, position, timestamp), current.compute(position))

    def test_bearing_derivative_wraps_at_pi(self):
        controller = PController(angular_kp=0.01, angular_kd=0.1, angle_tolerance=0.01)
        history = ControllerHistory()
        history.compute(controller, {'x': 0.01, 'y': -1}, 1.0)
        command = history.compute(controller, {'x': -0.01, 'y': -1}, 1.1)
        self.assertLess(abs(command.angular_speed), 0.05)

    def test_derivative_reduces_overshoot_with_actuator_lag(self):
        overshoots = []
        for kd in (0.0, 0.5):
            controller = PController(angular_kp=2, angular_kd=kd,
                                     max_angular_speed=2, angle_tolerance=0.001)
            history = ControllerHistory()
            bearing, angular_speed, minimum_bearing = 0.5, 0.0, 0.5
            for step in range(1000):
                command = history.compute(controller, {
                    'x': 0.3 * math.sin(bearing), 'y': 0.3 * math.cos(bearing),
                }, step * 0.02)
                # Simulate a velocity actuator with a 0.5 s response time.
                angular_speed += (0.02 / 0.5) * (command.angular_speed - angular_speed)
                bearing += angular_speed * 0.02
                minimum_bearing = min(minimum_bearing, bearing)
            overshoots.append(-minimum_bearing)
            self.assertLess(abs(bearing), 0.001)
        self.assertLess(overshoots[1], overshoots[0] / 2)

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
            self.assertLessEqual(abs(math.hypot(x, y) - controller.target_distance),
                                 controller.distance_tolerance)
