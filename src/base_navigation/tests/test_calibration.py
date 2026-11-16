"""Verify robot feedback, mounting offsets, and ground projection agree."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import camera_pose
from dashboard import inspect_frame
from controller import PController
from camera import pixel_to_ground_robot
import numpy as np


class CalibrationTests(unittest.TestCase):
    def test_horizontal_image_angle_stays_zero_on_center_line_as_pose_changes(self):
        for tilt in (0.5, 1.0, math.pi / 2, 2.0):
            for py in (180, 320, 460):
                with self.subTest(tilt=tilt, py=py):
                    position = pixel_to_ground_robot(240, py, 480, 640, 0, tilt, 90, 60)
                    command = PController().compute(position)
                    self.assertEqual(position['image_angle_rad'], 0)
                    self.assertEqual(command.angle_error, 0)
                    self.assertEqual(command.angular_speed, 0)

    def test_fixed_crosshair_keeps_same_steering_error_across_projection_sign_change(self):
        controller = PController(target_distance=0, angular_kd=0.2, angle_tolerance=0.001)
        from controller import ControllerHistory
        history = ControllerHistory()
        errors, commands = [], []
        for i, tilt in enumerate((1.0, 1.5, 1.6, 2.0, 1.5)):
            position = pixel_to_ground_robot(250, 320, 480, 640, 0, tilt, 90, 60)
            command = history.compute(controller, position, 1.0 + i * 0.1)
            errors.append(command.angle_error)
            commands.append(command.angular_speed)
        for error in errors:
            self.assertAlmostEqual(error, math.atan(10 / 240))
        for command in commands:
            self.assertAlmostEqual(command, commands[0])

    def setUp(self):
        self.settings = SimpleNamespace(camera_base_height=1.0, robot_z=0.1, tilt_rad=0.2,
                                        horizontal_fov=90, vertical_fov=60)
        self.feedback = dict(lift_fresh=True, lift_position=0.2, head_fresh=True, head_pitch=0.3)

    def test_live_robot_values_add_to_camera_offsets(self):
        pose = camera_pose(self.settings, self.feedback)
        self.assertAlmostEqual(pose['height'], 1.3)
        self.assertAlmostEqual(pose['tilt'], 0.5)
        self.assertTrue(pose['lift_fresh'] and pose['head_fresh'])

    def test_stale_or_invalid_values_do_not_enter_projection(self):
        for feedback in ({}, dict(self.feedback, lift_fresh=False, head_fresh=False),
                         dict(self.feedback, lift_position=float('nan'), head_pitch=float('inf'))):
            pose = camera_pose(self.settings, feedback)
            self.assertAlmostEqual(pose['height'], 1.1)
            self.assertAlmostEqual(pose['tilt'], 0.2)
            self.assertFalse(pose['lift_fresh'] or pose['head_fresh'])

    def test_detector_projection_uses_the_live_calculated_pose(self):
        camera = Mock()
        camera.detect_center.return_value = (240, 320)
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        _, position, _, _ = inspect_frame(camera, frame, self.settings, PController(), feedback=self.feedback)
        self.assertAlmostEqual(position['y'], 1.3 / math.tan(0.5))
        self.assertAlmostEqual(position['x'], 0)
        manual = pixel_to_ground_robot(240, 320, 480, 640, 0.3, 0.5, 90, 60,
                                       camera_base_height_m=1.0)
        self.assertAlmostEqual(position['distance'], manual['distance'])
