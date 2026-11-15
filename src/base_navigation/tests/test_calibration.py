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
