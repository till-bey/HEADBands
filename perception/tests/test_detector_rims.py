"""Check rim detection and RGB measurements on known synthetic discs."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from detector import DetectorSettings, detect


class RimDetectorTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((240, 320, 3), (20, 70, 25), dtype=np.uint8)
        self.settings = DetectorSettings(min_radius=5, max_radius=90,
                                         min_ellipse_ratio=0.1, min_area=50)

    def test_colored_disc_rim_does_not_require_white_interior(self):
        cv2.ellipse(self.frame, (160, 120), (48, 22), 25, 0, 360, (40, 80, 230), -1)
        candidates, debug = detect(self.frame, self.settings)
        self.assertTrue(candidates)
        self.assertEqual(cv2.countNonZero(debug['white_mask']), 0)
        self.assertGreater(cv2.countNonZero(debug['edge_mask']), 0)
        self.assertEqual(candidates[0].method, 'Rim ellipse')
        self.assertLess(np.linalg.norm(np.array(candidates[0].center) - (160, 120)), 3)

    def test_thin_side_view_is_an_ellipse(self):
        cv2.ellipse(self.frame, (160, 120), (50, 9), 10, 0, 360, (40, 80, 230), -1)
        candidates, _ = detect(self.frame, self.settings)
        self.assertEqual(len(candidates), 1)
        self.assertLess(min(candidates[0].axes) / max(candidates[0].axes), 0.3)

    def test_rgb_mean_includes_dark_pixels_inside_inner_seventy_five_percent_radii(self):
        cv2.ellipse(self.frame, (160, 120), (48, 25), 30, 0, 360, (240, 240, 240), -1)
        cv2.ellipse(self.frame, (160, 120), (32, 15), 30, 0, 360, (20, 20, 20), -1)
        candidates, _ = detect(self.frame, self.settings)
        self.assertTrue(candidates)
        candidate = max(candidates, key=lambda c: c.axes[0] * c.axes[1])
        mask = np.zeros(self.frame.shape[:2], np.uint8)
        cv2.ellipse(mask, tuple(v * 4 for v in candidate.center), tuple(v * 3 for v in candidate.axes),
                    candidate.angle, 0, 360, 255, -1, shift=2)
        expected = self.frame[mask != 0].astype(np.uint16).sum(axis=1).mean()
        self.assertAlmostEqual(candidate.mean_rgb_sum, expected, places=8)
        self.assertLess(candidate.mean_rgb_sum, 600)  # White-only would be 720.

    def test_bright_outer_rim_is_excluded_from_rgb_average(self):
        cv2.ellipse(self.frame, (160, 120), (48, 25), 30, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(self.frame, (160, 120), (40, 21), 30, 0, 360, (30, 50, 70), -1)
        candidates, _ = detect(self.frame, self.settings)
        self.assertTrue(candidates)
        candidate = max(candidates, key=lambda c: c.axes[0] * c.axes[1])
        self.assertAlmostEqual(candidate.mean_rgb_sum, 150.0)

    def test_bright_pixels_do_not_overflow_and_rim_is_not_duplicated(self):
        cv2.ellipse(self.frame, (160, 120), (48, 25), 0, 0, 360, (255, 255, 255), -1)
        candidates, _ = detect(self.frame, self.settings)
        self.assertEqual(len(candidates), 1)
        self.assertGreater(candidates[0].mean_rgb_sum, 700)
        self.assertLessEqual(candidates[0].mean_rgb_sum, 765)

    def test_rectangle_and_empty_frame_are_not_discs(self):
        self.assertEqual(detect(self.frame, self.settings)[0], [])
        cv2.rectangle(self.frame, (100, 90), (220, 150), (240, 240, 240), -1)
        self.assertEqual(detect(self.frame, self.settings)[0], [])
