"""Check restored white-mask detection and diagnostic RGB measurements."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from detector import Candidate, DetectorSettings, detect, draw_detections


class WhiteMaskDetectorTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((240, 320, 3), (20, 70, 25), dtype=np.uint8)
        self.settings = DetectorSettings(min_radius=5, max_radius=90,
                                         min_ellipse_ratio=0.1, min_area=50)

    def test_nonwhite_elliptical_edges_do_not_bypass_white_thresholds(self):
        cv2.ellipse(self.frame, (160, 120), (48, 22), 25, 0, 360, (40, 80, 230), -1)
        candidates, debug = detect(self.frame, self.settings)
        self.assertEqual(candidates, [])
        self.assertEqual(cv2.countNonZero(debug['white_mask']), 0)
        self.assertEqual(set(debug), {'threshold_mask', 'white_mask'})

    def test_white_detection_does_not_call_canny(self):
        cv2.ellipse(self.frame, (160, 120), (48, 22), 25, 0, 360, (240, 240, 240), -1)
        with patch.object(cv2, 'Canny', side_effect=AssertionError('Canny must not be used')):
            candidates, _ = detect(self.frame, self.settings)
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].method, 'Ellipse')

    def test_thin_white_ellipse_uses_configured_aspect_ratio(self):
        cv2.ellipse(self.frame, (160, 120), (50, 9), 10, 0, 360, (240, 240, 240), -1)
        candidates, _ = detect(self.frame, self.settings)
        self.assertEqual(len(candidates), 1)
        self.assertLess(min(candidates[0].axes) / max(candidates[0].axes), 0.3)

    def test_rgb_statistics_include_dark_pixels_inside_inner_sixty_five_percent_radii(self):
        cv2.ellipse(self.frame, (160, 120), (48, 25), 30, 0, 360, (240, 240, 240), -1)
        cv2.ellipse(self.frame, (160, 120), (32, 15), 30, 0, 360, (20, 20, 20), -1)
        candidates, _ = detect(self.frame, self.settings)
        self.assertTrue(candidates)
        candidate = max(candidates, key=lambda c: c.axes[0] * c.axes[1])
        mask = np.zeros(self.frame.shape[:2], np.uint8)
        cv2.ellipse(mask, tuple(v * 4 for v in candidate.center),
                    tuple(round(v * 0.65 * 4) for v in candidate.axes),
                    candidate.angle, 0, 360, 255, -1, shift=2)
        rgb_sums = self.frame[mask != 0].astype(np.float64).sum(axis=1)
        self.assertAlmostEqual(candidate.mean_rgb_sum, rgb_sums.mean(), places=8)
        self.assertAlmostEqual(candidate.std_rgb_sum, rgb_sums.std(), places=8)
        self.assertGreater(candidate.std_rgb_sum, 0)
        self.assertLess(candidate.mean_rgb_sum, 600)  # White-only would be 720.

    def test_bright_outer_rim_is_excluded_from_rgb_statistics(self):
        cv2.ellipse(self.frame, (160, 120), (48, 25), 30, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(self.frame, (160, 120), (35, 19), 30, 0, 360, (200, 210, 220), -1)
        candidates, _ = detect(self.frame, self.settings)
        self.assertTrue(candidates)
        candidate = max(candidates, key=lambda c: c.axes[0] * c.axes[1])
        self.assertAlmostEqual(candidate.mean_rgb_sum, 630.0)
        self.assertAlmostEqual(candidate.std_rgb_sum, 0.0)

    def test_bright_pixels_do_not_overflow(self):
        cv2.ellipse(self.frame, (160, 120), (48, 25), 0, 0, 360, (255, 255, 255), -1)
        candidates, _ = detect(self.frame, self.settings)
        self.assertEqual(len(candidates), 1)
        self.assertGreater(candidates[0].mean_rgb_sum, 700)
        self.assertLessEqual(candidates[0].mean_rgb_sum, 765)

    def test_empty_frame_has_no_detections(self):
        self.assertEqual(detect(self.frame, self.settings)[0], [])

    def test_side_threshold_uses_unrounded_standard_deviation(self):
        for std, expected_side in [(0.0, 0), (29.999, 0), (30.0, 1), (30.001, 1), (100.0, 1)]:
            with self.subTest(std=std):
                candidate = Candidate((160, 120), (48, 25), 0, 'Ellipse', 0.9,
                                      mean_rgb_sum=600, std_rgb_sum=std)
                self.assertEqual(candidate.side, expected_side)
                with patch.object(cv2, 'putText', wraps=cv2.putText) as put_text:
                    draw_detections(self.frame, [candidate], self.settings)
                self.assertIn(f'Side {expected_side}', put_text.call_args.args[1])
