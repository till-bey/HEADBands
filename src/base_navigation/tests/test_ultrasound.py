"""Verify range validity, freshness, and direction mapping without ROS."""

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultrasound import UltrasoundReadings, suggested_side


def sample(distance, *, radiation=0, frame='ultrasound_front', minimum=0.05, maximum=4.0):
    return SimpleNamespace(range=distance, min_range=minimum, max_range=maximum,
                           radiation_type=radiation, header=SimpleNamespace(frame_id=frame))


class UltrasoundTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.readings = UltrasoundReadings(clock=lambda: self.now)

    def test_distance_is_in_meters_and_infrared_is_excluded(self):
        self.readings.update('/sonar/front', sample(0.23))
        self.readings.update('/infrared/rear', sample(0.1, radiation=1))
        result = self.readings.snapshot()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['distance_m'], 0.23)
        self.assertEqual(result[0]['side'], 'front')
        self.assertTrue(result[0]['fresh'])

    def test_infinite_and_invalid_ranges_are_json_safe(self):
        for distance, expected in ((-math.inf, 'too_close'), (math.inf, 'beyond_range'),
                                   (math.nan, 'invalid'), (-1, 'invalid'), (9, 'invalid')):
            with self.subTest(distance=distance):
                self.readings.update('/sonar/front', sample(distance))
                result = self.readings.snapshot()
                self.assertEqual(result[0]['status'], expected)
                json.dumps(result, allow_nan=False)

    def test_old_values_expire_and_new_values_restore_sensor(self):
        self.readings.update('/sonar/front', sample(0.2))
        self.now = 1.1
        stale = self.readings.snapshot()[0]
        self.assertEqual(stale['status'], 'stale')
        self.assertIsNone(stale['distance_m'])
        self.readings.update('/sonar/front', sample(0.8))
        self.assertTrue(self.readings.snapshot()[0]['fresh'])

    def test_ambiguous_sensor_placement_is_not_guessed(self):
        self.assertEqual(suggested_side('/ultrasound/back', 'sonar_rear'), 'rear')
        self.assertEqual(suggested_side('/ultrasound/range3', 'sonar_left'), 'left')
        self.assertIsNone(suggested_side('/ultrasound/front_left', 'sensor0'))
        self.assertIsNone(suggested_side('/ultrasound/range3', 'sensor3'))
