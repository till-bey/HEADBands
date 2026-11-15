"""Exercise the potential field and sensor coverage without ROS or hardware."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avoidance import PotentialFieldController, sensor_distances, validate_mapping
from controller import PController


def readings(**distances):
    return [dict(topic='/sonar/' + side, side=side, distance_m=distances.get(side, 2.0),
                 fresh=True, status='valid') for side in ('front', 'right', 'rear', 'left')]


class AvoidanceTests(unittest.TestCase):
    def setUp(self):
        self.base = PController()
        self.target = {'x': 0, 'y': 2}

    def command(self, data, target=None, mapping=None):
        return PotentialFieldController(self.base, data, mapping or {}).compute(
            self.target if target is None else target)

    def test_clear_path_keeps_normal_tracking(self):
        self.assertEqual(self.command(readings()), self.base.compute(self.target))

    def test_side_obstacles_steer_away_and_keep_speeds_bounded(self):
        for side, turn_sign in [('right', 1), ('left', -1)]:
            command = self.command(readings(**{side: 0.35}))
            self.assertGreater(command.angular_speed * turn_sign, 0)
            self.assertGreater(command.linear_speed, 0)
            self.assertLessEqual(command.linear_speed, self.base.max_linear_speed)
            self.assertLessEqual(abs(command.angular_speed), self.base.max_angular_speed)

    def test_front_obstacle_slows_and_local_minimum_stops(self):
        slowed = self.command(readings(front=0.5))
        self.assertGreater(slowed.linear_speed, 0)
        self.assertLess(slowed.linear_speed, self.base.compute(self.target).linear_speed)
        stopped = self.command(readings(front=0.25))
        self.assertEqual((stopped.linear_speed, stopped.angular_speed), (0, 0))
        self.assertEqual(stopped.state, 'obstacle_stop')

    def test_close_obstacle_on_any_side_stops_both_axes(self):
        for side in ('front', 'right', 'rear', 'left'):
            command = self.command(readings(**{side: 0.2}), {'x': 1, 'y': 2})
            self.assertEqual((command.linear_speed, command.angular_speed), (0, 0))

    def test_stale_missing_invalid_or_unmapped_sensors_block_motion(self):
        for change in ({'fresh': False}, {'status': 'invalid'}, {'distance_m': float('nan')},
                       {'status': 'beyond_range', 'distance_m': 0.4}):
            data = readings()
            data[0].update(change)
            self.assertEqual(self.command(data).state, 'ultrasound_unavailable')
        self.assertEqual(self.command(readings()[1:]).state, 'ultrasound_unavailable')
        data = readings()
        data.append(dict(data[0], topic='/sonar/number7', side=None))
        self.assertEqual(self.command(data).state, 'ultrasound_unavailable')

    def test_manual_placement_matches_the_physical_sides(self):
        data = readings(right=0.35)
        for reading in data:
            reading['side'] = None
        mapping = {side: ['/sonar/' + side] for side in ('front', 'right', 'rear', 'left')}
        command = self.command(data, mapping=mapping)
        self.assertGreater(command.angular_speed, 0)
        self.assertEqual(sensor_distances(data, mapping)[0]['right'], 0.35)

    def test_no_target_or_reached_goal_cannot_initiate_avoidance_motion(self):
        controller = PotentialFieldController(self.base, readings(right=0.35), {})
        for target in (None, {'x': 0, 'y': 0.3}):
            command = controller.compute(target)
            self.assertEqual((command.linear_speed, command.angular_speed), (0, 0))

    def test_too_close_flag_stops_even_when_bound_exceeds_stop_distance(self):
        data = readings()
        data[0].update(status='too_close', distance_m=0.4)
        self.assertEqual(self.command(data).state, 'obstacle_stop')

    def test_invalid_mapping_is_rejected(self):
        for value in (None, {'up': []}, {'front': '/sonar/front'}, {'front': [3]},
                      {'front': ['/sonar/front', '/sonar/front']}):
            with self.assertRaises(ValueError):
                validate_mapping(value)
