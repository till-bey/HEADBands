"""Exercise camera worker failure handling without a camera or ROS."""

import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller import PController
from main import control_loop
from camera import CameraTargetReader


class FakeCamera:
    def __init__(self, results):
        self.results = iter(results)
        self.closed = False
        self.reader = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def get_robot_ground_position(self, **kwargs):
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


def check_worker_invalidates_previous_target_on_camera_failure():
    camera = FakeCamera([{"x": 0, "y": 1}, RuntimeError("disconnected")])
    reader = CameraTargetReader(camera, robot_z=0, tilt_rad=0.3, horizontal_fov_deg=90)
    reader.start()
    reader._thread.join(timeout=1)
    position, observed_at, status = reader.snapshot()
    assert position is None
    assert observed_at > 0
    assert "disconnected" in status
    assert camera.closed
    reader.close()


def check_worker_clears_target_on_timeout_and_invalid_projection():
    for failure in (TimeoutError("no frame"), ValueError("above horizon")):
        camera = FakeCamera([{"x": 0, "y": 1}, failure])
        reader = CameraTargetReader(camera, robot_z=0, tilt_rad=0.3, horizontal_fov_deg=90)
        # Run synchronously and stop immediately after the second result is saved.
        snapshots = []

        def finish_after_second_result(timeout):
            snapshots.append(reader.snapshot())
            if len(snapshots) == 2:
                reader._stop.set()

        reader._stop.wait = finish_after_second_result
        reader._run()
        assert snapshots[0][0] == {"x": 0, "y": 1}
        assert snapshots[1][0] is None
        assert snapshots[1][2] == str(failure)
        assert camera.closed


class CameraWorkerTests(unittest.TestCase):
    def test_camera_failure(self):
        check_worker_invalidates_previous_target_on_camera_failure()

    def test_timeout_and_invalid_projection(self):
        check_worker_clears_target_on_timeout_and_invalid_projection()


class ControlLoopTests(unittest.TestCase):
    def test_live_settings_and_status_updates_each_iteration(self):
        reader, robot = Mock(), Mock()
        reader.snapshot.return_value = ({"x": 0, "y": 0.2}, 99.9, "Tracking")
        type(robot).running = property(Mock(side_effect=[True, True, True, False]))
        robot.set_velocity.return_value = True
        prepare = Mock(side_effect=[
            (PController(linear_kp=0.1, target_distance=0), 0.5),
            (PController(linear_kp=0.2, target_distance=0), 0.5),
            (PController(linear_kp=0.2, target_distance=0), 0.05),
        ])
        report = Mock()
        with patch("main.time.monotonic", return_value=100):
            control_loop(reader, PController(), robot,
                         prepare_step=prepare, on_status=report)
        speeds = [call.args[0] for call in robot.set_velocity.call_args_list]
        self.assertAlmostEqual(speeds[0], 0.02)
        self.assertAlmostEqual(speeds[1], 0.04)
        self.assertEqual(speeds[2], 0)
        self.assertEqual([call.args[0] for call in report.call_args_list],
                         ['approaching', 'approaching', 'Camera result expired'])
        self.assertEqual(robot.log.call_count, 2)
        self.assertEqual(robot.wait.call_count, 3)

    def run_loop(self, position, timestamp=100, connected=True):
        reader, robot = Mock(), Mock()
        reader.snapshot.return_value = position, timestamp, "No frisbee detected"
        robot.running = True
        robot.set_velocity.return_value = connected
        robot.wait.side_effect = lambda seconds: setattr(robot, "running", False)
        with patch("main.time.monotonic", return_value=100):
            control_loop(reader, PController(), robot)
        robot.wait.assert_called_once_with(0.05)
        return robot

    def test_fresh_target_drives_and_steers_together(self):
        self.run_loop({"x": 0, "y": 2}).set_velocity.assert_called_once_with(0.15, 0)
        robot = self.run_loop({"x": 1, "y": 2})
        linear, angular = robot.set_velocity.call_args.args
        self.assertAlmostEqual(linear, 0.15 * 2 / (5 ** 0.5))
        self.assertEqual(angular, -0.4)

    def test_stale_or_missing_target_stops(self):
        for position, timestamp in (({"x": 0, "y": 2}, 99), (None, 100)):
            self.run_loop(position, timestamp).set_velocity.assert_called_once_with(0, 0)

    def test_disconnected_drive_reports_waiting(self):
        robot = self.run_loop({"x": 0, "y": 2}, connected=False)
        robot.log.assert_called_once_with("Waiting for cmd_vel subscriber")
