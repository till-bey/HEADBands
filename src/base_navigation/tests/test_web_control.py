"""Exercise browser-driven navigation with a fake robot, never a ROS publisher."""

from pathlib import Path
import json
import sys
from threading import Lock
import time
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller import PController
from web_control import WebRobotControl
from dashboard_server import DashboardServer


class TestClock:
    def __init__(self):
        self.offset = 0.0

    def __call__(self):
        return time.monotonic() + self.offset


class FakeRobot:
    """Record commands while emulating an independently spinning ROS session."""

    def __init__(self):
        self.connected = True
        self.running = False
        self.exited = False
        self._commands = []
        self._lock = Lock()
        self._lift_state = {"connected": True, "position": 0.2, "fresh": True, "moving": False}
        self._lift_commands = []
        self._lift_stops = 0
        self._head_state = {"connected": True, "fresh": True, "pitch": 0.0, "yaw": 0.3, "moving": False}
        self._head_commands = []

    def __enter__(self):
        self.running = True
        return self

    def __exit__(self, *args):
        self.set_velocity(0, 0)
        self.stop_lift()
        self.stop_head()
        self.running = False
        self.exited = True

    def set_velocity(self, linear_speed=0, angular_speed=0):
        with self._lock:
            self._commands.append((linear_speed, angular_speed))
        return self.connected

    def wait(self, seconds):
        time.sleep(min(seconds, 0.005))

    def log(self, message):
        pass

    def commands(self):
        with self._lock:
            return list(self._commands)

    @property
    def lift_status(self):
        with self._lock:
            return self._lift_state.copy()

    def set_lift_state(self, **changes):
        with self._lock:
            self._lift_state.update(changes)

    def move_lift(self, delta_m):
        with self._lock:
            if not self._lift_state['connected'] or not self._lift_state['fresh']:
                raise ValueError("Fresh connected lift feedback required")
            target = self._lift_state['position'] + delta_m
            if not 0 <= target <= 0.6:
                raise ValueError("Lift position outside travel limits")
            self._lift_commands.append(delta_m)
            self._lift_state.update(position=target, moving=True)

    def stop_lift(self):
        with self._lock:
            self._lift_stops += 1
            self._lift_state['moving'] = False

    def lift_commands(self):
        with self._lock:
            return list(self._lift_commands)

    def lift_stops(self):
        with self._lock:
            return self._lift_stops

    @property
    def head_status(self):
        with self._lock:
            return self._head_state.copy()

    @property
    def ultrasound_status(self):
        return getattr(self, '_ultrasound', [])

    def move_head(self, delta_rad):
        with self._lock:
            self._head_commands.append(delta_rad)
            self._head_state['pitch'] += delta_rad
            self._head_state['moving'] = True

    def stop_head(self):
        with self._lock:
            self._head_state['moving'] = False

    def set_head_state(self, **values):
        with self._lock:
            self._head_state.update(values)

    def head_commands(self):
        with self._lock:
            return self._head_commands.copy()


class WebControlTests(unittest.TestCase):
    def test_avoidance_toggle_stops_and_requires_a_new_run_request(self):
        self.start()
        old_generation = self.control.status()['generation']
        self.control.configure_avoidance(True, {}, expected_generation=old_generation)
        self.assertFalse(self.control.status()['running'])
        self.wait_for(lambda: self.robot.commands()[-1] == (0, 0))
        with self.assertRaises(ValueError):
            self.control.start(expected_generation=old_generation)
        with self.assertRaises(ValueError):
            self.control.start()  # Missing ultrasound coverage.
        with self.assertRaises(ValueError):
            self.control.configure_avoidance(False, {}, expected_generation=old_generation)

    def test_avoidance_stale_sensor_stops_and_does_not_resume_on_recovery(self):
        from test_avoidance import readings
        self.robot._ultrasound = readings(right=0.35)
        self.connect()
        self.control.configure_avoidance(True, {})
        self.wait_for(lambda: self.control.status()['avoidance_ready'])
        self.update_target()
        self.control.start()
        self.wait_for(lambda: any(linear > 0 and angular > 0 for linear, angular in self.robot.commands()))
        stale = readings()
        stale[0]['fresh'] = False
        self.robot._ultrasound = stale
        self.wait_for(lambda: not self.control.status()['running'])
        self.wait_for(lambda: self.robot.commands()[-1] == (0, 0))
        self.robot._ultrasound = readings()
        self.wait_for(lambda: self.control.status()['avoidance_ready'])
        self.assertFalse(self.control.status()['running'])

    def test_avoidance_final_send_rechecks_close_obstacle_and_stop(self):
        from test_avoidance import readings
        self.robot._ultrasound = readings()
        self.connect()
        self.control.configure_avoidance(True, {})
        self.wait_for(lambda: self.control.status()['avoidance_ready'])
        self.update_target()
        self.control.start()
        self.robot._ultrasound = readings(front=0.1)
        self.control._send_velocity(self.robot, 0.15, 0.4)
        self.assertEqual(self.robot.commands()[-1], (0, 0))
        self.control.stop()
        self.robot._ultrasound = readings()
        self.control._send_velocity(self.robot, 0.15, 0.4)
        self.assertEqual(self.robot.commands()[-1], (0, 0))

    def test_http_avoidance_toggle_and_mapping_are_token_guarded(self):
        display = DashboardServer(port=0, robot_control=self.control)
        try:
            def post(values):
                request = Request(display.url + '/robot/avoidance', data=json.dumps(values).encode(),
                                  headers={'Content-Type': 'application/json'})
                with urlopen(request, timeout=2) as response:
                    return json.load(response)
            with self.assertRaises(HTTPError) as error:
                post({'enabled': True, 'mapping': {}})
            self.assertEqual(error.exception.code, 400)
            old = self.control.status()['generation']
            post({'enabled': True, 'mapping': {'front': ['/sonar/front']}, 'generation': old})
            self.assertTrue(self.control.status()['avoidance_enabled'])
            self.assertEqual(self.control.status()['avoidance_mapping'], {'front': ['/sonar/front']})
            with self.assertRaises(HTTPError) as error:
                post({'enabled': False, 'mapping': {}, 'generation': old})
            self.assertEqual(error.exception.code, 409)
            post({'enabled': False, 'mapping': {}, 'generation': self.control.status()['generation']})
            self.assertFalse(self.control.status()['avoidance_enabled'])
            self.assertFalse(self.control.status()['running'])
        finally:
            display.close()

    def test_ultrasound_readings_reach_browser_status_and_clear_on_disconnect(self):
        self.robot._ultrasound = [{'topic': '/sonar/front', 'side': 'front', 'distance_m': 0.2,
                                   'fresh': True, 'status': 'valid', 'frame_id': 'front', 'age_seconds': 0}]
        self.connect()
        self.wait_for(lambda: bool(self.control.status()['ultrasound']))
        self.assertEqual(self.control.status()['ultrasound'][0]['distance_m'], 0.2)
        self.control.disconnect()
        self.wait_for(lambda: self.control.status()['connection'] == 'disconnected')
        self.assertEqual(self.control.status()['ultrasound'], [])

    def setUp(self):
        self.robot = FakeRobot()
        self.factory = Mock(return_value=self.robot)
        self.clock = TestClock()
        self.control = WebRobotControl(robot_factory=self.factory, clock=self.clock)
        self.addCleanup(self.control.close)
        self.controller = PController()

    def wait_for(self, predicate, message="Worker did not reach the expected state"):
        deadline = time.monotonic() + 0.8
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.005)
        self.fail(message + f": {self.control.status()}")

    def update_target(self, position=None, age=0):
        if position is None:
            position = {"x": 0, "y": 2}
        self.control.update_target(position, self.clock() - age, self.controller,
                                   target_timeout=0.5, camera_status="Tracking")

    def connect(self):
        self.control.connect()
        self.wait_for(lambda: self.control.status()['connection'] == 'connected'
                      and self.control.status()['drive_connected'])

    def start(self):
        self.update_target()
        self.connect()
        self.control.start()
        self.wait_for(lambda: any(linear > 0 for linear, _ in self.robot.commands()))

    def test_preview_never_initializes_ros_or_starts_without_connection(self):
        self.update_target()
        self.factory.assert_not_called()
        self.assertFalse(self.control.status()['running'])
        with self.assertRaises(ValueError):
            self.control.start()
        self.factory.assert_not_called()

    def test_connection_publishes_only_zero_until_run_and_stop_latches(self):
        self.update_target()
        self.connect()
        self.wait_for(lambda: len(self.robot.commands()) >= 2)
        self.assertTrue(all(command == (0, 0) for command in self.robot.commands()))
        self.assertFalse(self.control.status()['running'])
        self.control.start()
        self.wait_for(lambda: any(linear > 0 for linear, _ in self.robot.commands()))
        self.control.stop()
        first_after_stop = len(self.robot.commands())
        self.wait_for(lambda: len(self.robot.commands()) > first_after_stop)
        self.update_target()
        self.control.heartbeat()
        time.sleep(0.02)
        self.assertFalse(self.control.status()['running'])
        self.assertTrue(all(command == (0, 0) for command in self.robot.commands()[first_after_stop:]))

    def test_start_requires_a_fresh_target_and_drive_subscriber(self):
        self.connect()
        with self.assertRaises(ValueError):
            self.control.start()
        self.update_target(age=2)
        with self.assertRaises(ValueError):
            self.control.start()
        self.update_target()
        self.robot.connected = False
        self.wait_for(lambda: not self.control.status()['drive_connected'])
        with self.assertRaises(ValueError):
            self.control.start()
        self.assertFalse(any(linear or angular for linear, angular in self.robot.commands()))

    def test_stale_or_missing_camera_target_commands_zero(self):
        self.start()
        self.update_target(age=2)
        self.wait_for(lambda: self.robot.commands()[-1] == (0, 0))
        self.control.update_target(None, self.clock(), self.controller,
                                   target_timeout=0.5, camera_status="No frisbee")
        first_missing = len(self.robot.commands())
        self.wait_for(lambda: len(self.robot.commands()) > first_missing)
        self.assertTrue(all(command == (0, 0) for command in self.robot.commands()[first_missing:]))

    def test_camera_reconfiguration_disarms_until_another_run_click(self):
        self.start()
        self.control.clear_target("Camera settings changed")
        self.assertFalse(self.control.status()['running'])
        first_after_clear = len(self.robot.commands())
        self.update_target()
        self.control.heartbeat()
        self.wait_for(lambda: len(self.robot.commands()) > first_after_clear)
        self.assertTrue(all(command == (0, 0) for command in self.robot.commands()[first_after_clear:]))

    def test_browser_lease_expiry_stops_and_heartbeat_does_not_rearm(self):
        self.start()
        self.clock.offset += 4
        self.wait_for(lambda: not self.control.status()['running'])
        self.wait_for(lambda: self.robot.commands()[-1] == (0, 0))
        first_after_expiry = len(self.robot.commands())
        self.update_target()
        self.control.heartbeat()
        self.wait_for(lambda: len(self.robot.commands()) > first_after_expiry)
        self.assertFalse(self.control.status()['running'])
        self.assertTrue(all(command == (0, 0) for command in self.robot.commands()[first_after_expiry:]))

    def test_inflight_observation_from_previous_camera_cannot_rearm(self):
        self.connect()
        generation = self.control.target_generation()
        self.control.clear_target("Camera changed")
        self.control.update_target({'x': 0, 'y': 2}, self.clock(), self.controller,
                                   target_timeout=0.5, expected_generation=generation)
        with self.assertRaises(ValueError):
            self.control.start()
        self.assertFalse(self.control.status()['running'])

    def test_stop_is_rechecked_before_sending_an_inflight_command(self):
        self.start()
        self.control.stop()
        self.control._send_velocity(self.robot, 0.15, 0.4)
        self.assertEqual(self.robot.commands()[-1], (0, 0))

    def test_disconnect_and_shutdown_stop_and_release_robot(self):
        self.start()
        self.control.disconnect()
        self.wait_for(lambda: self.control.status()['connection'] == 'disconnected')
        self.assertTrue(self.robot.exited)
        self.assertEqual(self.robot.commands()[-1], (0, 0))
        self.assertFalse(self.control.status()['running'])

    def test_shutdown_while_running_stops_before_releasing_robot(self):
        self.start()
        self.control.close()
        self.assertTrue(self.robot.exited)
        self.assertEqual(self.robot.commands()[-1], (0, 0))
        self.assertFalse(self.control.status()['running'])
        self.assertEqual(self.control.status()['connection'], 'disconnected')

    def test_drive_reconnection_does_not_resume_motion_without_run(self):
        self.start()
        self.robot.connected = False
        self.wait_for(lambda: not self.control.status()['running'])
        self.wait_for(lambda: self.robot.commands()[-1] == (0, 0))
        self.robot.connected = True
        first_after_reconnect = len(self.robot.commands())
        self.update_target()
        self.control.heartbeat()
        self.wait_for(lambda: self.control.status()['drive_connected']
                      and len(self.robot.commands()) > first_after_reconnect)
        self.assertFalse(self.control.status()['running'])
        self.assertTrue(all(command == (0, 0)
                            for command in self.robot.commands()[first_after_reconnect:]))

    def test_unavailable_ros_keeps_preview_service_stopped(self):
        self.factory.side_effect = ImportError("No module named rclpy")
        self.control.connect()
        self.wait_for(lambda: self.control.status()['connection'] == 'error')
        status = self.control.status()
        self.assertIn('rclpy', status['status'])
        self.assertFalse(status['running'])
        self.assertEqual((status['linear_speed'], status['angular_speed']), (0, 0))
        with self.assertRaises(ValueError):
            self.control.start()

    def connect_lift(self):
        self.connect()
        self.wait_for(lambda: self.control.status()['lift_connected']
                      and self.control.status()['lift_fresh'])

    def test_head_steps_pause_navigation_block_restart_and_stop_on_lease_expiry(self):
        import math
        self.start()
        self.wait_for(lambda: self.control.status()['head_fresh'])
        self.control.jog_head('down')
        self.wait_for(lambda: bool(self.robot.head_commands()))
        self.wait_for(lambda: self.control.status()['head_moving'])
        self.assertAlmostEqual(self.robot.head_commands()[0], math.radians(5))
        self.assertFalse(self.control.status()['running'])
        self.update_target()
        with self.assertRaises(ValueError):
            self.control.start()
        self.clock.offset += 4
        self.wait_for(lambda: not self.control.status()['head_moving'])
        self.assertFalse(self.robot.head_status['moving'])

    def test_head_requires_fresh_feedback_and_rejects_old_tokens(self):
        with self.assertRaises(ValueError):
            self.control.jog_head('up')
        self.connect()
        self.wait_for(lambda: self.control.status()['head_fresh'])
        old = self.control.status()['generation']
        self.control.stop()
        with self.assertRaises(ValueError):
            self.control.jog_head('up', expected_generation=old)
        self.robot.set_head_state(fresh=False)
        self.wait_for(lambda: not self.control.status()['head_fresh'])
        with self.assertRaises(ValueError):
            self.control.jog_head('up')
        self.assertEqual(self.robot.head_commands(), [])

    def test_http_head_buttons_and_stop(self):
        import math
        self.connect()
        self.wait_for(lambda: self.control.status()['head_fresh'])
        display = DashboardServer(port=0, robot_control=self.control)
        try:
            def post(path, values):
                request = Request(display.url + path, data=json.dumps(values).encode(),
                                  headers={'Content-Type': 'application/json'})
                with urlopen(request, timeout=2) as response:
                    return json.load(response)
            with self.assertRaises(HTTPError) as error:
                post('/robot/head/down', {})
            self.assertEqual(error.exception.code, 400)
            for direction, sign in [('up', -1), ('down', 1)]:
                count = len(self.robot.head_commands())
                post('/robot/head/' + direction, {'generation': self.control.status()['generation']})
                self.wait_for(lambda: len(self.robot.head_commands()) > count)
                self.assertAlmostEqual(self.robot.head_commands()[-1], sign * math.radians(5))
                post('/robot/stop', {})
                self.wait_for(lambda: not self.control.status()['head_moving'])
            post('/robot/head/down', {'generation': self.control.status()['generation']})
            self.wait_for(lambda: self.control.status()['head_moving'])
            post('/robot/disconnect', {})
            self.wait_for(lambda: self.robot.exited)
            self.assertFalse(self.robot.head_status['moving'])
        finally:
            display.close()

    def test_lift_buttons_queue_one_centimeter_up_and_down(self):
        self.connect_lift()
        self.control.jog_lift('up')
        self.wait_for(lambda: len(self.robot.lift_commands()) == 1)
        self.assertAlmostEqual(self.robot.lift_commands()[0], 0.01)
        self.wait_for(lambda: self.control.status()['lift_moving'])
        self.assertFalse(self.control.status()['running'])
        self.update_target()
        with self.assertRaises(ValueError):
            self.control.start()
        self.robot.set_lift_state(moving=False)
        self.wait_for(lambda: not self.control.status()['lift_moving'])
        self.control.jog_lift('down')
        self.wait_for(lambda: len(self.robot.lift_commands()) == 2)
        self.assertAlmostEqual(self.robot.lift_commands()[1], -0.01)

    def test_lift_requires_ros_subscriber_and_fresh_feedback(self):
        with self.assertRaises(ValueError):
            self.control.jog_lift('up')
        self.connect_lift()
        self.robot.set_lift_state(fresh=False)
        self.wait_for(lambda: not self.control.status()['lift_fresh'])
        with self.assertRaises(ValueError):
            self.control.jog_lift('up')
        self.robot.set_lift_state(fresh=True, connected=False)
        self.wait_for(lambda: not self.control.status()['lift_connected'])
        with self.assertRaises(ValueError):
            self.control.jog_lift('down')
        self.assertEqual(self.robot.lift_commands(), [])

    def test_lift_does_not_require_a_mobile_base_subscriber(self):
        self.connect_lift()
        self.robot.connected = False
        self.wait_for(lambda: not self.control.status()['drive_connected'])
        self.control.jog_lift('up')
        self.wait_for(lambda: self.robot.lift_commands() == [0.01])

    def test_lift_jog_disarms_navigation_and_requires_a_new_observation(self):
        self.start()
        self.wait_for(lambda: self.control.status()['lift_fresh'])
        generation = self.control.target_generation()
        self.control.jog_lift('up')
        self.assertFalse(self.control.status()['running'])
        self.assertNotEqual(self.control.target_generation(), generation)
        self.wait_for(lambda: self.robot.lift_commands() == [0.01])
        self.wait_for(lambda: self.robot.commands()[-1] == (0, 0))
        self.robot.set_lift_state(moving=False)
        self.wait_for(lambda: not self.control.status()['lift_moving'])
        with self.assertRaises(ValueError):
            self.control.start()

    def test_stop_cancels_lift_motion_and_rejects_a_delayed_jog(self):
        self.connect_lift()
        generation = self.control.status()['generation']
        self.control.stop()
        with self.assertRaises(ValueError):
            self.control.jog_lift('up', expected_generation=generation)
        self.assertEqual(self.robot.lift_commands(), [])
        self.control.jog_lift('up')
        self.wait_for(lambda: self.control.status()['lift_moving'])
        before_stop = self.robot.lift_stops()
        self.control.stop()
        self.wait_for(lambda: self.robot.lift_stops() > before_stop)
        self.wait_for(lambda: not self.control.status()['lift_moving'])

    def test_browser_lease_expiry_stops_lift_motion(self):
        self.connect_lift()
        self.control.jog_lift('up')
        self.wait_for(lambda: self.control.status()['lift_moving'])
        before_stop = self.robot.lift_stops()
        self.clock.offset += 4
        self.wait_for(lambda: self.robot.lift_stops() > before_stop)
        self.wait_for(lambda: not self.control.status()['lift_moving'])

    def test_stop_cancels_a_lift_step_before_the_worker_sends_it(self):
        # Simulate discovery without starting the worker, so Stop deterministically
        # arrives after enqueueing and before the next ROS loop iteration.
        with self.control._lock:
            self.control._connection = 'connected'
            self.control._lift = self.robot.lift_status
        self.control.jog_lift('up')
        self.control.stop()
        self.control._service_lift(self.robot)
        self.assertFalse(self.control.status()['lift_moving'])
        self.assertEqual(self.robot.lift_commands(), [])

    def test_http_lift_requires_current_token_and_stops_on_disconnect(self):
        self.connect_lift()
        display = DashboardServer(port=0, robot_control=self.control)
        try:
            def post(path, values):
                request = Request(display.url + path, data=json.dumps(values).encode(),
                                  headers={'Content-Type': 'application/json'})
                return urlopen(request, timeout=2)

            with self.assertRaises(HTTPError) as error:
                post('/robot/lift/up', {})
            self.assertEqual(error.exception.code, 400)
            generation = self.control.status()['generation']
            self.control.stop()
            with self.assertRaises(HTTPError) as error:
                post('/robot/lift/up', {'generation': generation})
            self.assertEqual(error.exception.code, 409)
            with post('/robot/lift/up', {'generation': self.control.status()['generation']}) as response:
                self.assertEqual(response.status, 200)
            self.wait_for(lambda: self.robot.lift_commands() == [0.01])
            with post('/robot/disconnect', {}) as response:
                self.assertEqual(response.status, 200)
            self.wait_for(lambda: self.robot.exited)
            self.assertFalse(self.robot.lift_status['moving'])
        finally:
            display.close()

    def test_http_buttons_run_stop_and_disconnect_the_fake_robot(self):
        display = DashboardServer(port=0, robot_control=self.control)
        try:
            def post(path, values=None):
                if values is None:
                    values = {'generation': status()['generation']} if path == '/robot/start' else {}
                request = Request(display.url + path, data=json.dumps(values).encode(),
                                  headers={'Content-Type': 'application/json'})
                with urlopen(request, timeout=2) as response:
                    self.assertEqual(response.status, 200)
                    return json.load(response)

            def status():
                with urlopen(display.url + '/robot/status', timeout=2) as response:
                    return json.load(response)

            self.assertEqual(status()['connection'], 'disconnected')
            self.factory.assert_not_called()
            for headers, expected in (({'Content-Type': 'text/plain'}, 415),
                                      ({'Content-Type': 'application/json', 'Origin': 'https://example.invalid'}, 403)):
                request = Request(display.url + '/robot/connect', data=b'{}', headers=headers)
                with self.assertRaises(HTTPError) as error:
                    urlopen(request, timeout=2)
                self.assertEqual(error.exception.code, expected)
                self.factory.assert_not_called()
            post('/robot/connect')
            self.wait_for(lambda: status()['drive_connected'])
            with self.assertRaises(HTTPError) as error:
                post('/robot/start')
            self.assertEqual(error.exception.code, 409)
            self.assertFalse(any(linear or angular for linear, angular in self.robot.commands()))

            self.update_target()
            old_generation = status()['generation']
            post('/robot/stop')
            with self.assertRaises(HTTPError) as error:
                post('/robot/start', {'generation': old_generation})
            self.assertEqual(error.exception.code, 409)
            self.assertFalse(status()['running'])
            post('/robot/start')
            post('/robot/heartbeat')
            self.wait_for(lambda: any(linear > 0 for linear, _ in self.robot.commands()))
            self.assertTrue(status()['running'])
            self.assertGreater(status()['linear_speed'], 0)

            post('/robot/stop')
            first_after_stop = len(self.robot.commands())
            self.wait_for(lambda: len(self.robot.commands()) > first_after_stop)
            self.assertFalse(status()['running'])
            self.assertTrue(all(command == (0, 0)
                                for command in self.robot.commands()[first_after_stop:]))
            post('/robot/disconnect')
            self.wait_for(lambda: status()['connection'] == 'disconnected')
            self.assertTrue(self.robot.exited)
            self.assertEqual(self.robot.commands()[-1], (0, 0))
        finally:
            display.close()
        self.assertFalse(display._thread.is_alive())


if __name__ == '__main__':
    unittest.main()
