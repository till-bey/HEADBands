"""Check dashboard geometry and rendering without opening a GUI or camera."""

import math
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, MagicMock, patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller import PController
from dashboard import inspect_frame, render_view, run_dashboard
from dashboard_server import DashboardServer
from config import parse_args, SETTINGS_FIELDS


class DashboardTests(unittest.TestCase):
    def test_debug_panel_shows_controller_errors_and_clears_when_target_is_lost(self):
        position = {'x': -0.1, 'y': 0.2, 'distance': math.hypot(-0.1, 0.2),
                    'bearing_rad': math.atan2(-0.1, 0.2),
                    'bearing_deg': math.degrees(math.atan2(-0.1, 0.2))}
        command = self.controller.compute(position)
        with patch.object(cv2, 'putText', wraps=cv2.putText) as put_text:
            render_view(self.frame, (100, 100), position, command, 'Tracking')
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn('Position error: -0.076 m', labels)
        self.assertIn('Angle error: -0.464 rad (-26.6 deg)', labels)
        with patch.object(cv2, 'putText', wraps=cv2.putText) as put_text:
            render_view(self.frame, None, None, self.controller.compute(None), 'No target')
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn('Position error: -- m', labels)
        self.assertIn('Angle error: -- rad (-- deg)', labels)

    def setUp(self):
        self.args = SimpleNamespace(robot_z=0, tilt_rad=math.pi / 6,
                                    horizontal_fov=90, vertical_fov=None)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.camera = Mock()
        self.controller = PController()

    def test_live_frames_are_portrait_before_detection_and_projection(self):
        from camera import FrisbeeCamera, pixel_to_ground_robot
        frame = np.full((480, 640, 3), (20, 100, 20), np.uint8)
        cv2.circle(frame, (400, 100), 20, (240, 240, 240), -1)
        camera = FrisbeeCamera('http://127.0.0.1/unused')
        with patch.object(camera, '_camera') as source:
            source.get_latest.return_value = SimpleNamespace(frame=frame, sequence=1, age_seconds=0)
            upright = camera.read_frame()
            self.assertEqual(upright.shape, (640, 480, 3))
            np.testing.assert_array_equal(upright, cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))
            center = camera.detect_center(upright)
            self.assertIsNotNone(center)
            self.assertLessEqual(math.dist(center, (379, 400)), 3)
            source.get_latest.return_value.sequence = 2
            position = camera.get_robot_ground_position(
                robot_z=0, tilt_rad=1.57, horizontal_fov_deg=77.32, vertical_fov_deg=61.93)
            self.assertEqual(position, pixel_to_ground_robot(
                *center, 480, 640, 0, 1.57, 77.32, 61.93))
            camera.close()

    def test_already_portrait_frames_are_not_rotated_again(self):
        from camera import FrisbeeCamera
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        frame[0, 0] = (20, 100, 200)
        camera = FrisbeeCamera('http://127.0.0.1/unused')
        with patch.object(camera, '_camera') as source:
            source.get_latest.return_value = SimpleNamespace(frame=frame, sequence=1, age_seconds=0)
            np.testing.assert_array_equal(camera.read_frame(), frame)
            camera.close()

    def test_ellipse_detector_searches_upper_image(self):
        from camera import FrisbeeCamera
        for y in (60, 115, 360):
            with self.subTest(y=y):
                frame = np.full((480, 640, 3), (20, 100, 20), np.uint8)
                cv2.circle(frame, (320, y), 20, (240, 240, 240), -1)
                camera = FrisbeeCamera('http://127.0.0.1/unused')
                try:
                    center = camera.detect_center(frame)
                    self.assertIsNotNone(center)
                    self.assertLessEqual(math.dist(center, (320, y)), 3)
                finally:
                    camera.close()

    def test_selected_center_and_projected_geometry_match(self):
        self.camera.detect_center.return_value = (400, 320)
        center, position, command, status = inspect_frame(
            self.camera, self.frame, self.args, self.controller)
        self.camera.detect_center.assert_called_once_with(self.frame)
        self.assertEqual(center, (400, 320))
        self.assertAlmostEqual(position['distance'], math.hypot(position['x'], position['y']))
        self.assertAlmostEqual(position['bearing_rad'], math.atan2(position['x'], position['y']))
        self.assertGreater(position['bearing_rad'], 0)
        self.assertLess(command.angular_speed, 0)
        self.assertEqual(command.angle_error, position['image_angle_rad'])
        self.assertEqual(status, "Tracking")
        rendered = render_view(self.frame, center, position, command, status)
        self.assertGreaterEqual(rendered.shape[0], self.frame.shape[0])
        self.assertGreater(rendered.shape[1], self.frame.shape[1])
        self.assertTrue(np.any(rendered[:self.frame.shape[0], self.frame.shape[1]:]))
        self.assertTrue(np.any(rendered[300:340, 380:420]))
        self.assertFalse(np.any(self.frame))  # Overlay leaves source untouched.

    def test_no_detection_clears_measurements(self):
        self.camera.detect_center.return_value = None
        center, position, command, status = inspect_frame(
            self.camera, self.frame, self.args, self.controller)
        self.assertIsNone(center)
        self.assertIsNone(position)
        self.assertEqual(command.state, "lost")
        self.assertEqual(status, "No frisbee detected")
        render_view(self.frame, center, position, command, status)

    def test_horizon_detection_keeps_marker_without_ground_measurements(self):
        self.camera.detect_center.return_value = (320, 100)
        self.args.tilt_rad = 0
        center, position, command, status = inspect_frame(
            self.camera, self.frame, self.args, self.controller)
        self.assertEqual(center, (320, 100))
        self.assertIsNone(position)
        self.assertEqual(command.state, "lost")
        self.assertIn("Invalid ground projection", status)
        render_view(self.frame, center, position, command, status)

    def test_browser_preview_serves_image_without_highgui(self):
        with patch.object(cv2, "namedWindow", side_effect=AssertionError("GUI used")), \
             patch.object(cv2, "destroyAllWindows", side_effect=AssertionError("GUI used")):
            display = DashboardServer(port=0)
            try:
                display.show(self.frame)
                with urlopen(display.url, timeout=2) as response:
                    self.assertIn(b"Navigation dashboard", response.read())
                with urlopen(display.url + "/frame.jpg", timeout=2) as response:
                    self.assertEqual(response.headers['Content-Type'], 'image/jpeg')
                    decoded = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape, self.frame.shape)
            finally:
                display.close()

    def test_default_dashboard_display_does_not_call_window_cleanup(self):
        camera = MagicMock()
        camera.__enter__.return_value = camera
        camera.read_frame.side_effect = KeyboardInterrupt
        with patch('dashboard.create_camera', return_value=camera), \
             patch('dashboard.DashboardServer') as browser, \
             patch.object(cv2, 'destroyAllWindows', side_effect=AssertionError("GUI used")):
            settings = parse_args(['http://127.0.0.1/unused', '--tilt-rad', '0.3'], dashboard=True)
            browser.return_value.take_update.return_value = ('connect', settings)
            from dashboard import main
            main(['http://127.0.0.1/unused', '--tilt-rad', '0.3'])
            self.assertEqual(browser.call_args.args, (8765,))
            self.assertEqual(browser.call_args.kwargs['settings'].url, settings.url)
            browser.return_value.close.assert_called_once()
            camera.close.assert_called_once()

    def test_dashboard_can_wait_for_browser_configuration(self):
        args = parse_args([''], dashboard=True)
        self.assertEqual(args.url, '')
        self.assertEqual((args.horizontal_fov, args.vertical_fov), (77.32, 61.93))
        display = Mock()
        display.take_update.side_effect = [None, KeyboardInterrupt]
        with patch('dashboard.create_camera') as create, patch('dashboard.time.sleep'):
            with self.assertRaises(KeyboardInterrupt):
                run_dashboard(display, args)
            create.assert_not_called()
            display.show.assert_called_once()

    def test_reconnect_and_disconnect_close_previous_cameras(self):
        args = parse_args(['http://127.0.0.1/first'], dashboard=True)
        second = parse_args(['http://127.0.0.1/second', '--horizontal-fov', '80'], dashboard=True)
        cameras = [Mock(), Mock()]
        for camera in cameras:
            camera.read_frame.return_value = self.frame
            camera.detect_center.return_value = None
        display = Mock()
        display.take_update.side_effect = [('connect', args), ('connect', second),
                                          ('disconnect', None), KeyboardInterrupt]
        with patch('dashboard.create_camera', side_effect=cameras) as create, patch('dashboard.time.sleep'):
            with self.assertRaises(KeyboardInterrupt):
                run_dashboard(display, args)
            self.assertEqual(create.call_args_list[1].args[0].horizontal_fov, 80)
            for camera in cameras:
                camera.close.assert_called_once()
            self.assertEqual(display.show.call_count, 3)

    def test_browser_settings_validation_and_disconnect(self):
        from tempfile import TemporaryDirectory
        from config import save_settings
        from test_settings import sample_values
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        settings_path = Path(directory.name) / 'settings.json'
        save_settings(settings_path, sample_values())
        args = parse_args(['--settings-file', str(settings_path)], dashboard=True)
        values = {name: getattr(args, name) for name in SETTINGS_FIELDS}
        values.update(url='http://127.0.0.1/stream', robot_z=0.2, horizontal_fov=80,
                      rgb_sum_min=180, channel_difference_max=65)
        display = DashboardServer(0, settings=args)
        try:
            def post(path, payload):
                request = Request(display.url + path, data=json.dumps(payload).encode(),
                                  headers={'Content-Type': 'application/json'})
                return urlopen(request, timeout=2)

            with urlopen(display.url + '/settings', timeout=2) as response:
                self.assertEqual(json.load(response)['horizontal_fov'], 77.32)
            with post('/settings', values) as response:
                self.assertEqual(response.status, 200)
            action, settings = display.take_update()
            self.assertEqual(action, 'connect')
            self.assertEqual((settings.url, settings.robot_z, settings.horizontal_fov),
                             (values['url'], 0.2, 80))
            from camera import create_camera
            camera = create_camera(settings)
            self.assertEqual((camera.settings.rgb_sum_min, camera.settings.channel_difference_max),
                             (180, 65))
            camera.close()
            self.assertIsNone(display.take_update())
            for changes in ({'horizontal_fov': 180}, {'linear_kp': -1}, {'url': 'not-a-url'},
                            {'robot_z': -1}, {'tilt_rad': float('nan')},
                            {'rgb_sum_min': -1}, {'rgb_sum_min': 766}, {'rgb_sum_min': 160.5},
                            {'channel_difference_max': -1}, {'channel_difference_max': 256},
                            {'channel_difference_max': 55.5}):
                with self.subTest(changes=changes), self.assertRaises(HTTPError) as error:
                    post('/settings', values | changes)
                self.assertEqual(error.exception.code, 400)
                self.assertIsNone(display.take_update())
            with post('/disconnect', {}) as response:
                self.assertEqual(response.status, 200)
            self.assertEqual(display.take_update(), ('disconnect', None))
        finally:
            display.close()
