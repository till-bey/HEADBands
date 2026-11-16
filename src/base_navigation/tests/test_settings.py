"""Exercise saved settings and browser persistence without touching real config."""

from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "perception"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SETTINGS_FIELDS, validate_settings, load_settings, parse_args, save_settings
from dashboard_server import DashboardServer


def sample_values(**changes):
    values = {
        "camera_base_height": 0.83,
        "url": "http://127.0.0.1:8080/camera.jpg",
        "snapshot": True,
        "robot_z": 0.22,
        "tilt_rad": 0.4,
        "horizontal_fov": 77.32,
        "vertical_fov": 61.93,
        "resolution": 480,
        "rgb_sum_min": 160,
        "channel_difference_max": 55,
        "min_radius": 12,
        "max_radius": 100,
        "min_ellipse_ratio": 0.4,
        "min_area": 180,
        "min_white_fraction": 0.55,
        "side_std_threshold": 30.0,
        "angular_kp": 0.9,
        "linear_kp": 0.3,
        "angular_kd": 0.0,
        "linear_kd": 0.0,
        "max_angular_speed": 0.35,
        "max_linear_speed": 0.12,
        "angle_tolerance": 0.04,
        "distance_tolerance": 0.06,
        "target_distance": 0.4,
        "target_timeout": 0.6,
    }
    values.update(changes)
    return values


class SettingsFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "settings.json"

    def test_saved_values_survive_restart_and_cli_overrides_only_selected_values(self):
        values = sample_values()
        self.assertEqual(set(values), set(SETTINGS_FIELDS))
        save_settings(self.path, values, port=8877)
        self.assertEqual(json.loads(self.path.read_text()), values | {"port": 8877})
        loaded = load_settings(self.path)
        self.assertEqual(vars(loaded), values | {"port": 8877})

        args = parse_args(["--settings-file", str(self.path)], dashboard=True)
        self.assertEqual((args.url, args.snapshot, args.port, args.resolution),
                         (values["url"], True, 8877, 480))
        args = parse_args([
            "--settings-file", str(self.path), "http://127.0.0.1:8081/stream",
            "--no-snapshot", "--rgb-sum-min", "190", "--linear-kp", "0.5",
            "--target-distance", "0.3", "--angular-kd", "0.12", "--linear-kd", "0.25",
        ], dashboard=True)
        self.assertEqual((args.url, args.snapshot, args.rgb_sum_min, args.linear_kp),
                         ("http://127.0.0.1:8081/stream", False, 190, 0.5))
        self.assertEqual((args.channel_difference_max, args.robot_z, args.port),
                         (55, 0.22, 8877))
        self.assertEqual(args.target_distance, 0.3)
        self.assertEqual((args.angular_kd, args.linear_kd), (0.12, 0.25))
        self.assertEqual(json.loads(self.path.read_text()), values | {"port": 8877})

    def test_blank_camera_url_can_be_saved_but_cannot_start_navigation(self):
        values = sample_values(url="")
        save_settings(self.path, values)
        self.assertEqual(load_settings(self.path).url, "")
        self.assertEqual(parse_args(["--settings-file", str(self.path)], dashboard=True).url, "")
        with self.assertRaises(ValueError):
            validate_settings(values)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            parse_args(["--settings-file", str(self.path)])
        self.assertEqual(error.exception.code, 2)

    def test_main_can_use_saved_camera_and_calibration(self):
        values = sample_values()
        save_settings(self.path, values)
        args = parse_args(["--settings-file", str(self.path)])
        self.assertEqual((args.url, args.tilt_rad, args.snapshot),
                         (values["url"], values["tilt_rad"], True))

    def test_old_detector_choice_is_ignored_and_removed_on_save(self):
        values = sample_values()
        for method in ("ellipse", "hough"):
            with self.subTest(method=method):
                legacy = values | {"port": 8877, "white_detector": method}
                self.path.write_text(json.dumps(legacy))
                loaded = load_settings(self.path)
                self.assertEqual(vars(loaded), values | {"port": 8877})
                # Loading a legacy file must not silently rewrite it.
                self.assertEqual(json.loads(self.path.read_text()), legacy)
                save_settings(self.path, {name: getattr(loaded, name) for name in SETTINGS_FIELDS},
                              port=loaded.port)
                saved = json.loads(self.path.read_text())
                self.assertNotIn("white_detector", saved)
                self.assertEqual(saved, values | {"port": 8877})

    def test_invalid_save_preserves_previous_file(self):
        values = sample_values()
        save_settings(self.path, values)
        before = self.path.read_bytes()
        for changes in ({"rgb_sum_min": 766}, {"channel_difference_max": -1},
                        {"angular_kd": -1}, {"linear_kd": float("inf")},
                        {"side_std_threshold": -1}, {"side_std_threshold": 766},
                        {"side_std_threshold": float("nan")},
                        {"linear_kp": float("nan")}, {"snapshot": "yes"},
                        {"horizontal_fov": 180}, {"url": "not-a-camera-url"},
                        {"min_radius": 100}, {"min_radius": 1.5}, {"max_radius": 0},
                        {"min_area": 0}, {"min_ellipse_ratio": float("nan")},
                        {"min_ellipse_ratio": 0}, {"min_white_fraction": 1.1}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError, OverflowError)):
                save_settings(self.path, values | changes)
            self.assertEqual(self.path.read_bytes(), before)

    def test_atomic_save_preserves_existing_file_permissions(self):
        save_settings(self.path, sample_values())
        self.path.chmod(0o640)
        save_settings(self.path, sample_values(robot_z=0.37, tilt_rad=0.72))
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o640)

    def test_cross_user_atomic_save_cannot_replace_file_with_inaccessible_owner(self):
        save_settings(self.path, sample_values())
        before = self.path.read_bytes()
        original_stat = self.path.stat()
        with patch('config.os.fstat', return_value=SimpleNamespace(
                st_uid=original_stat.st_uid + 1, st_gid=original_stat.st_gid)), \
             patch('config.os.fchown', side_effect=PermissionError('different user')):
            with self.assertRaisesRegex(PermissionError, 'existing configuration was left unchanged'):
                save_settings(self.path, sample_values(tilt_rad=0.9))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.path.stat().st_uid, original_stat.st_uid)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_legacy_settings_keep_detector_defaults_without_rewriting_file(self):
        from detector import DetectorSettings
        names = ('min_radius', 'max_radius', 'min_ellipse_ratio', 'min_area', 'min_white_fraction', 'side_std_threshold')
        legacy = {k: v for k, v in sample_values().items() if k not in (*names, 'camera_base_height')}
        legacy['port'] = 8765
        self.path.write_text(json.dumps(legacy))
        before = self.path.read_bytes()
        loaded = load_settings(self.path)
        self.assertEqual(loaded.camera_base_height, 0.83)
        for name in names:
            self.assertEqual(getattr(loaded, name), getattr(DetectorSettings(), name))
        self.assertEqual(self.path.read_bytes(), before)

    def test_legacy_settings_default_to_zero_derivative_gains(self):
        legacy = sample_values() | {'port': 8765}
        del legacy['angular_kd'], legacy['linear_kd']
        self.path.write_text(json.dumps(legacy))
        loaded = load_settings(self.path)
        self.assertEqual((loaded.angular_kd, loaded.linear_kd), (0.0, 0.0))

    def test_retired_edge_settings_are_ignored_and_removed_on_save(self):
        values = sample_values()
        legacy = values | {'port': 8765, 'canny_low': 40, 'canny_high': 120}
        self.path.write_text(json.dumps(legacy))
        loaded = load_settings(self.path)
        self.assertEqual(vars(loaded), values | {'port': 8765})
        self.assertEqual(json.loads(self.path.read_text()), legacy)
        save_settings(self.path, {key: getattr(loaded, key) for key in SETTINGS_FIELDS})
        self.assertEqual(json.loads(self.path.read_text()), values | {'port': 8765})

    def test_ellipse_settings_change_camera_detection(self):
        import cv2
        import numpy as np
        from camera import create_camera
        image = np.full((480, 640, 3), (20, 100, 20), np.uint8)
        cv2.ellipse(image, (320, 240), (40, 14), 20, 0, 360, (240, 240, 240), -1)
        for ratio, detected in ((0.7, False), (0.2, True)):
            values = sample_values(resolution=640, min_ellipse_ratio=ratio)
            camera = create_camera(validate_settings(values))
            try:
                self.assertEqual(camera.detect_center(image) is not None, detected)
            finally:
                camera.close()


class BrowserSettingsPersistenceTests(unittest.TestCase):
    def test_saved_endpoint_reads_disk_instead_of_cached_form_values(self):
        changed = sample_values(tilt_rad=0.9, camera_base_height=1.1)
        save_settings(self.path, changed, port=8877)
        self.assertEqual(self.get_current(), self.original)
        with urlopen(self.display.url + '/settings/saved', timeout=2) as response:
            self.assertEqual(json.load(response), changed | {'port': 8877})

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "settings.json"
        self.original = sample_values()
        save_settings(self.path, self.original, port=8877)
        args = parse_args(["--settings-file", str(self.path)], dashboard=True)
        self.display = DashboardServer(0, settings=args)
        self.addCleanup(self.display.close)

    def get_current(self):
        with urlopen(self.display.url + "/settings", timeout=2) as response:
            return json.load(response)

    def post(self, path, values):
        request = Request(self.display.url + path, data=json.dumps(values).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=2) as response:
            return json.load(response)

    def test_save_then_reload_restores_form_values_without_applying_camera(self):
        changed = sample_values(url="http://127.0.0.1:8082/other.jpg", robot_z=0.5,
                                angular_kd=0.12, linear_kd=0.25,
                                side_std_threshold=45.5,
                                rgb_sum_min=190, channel_difference_max=65,
                                tilt_rad=0.73, max_linear_speed=0.08, target_distance=0.3, min_radius=15, max_radius=120,
                                min_ellipse_ratio=0.28, min_area=220, min_white_fraction=0.6)
        self.post("/settings/save", changed)
        self.assertEqual(json.loads(self.path.read_text()), changed | {"port": 8877})
        self.assertEqual(self.get_current(), changed)
        # Saving must not replace or consume the startup connection request.
        action, pending = self.display.take_update()
        self.assertEqual((action, pending.url), ("connect", self.original["url"]))
        self.assertIsNone(self.display.take_update())

        with urlopen(self.display.url + '/settings/saved', timeout=2) as response:
            self.assertEqual(json.load(response), changed | {'port': 8877})

        restored = self.post("/settings/reload", {})
        self.assertEqual(restored["settings"], changed)
        self.assertEqual(self.get_current(), changed)
        self.assertIsNone(self.display.take_update())

    def test_apply_persists_height_and_tilt_for_browser_refresh_and_restart(self):
        changed = sample_values(robot_z=0.37, tilt_rad=0.82, camera_base_height=0.95)
        self.display.take_update()
        self.post('/settings', changed)
        self.assertEqual(self.get_current(), changed)
        action, applied = self.display.take_update()
        self.assertEqual(action, 'connect')
        self.assertEqual((applied.robot_z, applied.tilt_rad), (0.37, 0.82))
        restarted = parse_args(['--settings-file', str(self.path)], dashboard=True)
        self.assertEqual((restarted.robot_z, restarted.tilt_rad), (0.37, 0.82))
        self.assertEqual(restarted.camera_base_height, 0.95)

    def test_live_pose_uses_applied_offsets_not_unapplied_saved_values(self):
        feedback = {'lift_fresh': True, 'lift_position': 0.2,
                    'head_fresh': True, 'head_pitch': 0.3}
        changed = sample_values(camera_base_height=1.1, robot_z=0.1, tilt_rad=0.2)
        with patch.object(self.display.robot_control, 'status', return_value=feedback):
            self.post('/settings/save', changed)
            with urlopen(self.display.url + '/robot/status', timeout=2) as response:
                pose = json.load(response)['camera_pose']
            self.assertAlmostEqual(pose['height'], 0.83 + 0.22 + 0.2)
            self.assertAlmostEqual(pose['tilt'], 0.4 + 0.3)
            self.post('/settings', changed)
            with urlopen(self.display.url + '/robot/status', timeout=2) as response:
                pose = json.load(response)['camera_pose']
            self.assertAlmostEqual(pose['height'], 1.4)
            self.assertAlmostEqual(pose['tilt'], 0.5)

    def test_failed_apply_does_not_change_form_file_or_camera(self):
        self.display.take_update()
        before = self.path.read_bytes()
        with patch('dashboard_server.save_settings', side_effect=PermissionError('read-only settings')):
            with self.assertRaises(HTTPError) as error:
                self.post('/settings', sample_values(robot_z=0.37, tilt_rad=0.82))
        self.assertEqual(error.exception.code, 500)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.get_current(), self.original)
        self.assertIsNone(self.display.take_update())

    def test_invalid_save_or_reload_preserves_form_file_and_pending_connection(self):
        self.display.take_update()
        previous_file = self.path.read_bytes()
        with self.assertRaises(HTTPError) as error:
            self.post("/settings/save", self.original | {"channel_difference_max": 300})
        self.assertEqual(error.exception.code, 400)
        self.assertEqual(self.path.read_bytes(), previous_file)
        self.assertEqual(self.get_current(), self.original)
        self.assertIsNone(self.display.take_update())

        for invalid_file in ("{broken json", json.dumps(self.original | {"port": 8877, "robot_z": -1})):
            self.path.write_text(invalid_file)
            with self.subTest(invalid_file=invalid_file), self.assertRaises(HTTPError):
                self.post("/settings/reload", {})
            self.assertEqual(self.get_current(), self.original)
            self.assertEqual(self.path.read_text(), invalid_file)
            self.assertIsNone(self.display.take_update())

        self.path.unlink()
        with self.assertRaises(HTTPError):
            self.post("/settings/reload", {})
        self.assertEqual(self.get_current(), self.original)
        self.assertFalse(self.path.exists())
        self.assertIsNone(self.display.take_update())
