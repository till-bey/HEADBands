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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "perception"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SETTINGS_FIELDS, validate_settings, load_settings, parse_args, save_settings
from dashboard_server import DashboardServer


def sample_values(**changes):
    values = {
        "url": "http://127.0.0.1:8080/camera.jpg",
        "snapshot": True,
        "robot_z": 0.22,
        "tilt_rad": 0.4,
        "horizontal_fov": 77.32,
        "vertical_fov": 61.93,
        "resolution": 480,
        "rgb_sum_min": 160,
        "channel_difference_max": 55,
        "angular_kp": 0.9,
        "linear_kp": 0.3,
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
        ], dashboard=True)
        self.assertEqual((args.url, args.snapshot, args.rgb_sum_min, args.linear_kp),
                         ("http://127.0.0.1:8081/stream", False, 190, 0.5))
        self.assertEqual((args.channel_difference_max, args.robot_z, args.port),
                         (55, 0.22, 8877))
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
                        {"linear_kp": float("nan")}, {"snapshot": "yes"},
                        {"horizontal_fov": 180}, {"url": "not-a-camera-url"}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError, OverflowError)):
                save_settings(self.path, values | changes)
            self.assertEqual(self.path.read_bytes(), before)


class BrowserSettingsPersistenceTests(unittest.TestCase):
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
                                rgb_sum_min=190, channel_difference_max=65,
                                max_linear_speed=0.08)
        self.post("/settings/save", changed)
        self.assertEqual(json.loads(self.path.read_text()), changed | {"port": 8877})
        self.assertEqual(self.get_current(), self.original)
        # Saving must not replace or consume the startup connection request.
        action, pending = self.display.take_update()
        self.assertEqual((action, pending.url), ("connect", self.original["url"]))
        self.assertIsNone(self.display.take_update())

        restored = self.post("/settings/reload", {})
        self.assertEqual(restored["settings"], changed)
        self.assertEqual(self.get_current(), changed)
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
