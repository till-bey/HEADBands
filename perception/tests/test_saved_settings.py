"""Verify the Streamlit saved-config reader without ROS or a live dashboard."""

import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'perception'))
sys.path.insert(0, str(ROOT / 'src/base_navigation/tests'))
from test_settings import sample_values
from website.saved_settings import load_navigation_settings
from src.base_navigation.config import save_settings


class SavedSettingsTests(unittest.TestCase):
    def test_reads_camera_calibration_and_detector_values_from_shared_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.json'
            values = sample_values(camera_base_height=0.95, tilt_rad=0.72, min_radius=3,
                                   max_radius=400, min_ellipse_ratio=0.06)
            save_settings(path, values)
            loaded, source = load_navigation_settings(path)
            self.assertEqual(loaded, values | {'port': 8765})
            self.assertEqual(source, str(path))

    def test_permission_failure_uses_read_only_saved_endpoint(self):
        values = sample_values() | {'port': 8765}
        response = io.BytesIO(json.dumps(values).encode())
        with patch('website.saved_settings.load_settings', side_effect=PermissionError('other user')), \
             patch('website.saved_settings.urlopen', return_value=response) as request:
            loaded, source = load_navigation_settings()
        self.assertEqual(loaded, values)
        self.assertIn('saved-settings endpoint', source)
        self.assertTrue(request.call_args.args[0].endswith('/settings/saved'))

    def test_unavailable_config_is_reported_instead_of_claiming_loaded_defaults(self):
        with patch('website.saved_settings.load_settings', side_effect=PermissionError('other user')), \
             patch('website.saved_settings.urlopen', side_effect=ConnectionError('offline')):
            with self.assertRaisesRegex(ValueError, 'Run both websites as the same user'):
                load_navigation_settings()

    def test_permission_error_includes_actionable_ownership_repair(self):
        with patch('website.saved_settings.load_settings', side_effect=PermissionError('other user')), \
             patch('website.saved_settings.urlopen', side_effect=ConnectionError('offline')):
            with self.assertRaises(ValueError) as error:
                load_navigation_settings()
        self.assertIn('sudo chown', str(error.exception))
        self.assertIn('chmod u+rw', str(error.exception))
