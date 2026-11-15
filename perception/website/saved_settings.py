"""Read the navigation dashboard's saved configuration without starting ROS."""

import json
import shlex
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.request import urlopen

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.base_navigation.config import DEFAULT_SETTINGS_FILE, load_settings, settings_from_values

DEFAULT_DASHBOARD_URL = 'http://127.0.0.1:8765'


def load_navigation_settings(path=DEFAULT_SETTINGS_FILE, dashboard_url=DEFAULT_DASHBOARD_URL):
    """Prefer the saved file; use the server for cross-user file permissions."""
    path = Path(path).expanduser().resolve()
    try:
        return vars(load_settings(path)), str(path)
    except OSError as file_error:
        if path != DEFAULT_SETTINGS_FILE.resolve():
            raise ValueError(f'Could not read saved configuration: {file_error}') from file_error
        try:
            with urlopen(dashboard_url.rstrip('/') + '/settings/saved', timeout=1.5) as response:
                values = json.load(response)
            return vars(settings_from_values(values)), 'Navigation dashboard saved-settings endpoint'
        except (OSError, URLError, ValueError, TypeError, OverflowError) as server_error:
            repair = ''
            if isinstance(file_error, PermissionError):
                quoted_path = shlex.quote(str(path))
                repair = (
                    ' To repair ownership, run this in your terminal as your normal user: '
                    f'`sudo chown "$(id -u):$(id -g)" {quoted_path} '
                    f'&& chmod u+rw {quoted_path}`. Then reload the configuration.'
                )
            raise ValueError(
                f'Could not read the saved file ({file_error}) or the navigation dashboard. '
                'Run both websites as the same user, or start the navigation dashboard and reload.'
                + repair
            ) from server_error
