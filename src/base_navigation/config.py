"""Shared command-line options for navigation and the browser dashboard."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

if __package__:
    from .controller import PController
else:
    from controller import PController

DEFAULT_SETTINGS_FILE = Path(__file__).with_name("settings.json")
ELLIPSE_DEFAULTS = {
    "min_radius": 10, "max_radius": 200, "min_ellipse_ratio": 0.65,
    "min_area": 150, "min_white_fraction": 0.48,
}


def parse_args(argv=None, *, description="Track a frisbee with simultaneous steering and distance P control.", dashboard=False):
    source_parser = argparse.ArgumentParser(add_help=False)
    source_parser.add_argument("--settings-file", type=Path, default=DEFAULT_SETTINGS_FILE)
    source_args, _ = source_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--settings-file", type=Path, default=source_args.settings_file,
                        help="JSON settings file (default: settings.json next to this script)")
    parser.add_argument("url", nargs="?", default="",
                        help="Direct camera URL (overrides saved URL)")
    parser.add_argument("--snapshot", action=argparse.BooleanOptionalAction)
    parser.add_argument("--robot-z", type=float, help="Mounting height offset added to base height and measured body lift (m)")
    parser.add_argument("--camera-base-height", type=float, help="Camera base height at zero lift (m; default: %(default)s)")
    parser.add_argument("--tilt-rad", type=float,
                        help="Camera mounting tilt offset added to measured head pitch (rad)")
    parser.add_argument("--horizontal-fov", type=float, help="Original image horizontal FOV (deg; default: %(default)s)")
    parser.add_argument("--vertical-fov", type=float, help="Original image vertical FOV (deg; default: %(default)s)")
    parser.add_argument("--resolution", type=int, choices=[320, 480, 640, 960, 1280])
    parser.add_argument("--rgb-sum-min", type=int,
                        help="Minimum R + G + B (0–765; default: %(default)s)")
    parser.add_argument("--channel-difference-max", type=int,
                        help="Maximum channel delta, max(R,G,B) - min(R,G,B) (0–255; default: %(default)s)")
    parser.add_argument("--min-radius", type=int, help="Minimum ellipse radius (processing pixels)")
    parser.add_argument("--max-radius", type=int, help="Maximum ellipse radius (processing pixels)")
    parser.add_argument("--min-ellipse-ratio", type=float, help="Minimum minor/major axis ratio (0–1)")
    parser.add_argument("--min-area", type=int, help="Minimum contour area (processing pixels squared)")
    parser.add_argument("--min-white-fraction", type=float, help="Minimum white fill fraction (0–1)")
    parser.add_argument("--angular-kp", type=float)
    parser.add_argument("--linear-kp", type=float)
    parser.add_argument("--max-angular-speed", type=float, help="rad/s")
    parser.add_argument("--max-linear-speed", type=float, help="m/s")
    parser.add_argument("--angle-tolerance", type=float, help="rad")
    parser.add_argument("--distance-tolerance", type=float, help="m")
    parser.add_argument("--target-distance", type=float, help="Tracking target distance (m; default: %(default)s)")
    parser.add_argument("--target-timeout", type=float, help="Expire detector results after this many seconds")
    if dashboard:
        parser.add_argument("--port", type=int, help="Local dashboard port (default: %(default)s)")
    try:
        parser.set_defaults(**vars(load_settings(source_args.settings_file)))
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        # Help must remain available when repairing an invalid settings file.
        if not any(flag in (argv if argv is not None else sys.argv[1:])
                   for flag in ("-h", "--help")):
            parser.error(f"Could not load settings from {source_args.settings_file}: {exc}")
    args = parser.parse_args(argv)
    try:
        validated = validate_settings({name: getattr(args, name) for name in SETTINGS_FIELDS},
                                      allow_empty_url=dashboard)
        vars(args).update(vars(validated))
        _validate_port(args.port)
    except (ValueError, TypeError, OverflowError) as exc:
        parser.error(str(exc))
    args.settings_file = args.settings_file.expanduser().resolve()
    return args


def validate_args(args):
    if not math.isfinite(args.camera_base_height) or args.camera_base_height <= 0:
        raise ValueError("Camera base height must be finite and positive")
    if not 1 <= args.min_radius < args.max_radius:
        raise ValueError("Ellipse radii must satisfy 1 <= minimum < maximum")
    if args.min_area < 1:
        raise ValueError("Minimum ellipse area must be positive")
    for name in ("min_ellipse_ratio", "min_white_fraction"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be greater than 0 and at most 1")
    for name, maximum in (("rgb_sum_min", 765), ("channel_difference_max", 255)):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
            raise ValueError(f"{name} must be an integer between 0 and {maximum}")
    if not math.isfinite(args.robot_z) or args.robot_z < 0:
        raise ValueError("Robot lift must be finite and nonnegative")
    if not math.isfinite(args.tilt_rad):
        raise ValueError("Camera tilt must be finite")
    for fov in (args.horizontal_fov, args.vertical_fov):
        if fov is not None and not (math.isfinite(fov) and 0 < fov < 180):
            raise ValueError("FOV must be between 0 and 180 degrees")
    if not math.isfinite(args.target_timeout) or args.target_timeout <= 0:
        raise ValueError("Target timeout must be finite and positive")
    create_controller(args)


def create_controller(args):
    return PController(**{name: getattr(args, name) for name in (
        "angular_kp", "linear_kp", "max_angular_speed", "max_linear_speed",
        "angle_tolerance", "distance_tolerance", "target_distance")})


SETTINGS_FIELDS = {
    "camera_base_height": float,
    "url": str, "snapshot": bool, "robot_z": float, "tilt_rad": float,
    "horizontal_fov": float, "vertical_fov": float, "resolution": int,
    "angular_kp": float, "linear_kp": float,
    "rgb_sum_min": int, "channel_difference_max": int,
    "min_radius": int, "max_radius": int, "min_ellipse_ratio": float,
    "min_area": int, "min_white_fraction": float,
    "max_angular_speed": float, "max_linear_speed": float,
    "angle_tolerance": float, "distance_tolerance": float,
    "target_distance": float, "target_timeout": float,
}


def validate_settings(values, *, allow_empty_url=False):
    """Validate a complete browser form before replacing the active camera."""
    if not isinstance(values, dict) or set(values) != set(SETTINGS_FIELDS):
        raise ValueError("Send all camera and controller settings")
    converted = {}
    for name, kind in SETTINGS_FIELDS.items():
        value = values[name]
        if kind is bool and not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false")
        if kind is str and not isinstance(value, str):
            raise ValueError(f"{name} must be text")
        if kind is int and (isinstance(value, bool) or float(value) != int(value)):
            raise ValueError(f"{name} must be an integer")
        converted[name] = kind(value)
    args = SimpleNamespace(**converted)
    validate_args(args)
    if args.resolution not in (320, 480, 640, 960, 1280):
        raise ValueError("Unsupported processing resolution")
    args.url = args.url.strip()
    if args.url or not allow_empty_url:
        from camera.camera_stream import validate_source, SNAPSHOT, VIDEO_STREAM
        args.url = validate_source(args.url, SNAPSHOT if args.snapshot else VIDEO_STREAM)
    return args


def _validate_port(port):
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("Port must be an integer between 1 and 65535")


def load_settings(path):
    """Read a complete, validated JSON configuration without changing runtime state."""
    with Path(path).expanduser().open(encoding="utf-8") as source:
        values = json.load(source)
    return settings_from_values(values)


def settings_from_values(values):
    """Validate saved settings from a file or the dashboard's read-only endpoint."""
    if isinstance(values, dict):
        values = values.copy()
        # Older saved files include a detector selector; detection is now ellipse-only.
        values.pop("white_detector", None)
        values.setdefault("camera_base_height", 0.83)
        # Retired edge-detector fields are ignored when reading older files.
        values.pop("canny_low", None)
        values.pop("canny_high", None)
        # Preserve previous detector behavior when opening older saved settings.
        for name, default in ELLIPSE_DEFAULTS.items():
            values.setdefault(name, default)
    if not isinstance(values, dict) or set(values) != set(SETTINGS_FIELDS) | {"port"}:
        raise ValueError("Settings file must contain all camera/controller fields and port")
    settings = validate_settings({name: values[name] for name in SETTINGS_FIELDS}, allow_empty_url=True)
    _validate_port(values["port"])
    settings.port = values["port"]
    return settings


def save_settings(path, values, *, port=8765):
    """Validate first, then replace the JSON file atomically on explicit Save."""
    settings = validate_settings(values, allow_empty_url=True)
    _validate_port(port)
    settings.port = port
    destination = Path(path).expanduser()
    # Atomic replacement must preserve a shared file's access permissions;
    # NamedTemporaryFile otherwise silently changes them to owner-only (0600).
    existing = destination.stat() if destination.exists() else None
    mode = existing.st_mode & 0o777 if existing else 0o600
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                         prefix=f".{destination.name}.", delete=False) as output:
            temporary = Path(output.name)
            json.dump(vars(settings), output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            if existing:
                temporary_stat = os.fstat(output.fileno())
                if (temporary_stat.st_uid, temporary_stat.st_gid) != (existing.st_uid, existing.st_gid):
                    try:
                        os.fchown(output.fileno(), existing.st_uid, existing.st_gid)
                    except PermissionError as exc:
                        raise PermissionError(
                            "Cannot preserve settings-file ownership. Run both websites as the "
                            "file's owner; the existing configuration was left unchanged."
                        ) from exc
            os.fchmod(output.fileno(), mode)
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return settings
