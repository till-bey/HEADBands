"""Online camera setup and background ground-position acquisition (no ROS)."""

from threading import Event, Lock, Thread
import time

from .camera_stream import SNAPSHOT, VIDEO_STREAM
from .frisbee_camera import CAMERA_BASE_HEIGHT_M, FrisbeeCamera
from detector import DetectorSettings

def create_camera(args):
    return FrisbeeCamera(
        args.url, DetectorSettings(
            canny_low=args.canny_low, canny_high=args.canny_high,
            processing_width=args.resolution,
            rgb_sum_min=args.rgb_sum_min, channel_difference_max=args.channel_difference_max,
            min_radius=args.min_radius, max_radius=args.max_radius,
            min_ellipse_ratio=args.min_ellipse_ratio, min_area=args.min_area,
            min_white_fraction=args.min_white_fraction,
        ),
        mode=SNAPSHOT if args.snapshot else VIDEO_STREAM,
    )


class CameraTargetReader:
    """Keep camera waits/inference outside the ROS executor's stop timer.

    A timestamp taken BEFORE acquisition also makes slow inference expire.
    A failed read replaces the previous target with None immediately.
    """

    def __init__(self, camera, *, robot_z, tilt_rad, horizontal_fov_deg,
                 vertical_fov_deg=None, camera_base_height_m=CAMERA_BASE_HEIGHT_M):
        self.camera = camera
        self.geometry = dict(robot_z=robot_z, tilt_rad=tilt_rad,
                             horizontal_fov_deg=horizontal_fov_deg,
                             vertical_fov_deg=vertical_fov_deg)
        self.geometry['camera_base_height_m'] = camera_base_height_m
        self._stop = Event()
        self._lock = Lock()
        self._result = (None, 0.0, "Waiting for camera")
        self._thread = Thread(target=self._run, daemon=True, name="navigation-detector")

    def start(self):
        self._thread.start()

    def snapshot(self):
        with self._lock:
            return self._result

    def update_geometry(self, *, robot_z, tilt_rad, camera_base_height_m):
        with self._lock:
            self.geometry.update(robot_z=robot_z, tilt_rad=tilt_rad,
                                 camera_base_height_m=camera_base_height_m)

    def close(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=1.0)
        else:
            self.camera.close()
        # The worker owns camera.close(), avoiding concurrent camera use.

    def _run(self):
        try:
            with self.camera:
                while not self._stop.is_set():
                    observed_at = time.monotonic()
                    try:
                        with self._lock:
                            geometry = self.geometry.copy()
                        position = self.camera.get_robot_ground_position(
                            **geometry, timeout=0.2)
                        status = "Tracking" if position is not None else "No frisbee detected"
                    except (TimeoutError, ValueError) as exc:
                        position, status = None, str(exc)
                    with self._lock:
                        self._result = position, observed_at, status
                    self._stop.wait(0.01)
        except Exception as exc:
            # Unexpected detector/camera failures must invalidate the old target.
            with self._lock:
                self._result = None, time.monotonic(), f"Camera failed: {exc}"

