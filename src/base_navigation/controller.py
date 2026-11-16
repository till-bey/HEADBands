"""Proportional/derivative frisbee approach control, independent of ROS."""

from dataclasses import dataclass
import math
from collections.abc import Mapping


@dataclass(frozen=True)
class VelocityCommand:
    linear_speed: float = 0.0  # m/s, positive forward
    angular_speed: float = 0.0  # rad/s, positive left (ROS)
    state: str = "lost"
    distance_error: float | None = None  # distance - target_distance, meters
    angle_error: float | None = None  # radians; horizontal image angle when available


@dataclass(frozen=True)
class PController:
    """Steer and approach simultaneously using camera-relative target position.

    Input X points right and Y forward, in meters (FrisbeeCamera convention).
    Optional D gains use rates supplied by worker-owned ControllerHistory.
    Camera positions supply image_angle_rad for tilt-independent steering;
    ground-only callers fall back to atan2(x, y), wrapped inside target distance.
    Stop at target_distance (default 0.3 m), within distance_tolerance.
    The camera must face straight ahead,
    with its ground origin aligned with the desired robot stopping point.
    """

    angular_kp: float = 1.0
    linear_kp: float = 0.4
    max_angular_speed: float = 0.4
    max_linear_speed: float = 0.15
    angle_tolerance: float = 0.05  # radians
    distance_tolerance: float = 0.05  # meters
    target_distance: float = 0.3  # meters from the camera's ground origin
    angular_kd: float = 0.0
    linear_kd: float = 0.0

    def __post_init__(self):
        for name in ("angular_kd", "linear_kd"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("angular_kp", "linear_kp", "max_angular_speed",
                     "max_linear_speed", "angle_tolerance", "distance_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.angle_tolerance >= math.pi / 2:
            raise ValueError("angle_tolerance must be less than pi/2")
        if not math.isfinite(self.target_distance) or self.target_distance < 0:
            raise ValueError("target_distance must be finite and nonnegative")

    def compute(self, position: Mapping[str, float] | None, *,
                distance_rate: float = 0.0, bearing_rate: float = 0.0) -> VelocityCommand:
        """Regulate signed distance error; return zero on missing/invalid input."""
        if position is None:
            return VelocityCommand()
        try:
            x, y = float(position["x"]), float(position["y"])
            image_angle = (float(position["image_angle_rad"])
                           if "image_angle_rad" in position else None)
        except (KeyError, TypeError, ValueError, OverflowError):
            return VelocityCommand()
        distance = math.hypot(x, y)
        if (not all(math.isfinite(v) for v in (x, y, distance))
                or (image_angle is not None and not math.isfinite(image_angle))):
            return VelocityCommand()
        error = distance - self.target_distance
        distance_reached = self.distance_within_tolerance(error)
        holding_heading = error < 0 and not distance_reached
        # At the camera origin there is no stable bearing. Back straight away
        # when a nonzero stand-off is required, without introducing a turn.
        bearing_available = image_angle is not None or distance > self.distance_tolerance
        raw_bearing = math.atan2(x, y) if distance > 0 else 0.0
        bearing = image_angle if image_angle is not None else raw_bearing if bearing_available else 0.0
        if error < 0 and image_angle is None:
            # Once closer than the goal, align the axis rather than turning
            # around toward a target that has passed behind the stopping point.
            bearing = math.remainder(bearing, math.pi)
        angular = 0.0
        if not holding_heading and abs(bearing) > self.angle_tolerance:
            # Camera bearing is positive right; ROS angular.z is positive left.
            angular = max(-self.max_angular_speed,
                          min(self.max_angular_speed,
                              -self.angular_kp * bearing - self.angular_kd * bearing_rate))
        linear = 0.0
        if not distance_reached and (error < 0 or y > 0):
            radial_speed = self.linear_kp * error + self.linear_kd * distance_rate
            # D may brake to zero, but must not drive away from the distance goal.
            if error > 0:
                radial_speed = max(0.0, min(self.max_linear_speed, radial_speed))
            else:
                radial_speed = min(0.0, max(-self.max_linear_speed, radial_speed))
            # Use the geometric bearing here: increasing separation from a
            # close rear target requires forward motion, not reversing into it.
            linear = radial_speed * math.cos(raw_bearing)
        state = ("tracking" if linear and angular else "backing_up" if linear < 0
                 else "retreating" if linear and error < 0 else "approaching" if linear
                 else "aligning" if angular else "braking" if not distance_reached
                 else "reached")
        return VelocityCommand(linear_speed=linear, angular_speed=angular, state=state,
                               distance_error=error, angle_error=bearing if bearing_available else None)

    def distance_within_tolerance(self, error: float) -> bool:
        return (abs(error) <= self.distance_tolerance
                or math.isclose(abs(error), self.distance_tolerance, abs_tol=1e-12))


class ControllerHistory:
    """Worker-owned derivative history; preview/validation never changes it."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sample = None
        self._controller = None
        self._distance_rate = self._bearing_rate = 0.0
        self._inside_target = None
        self._holding_heading = False

    def compute(self, controller, position, observed_at, *, max_gap=0.5):
        baseline = controller.compute(position)
        if baseline.state == "lost" or not math.isfinite(observed_at):
            self.reset()
            return VelocityCommand()
        if baseline.angle_error is None:
            self.reset()
            return baseline
        distance = math.hypot(float(position["x"]), float(position["y"]))
        bearing = baseline.angle_error
        inside_target = distance < controller.target_distance and "image_angle_rad" not in position
        holding_heading = (baseline.distance_error < 0
                           and not controller.distance_within_tolerance(baseline.distance_error))
        if self._controller != controller or self._inside_target != inside_target:
            # Switching angular conventions must not create a derivative kick.
            self.reset()
        if self._sample is not None:
            previous_time, previous_distance, previous_bearing = self._sample
            dt = observed_at - previous_time
            if dt < 0 or dt > max_gap:
                self.reset()
            elif dt > 0:
                # First-order low-pass filter with a 0.15 s time constant.
                alpha = dt / (0.15 + dt)
                period = math.pi if inside_target else 2 * math.pi
                bearing_delta = math.remainder(bearing - previous_bearing, period)
                self._distance_rate += alpha * ((distance - previous_distance) / dt - self._distance_rate)
                self._bearing_rate += alpha * (bearing_delta / dt - self._bearing_rate)
        if holding_heading or self._holding_heading:
            # Ignore angular motion during straight distance correction, and
            # restart steering with zero D on the first sample after it ends.
            self._bearing_rate = 0.0
        self._sample = observed_at, distance, bearing
        self._controller = controller
        self._inside_target = inside_target
        self._holding_heading = holding_heading
        return controller.compute(position, distance_rate=self._distance_rate,
                                  bearing_rate=self._bearing_rate)
