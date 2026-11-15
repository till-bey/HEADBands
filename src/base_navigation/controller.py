"""Proportional frisbee approach controller, independent of ROS and the camera."""

from dataclasses import dataclass
import math
from collections.abc import Mapping


@dataclass(frozen=True)
class VelocityCommand:
    linear_speed: float = 0.0  # m/s, positive forward
    angular_speed: float = 0.0  # rad/s, positive left (ROS)
    state: str = "lost"


@dataclass(frozen=True)
class PController:
    """Steer and approach simultaneously using camera-relative target position.

    Input X points right and Y forward, in meters (FrisbeeCamera convention).
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

    def __post_init__(self):
        for name in ("angular_kp", "linear_kp", "max_angular_speed",
                     "max_linear_speed", "angle_tolerance", "distance_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.angle_tolerance >= math.pi / 2:
            raise ValueError("angle_tolerance must be less than pi/2")
        if not math.isfinite(self.target_distance) or self.target_distance < 0:
            raise ValueError("target_distance must be finite and nonnegative")

    def compute(self, position: Mapping[str, float] | None) -> VelocityCommand:
        """Return zero on missing/invalid input; never command reverse motion."""
        if position is None:
            return VelocityCommand()
        try:
            x, y = float(position["x"]), float(position["y"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return VelocityCommand()
        distance = math.hypot(x, y)
        if not all(math.isfinite(v) for v in (x, y, distance)):
            return VelocityCommand()
        # Bearing is unstable at the camera origin, where no approach is needed.
        if distance <= self.distance_tolerance:
            return VelocityCommand(state="reached")
        error = distance - self.target_distance
        bearing = math.atan2(x, y)
        angular = 0.0
        if abs(bearing) > self.angle_tolerance:
            # Camera bearing is positive right; ROS angular.z is positive left.
            angular = max(-self.max_angular_speed,
                          min(self.max_angular_speed, -self.angular_kp * bearing))
        linear = 0.0
        if error > self.distance_tolerance and y > 0:
            # Approach while steering, slowing as the target moves off-center.
            linear = min(self.max_linear_speed, self.linear_kp * error) * math.cos(bearing)
        state = ("tracking" if linear and angular else "approaching" if linear
                 else "aligning" if angular else "reached")
        return VelocityCommand(linear_speed=linear, angular_speed=angular, state=state)
