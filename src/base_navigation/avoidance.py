"""Small, ROS-independent potential field for ultrasound-assisted tracking."""

from dataclasses import dataclass
import math

if __package__:
    from .controller import VelocityCommand
    from .ultrasound import SIDES
else:
    from controller import VelocityCommand
    from ultrasound import SIDES

INFLUENCE_M = 0.60
STOP_M = 0.20
REPULSION_GAIN = 0.008


def validate_mapping(mapping):
    if not isinstance(mapping, dict) or set(mapping) - set(SIDES):
        raise ValueError('Sensor mapping must name front, right, rear or left')
    result = {}
    for side, topics in mapping.items():
        if (not isinstance(topics, list) or len(topics) > 32
                or any(not isinstance(topic, str) or not topic.startswith('/')
                       or len(topic) > 256 for topic in topics)
                or len(set(topics)) != len(topics)):
            raise ValueError('Each sensor side must contain a list of unique ROS topic names')
        result[side] = topics.copy()
    return result


def sensor_distances(readings, mapping):
    """Require complete, fresh coverage; never interpret a missing range as free."""
    by_topic = {reading['topic']: reading for reading in readings}
    distances, used = {}, set()
    for side in SIDES:
        topics = mapping.get(side, [r['topic'] for r in readings if r.get('side') == side])
        if not topics:
            return None, f'Missing ultrasound coverage: {side}'
        values = []
        for topic in topics:
            reading = by_topic.get(topic)
            if reading is None or not reading.get('fresh'):
                return None, f'Missing or stale ultrasound: {topic}'
            distance = reading.get('distance_m')
            status = reading.get('status')
            if (not isinstance(distance, (int, float)) or not math.isfinite(distance)
                    or distance < 0 or status not in ('valid', 'too_close', 'beyond_range')):
                return None, f'Invalid ultrasound: {topic}'
            if status == 'beyond_range' and distance < INFLUENCE_M:
                return None, f'Insufficient ultrasound detection range: {topic}'
            values.append(0.0 if status == 'too_close' else distance)
            used.add(topic)
        distances[side] = min(values)
    if set(by_topic) - used:
        return None, 'Assign all ultrasound sensors to their physical sides'
    return distances, 'Ultrasound coverage ready'


@dataclass(frozen=True)
class PotentialFieldController:
    controller: object
    readings: list
    mapping: dict

    def compute(self, position):
        nominal = self.controller.compute(position)
        if nominal.state == 'lost':
            return nominal  # Repulsion cannot initiate motion without a target.
        distances, _ = sensor_distances(self.readings, self.mapping)
        if distances is None:
            return VelocityCommand(state='ultrasound_unavailable')
        if min(distances.values()) <= STOP_M:
            return VelocityCommand(state='obstacle_stop')
        if not (nominal.linear_speed or nominal.angular_speed):
            return nominal  # Do not leave an already-reached tracking position.
        if min(distances.values()) >= INFLUENCE_M:
            return nominal

        x, y = float(position['x']), float(position['y'])
        bearing = math.atan2(x, y)
        attraction = max(0.05, nominal.linear_speed)
        force_x, force_y = attraction * math.sin(bearing), attraction * math.cos(bearing)
        # Camera coordinates: X right, Y forward. Repulsion points away from each side.
        directions = {'front': (0, 1), 'rear': (0, -1), 'right': (1, 0), 'left': (-1, 0)}
        for side, distance in distances.items():
            if distance >= INFLUENCE_M:
                continue
            strength = REPULSION_GAIN * (1 / distance - 1 / INFLUENCE_M) / distance ** 2
            dx, dy = directions[side]
            force_x -= strength * dx
            force_y -= strength * dy
        if force_y <= 0 or math.hypot(force_x, force_y) < 1e-6:
            # A symmetric obstacle can create a potential-field local minimum.
            # Stop rather than reverse or choose an arbitrary escape direction.
            return VelocityCommand(state='obstacle_stop')
        heading = math.atan2(force_x, force_y)
        angular = 0.0 if abs(heading) <= self.controller.angle_tolerance else max(
            -self.controller.max_angular_speed,
            min(self.controller.max_angular_speed, -self.controller.angular_kp * heading))
        linear = min(nominal.linear_speed, max(0, force_y)) * max(0, math.cos(heading))
        return VelocityCommand(linear_speed=linear, angular_speed=angular, state='avoiding')
