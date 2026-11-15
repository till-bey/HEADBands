"""Normalize standard ROS ultrasound ranges into JSON-safe proximity readings."""

import math
import re
import time

SIDES = ('front', 'right', 'rear', 'left')


def suggested_side(topic, frame_id):
    words = set(re.findall(r'[a-z]+', f'{topic} {frame_id}'.lower()))
    aliases = {'front': {'front', 'forward'}, 'rear': {'rear', 'back', 'backward'},
               'left': {'left'}, 'right': {'right'}}
    matches = [side for side, names in aliases.items() if words & names]
    # Numbered sensors and corner names need explicit placement by the operator.
    return matches[0] if len(matches) == 1 else None


class UltrasoundReadings:
    def __init__(self, clock=time.monotonic, timeout=1.0):
        self._clock, self._timeout = clock, timeout
        self._readings = {}

    def update(self, topic, message):
        if message.radiation_type != 0:  # sensor_msgs/Range.ULTRASOUND
            self._readings.pop(topic, None)
            return
        distance = float(message.range)
        minimum, maximum = float(message.min_range), float(message.max_range)
        status, value = 'invalid', None
        valid_bounds = (math.isfinite(minimum) and math.isfinite(maximum)
                        and 0 <= minimum <= maximum)
        if valid_bounds:
            if distance == -math.inf:
                status, value = 'too_close', minimum
            elif distance == math.inf:
                status, value = 'beyond_range', maximum
            elif math.isfinite(distance) and minimum <= distance <= maximum:
                status, value = 'valid', distance
        frame_id = message.header.frame_id
        self._readings[topic] = {
            'topic': topic, 'frame_id': frame_id, 'side': suggested_side(topic, frame_id),
            'distance_m': value, 'status': status, 'received_at': self._clock(),
        }

    def snapshot(self):
        now = self._clock()
        result = []
        for reading in self._readings.values():
            item = reading.copy()
            item['age_seconds'] = max(0, now - item.pop('received_at'))
            item['fresh'] = item['age_seconds'] <= self._timeout
            if not item['fresh']:
                item['status'], item['distance_m'] = 'stale', None
            result.append(item)
        return sorted(result, key=lambda item: item['topic'])
