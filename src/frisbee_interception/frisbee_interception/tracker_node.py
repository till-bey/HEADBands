"""frisbee_tracker: follow the frisbee across frames with a lightweight constant-velocity Kalman filter.

Subscribes:  /frisbee/detections   frisbee_msgs/FrisbeeDetectionArray
Publishes:   /frisbee/track        frisbee_msgs/FrisbeeTrack (one per detection message, stamp = image stamp)
Time steps use the message stamps, never an assumed frame rate. Isolated or static false detections are not
confirmed (see core/tracking.py); short detection gaps are bridged by prediction (state 'coasting').
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node

from frisbee_msgs.msg import FrisbeeDetectionArray, FrisbeeTrack
from frisbee_interception.core.tracking import CONFIRMED, COASTING, FrisbeeTracker, Measurement, TrackerConfig
from frisbee_interception.params import declare_section


def stamp_to_s(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class TrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("frisbee_tracker")
        cfg = TrackerConfig.from_dict(declare_section(self, "tracking", TrackerConfig()))
        self.tracker = FrisbeeTracker(cfg)
        self.last_t: float | None = None
        self.create_subscription(FrisbeeDetectionArray, "/frisbee/detections", self._on_detections, 10)
        self.pub = self.create_publisher(FrisbeeTrack, "/frisbee/track", 10)

    def _on_detections(self, msg: FrisbeeDetectionArray) -> None:
        t = stamp_to_s(msg.header.stamp)
        if self.last_t is not None and t < self.last_t:
            self.get_logger().warn("detections went back in time (bag loop or clock jump): resetting the tracker")
            self.tracker.reset()
        self.last_t = t
        meas = [Measurement(t, d.u, d.v, d.major_px, float(d.confidence)) for d in msg.detections]
        state = self.tracker.step(t, meas)
        out = FrisbeeTrack()
        out.header = msg.header
        out.state = state
        out.image_width, out.image_height = msg.image_width, msg.image_height
        est = self.tracker.estimate(t)
        if est is not None:
            out.track_id = int(est["track_id"])
            out.confirmed = state in (CONFIRMED, COASTING)
            out.u, out.v, out.d = est["u"], est["v"], est["d"]
            out.du, out.dv, out.dd = est["du"], est["dv"], est["dd"]
            out.sigma_uv = est["sigma_uv"]
            hist = list(self.tracker.history)
            out.hist_t = [m.t for m in hist]
            out.hist_u = [m.u for m in hist]
            out.hist_v = [m.v for m in hist]
            out.hist_d = [m.d for m in hist]
            out.hist_confidence = [float(m.confidence) for m in hist]
        self.pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
