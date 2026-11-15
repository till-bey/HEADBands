"""frisbee_detector: find the frisbee in each camera image (OpenCV only, no GPU, no neural network).

Subscribes (QoS sensor data):
  image (remap)  sensor_msgs/Image, or sensor_msgs/CompressedImage when image_transport = compressed
Publishes:
  /frisbee/detections         frisbee_msgs/FrisbeeDetectionArray  (header = image header: stamp + camera frame)
  /frisbee/debug/detections   sensor_msgs/Image (bgr8)            only drawn while someone subscribes
Parameters: image_transport (raw|compressed), publish_debug, max_debug_width, detection.* (see config).
Old images are not skipped silently: with depth-1 best-effort QoS the newest frame always wins, and every
detection carries the image stamp so downstream nodes reject stale data.
"""

from __future__ import annotations

import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from frisbee_msgs.msg import FrisbeeDetection, FrisbeeDetectionArray
from frisbee_interception.core.detection import DetectorConfig, FrisbeeDetector, draw_detections
from frisbee_interception.params import declare_section
from frisbee_interception.ros_image import bgr_to_image, compressed_to_bgr, image_to_bgr


class DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("frisbee_detector")
        transport = self.declare_parameter("image_transport", "raw").value
        self.publish_debug = bool(self.declare_parameter("publish_debug", True).value)
        self.max_debug_width = int(self.declare_parameter("max_debug_width", 640).value)
        cfg = DetectorConfig.from_dict(declare_section(self, "detection", DetectorConfig()))
        self.detector = FrisbeeDetector(cfg)
        self.get_logger().info(f"detection method={cfg.method}, processing width={cfg.processing_width}")

        qos_in = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        if transport == "compressed":
            self.create_subscription(CompressedImage, "image/compressed", self._on_compressed, qos_in)
        else:
            self.create_subscription(Image, "image", self._on_image, qos_in)
        self.det_pub = self.create_publisher(FrisbeeDetectionArray, "/frisbee/detections", 10)
        self.dbg_pub = self.create_publisher(Image, "/frisbee/debug/detections", qos_profile_sensor_data)
        self._ms = 0.0
        self._n = 0
        self.create_timer(10.0, self._report)

    def _on_image(self, msg: Image) -> None:
        try:
            self._process(image_to_bgr(msg), msg.header)
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)

    def _on_compressed(self, msg: CompressedImage) -> None:
        try:
            self._process(compressed_to_bgr(msg), msg.header)
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)

    def _process(self, bgr, header) -> None:
        t0 = time.perf_counter()
        dets = self.detector.detect(bgr)
        self._ms += (time.perf_counter() - t0) * 1000
        self._n += 1
        out = FrisbeeDetectionArray()
        out.header = header
        out.image_height, out.image_width = bgr.shape[:2]
        for d in dets:
            m = FrisbeeDetection()
            m.u, m.v = d.u, d.v
            m.bbox_x, m.bbox_y, m.bbox_w, m.bbox_h = d.bbox
            m.major_px, m.minor_px, m.angle_deg = d.major_px, d.minor_px, d.angle_deg
            m.confidence = float(d.confidence)
            out.detections.append(m)
        self.det_pub.publish(out)
        if self.publish_debug and self.dbg_pub.get_subscription_count() > 0:
            img = draw_detections(bgr, dets)
            if img.shape[1] > self.max_debug_width:
                s = self.max_debug_width / img.shape[1]
                img = cv2.resize(img, (self.max_debug_width, int(img.shape[0] * s)))
            self.dbg_pub.publish(bgr_to_image(img, header, Image))

    def _report(self) -> None:
        if self._n:
            self.get_logger().info(f"{self._n} images, {self._ms / self._n:.1f} ms/image detection")
            self._ms, self._n = 0.0, 0


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectorNode()
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
