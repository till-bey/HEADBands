"""phone_camera: publish the phone camera (MJPEG / RTSP / HTTP snapshot URL) or a video file as ROS 2 images.

The project had no ROS 2 node for the phone camera: perception/camera reads the phone stream in plain Python.
This node reuses that reader (perception/camera/camera_stream.py, LatestFrameCamera) instead of duplicating
it, so the frisbee nodes can subscribe to a normal sensor_msgs topic. With a video file it replays at the
file's frame rate (used for the recorded-video tests).

Publishes (QoS: sensor data, best effort, depth 2):
  ~/image_raw            sensor_msgs/Image (bgr8)                 if publish_raw
  ~/image_raw/compressed sensor_msgs/CompressedImage (jpeg)       if publish_compressed
  ~/camera_info          sensor_msgs/CameraInfo                   only if a calibration file is given
Parameters: source, frame_id, rotate (none|cw|ccw|180), fps_limit, calibration_file, perception_path, loop.
Stamps: capture time (now minus the reader's frame age) for live streams; frame index / fps for files.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Header

from frisbee_interception.ros_image import bgr_to_compressed, bgr_to_image

ROTATIONS = {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}


def _load_camera_info(path: str) -> CameraInfo | None:
    """Standard camera_calibration YAML (as written by `ros2 run camera_calibration cameracalibrator`)."""
    if not path:
        return None
    import yaml

    d = yaml.safe_load(Path(path).read_text())
    info = CameraInfo()
    info.width, info.height = int(d["image_width"]), int(d["image_height"])
    info.distortion_model = d.get("distortion_model", "plumb_bob")
    info.d = [float(x) for x in d["distortion_coefficients"]["data"]]
    info.k = [float(x) for x in d["camera_matrix"]["data"]]
    info.r = [float(x) for x in d.get("rectification_matrix", {"data": [1, 0, 0, 0, 1, 0, 0, 0, 1]})["data"]]
    info.p = [float(x) for x in d.get("projection_matrix", {"data": info.k[:3] + [0] + info.k[3:6] + [0] + info.k[6:] + [0]})["data"]]
    return info


class PhoneCamera(Node):
    def __init__(self) -> None:
        super().__init__("phone_camera")
        p = self.declare_parameter
        self.source = p("source", "").value
        self.frame_id = p("frame_id", "phone_camera_optical_frame").value
        self.rotate = p("rotate", "none").value
        self.fps_limit = float(p("fps_limit", 30.0).value)
        self.loop = bool(p("loop", False).value)
        self.publish_raw = bool(p("publish_raw", True).value)
        self.publish_compressed = bool(p("publish_compressed", False).value)
        self.jpeg_quality = int(p("jpeg_quality", 80).value)
        perception_path = p("perception_path", "").value
        self.info = _load_camera_info(p("calibration_file", "").value)
        if not self.source:
            raise RuntimeError("Set the 'source' parameter: a phone stream URL or a video file path.")

        self.raw_pub = self.create_publisher(Image, "~/image_raw", qos_profile_sensor_data) if self.publish_raw else None
        self.comp_pub = (
            self.create_publisher(CompressedImage, "~/image_raw/compressed", qos_profile_sensor_data)
            if self.publish_compressed else None
        )
        self.info_pub = self.create_publisher(CameraInfo, "~/camera_info", qos_profile_sensor_data) if self.info else None

        self.is_file = os.path.isfile(self.source)
        self.reader = None
        self.cap = None
        if self.is_file:
            self.cap = cv2.VideoCapture(self.source)
            self.file_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.frame_index = 0
            self.t0 = self.get_clock().now()
            period = 1.0 / self.file_fps
        else:
            self.reader = self._open_stream(perception_path)
            period = 1.0 / max(1.0, self.fps_limit)
        self.last_seq = -1
        self.timer = self.create_timer(period, self._tick)
        self.get_logger().info(f"phone_camera: {'file' if self.is_file else 'stream'} -> {self.get_name()}/image_raw")

    def _open_stream(self, perception_path: str):
        """Reuse perception/camera/camera_stream.py (LatestFrameCamera); plain OpenCV only if it is missing."""
        candidates = [perception_path] if perception_path else []
        candidates += [str(Path(__file__).resolve().parents[i] / "perception") for i in range(2, 6)]
        candidates += ["/workspace/perception"]
        for c in candidates:
            if c and (Path(c) / "camera" / "camera_stream.py").is_file():
                sys.path.insert(0, c)
                from camera.camera_stream import LatestFrameCamera  # noqa: PLC0415

                self.get_logger().info(f"using perception/camera stream reader from {c}")
                return LatestFrameCamera(self.source).start()
        self.get_logger().warn("perception/camera not found: falling back to cv2.VideoCapture")
        self.cap = cv2.VideoCapture(self.source)
        return None

    def _publish(self, frame, stamp) -> None:
        if self.rotate in ROTATIONS:
            frame = cv2.rotate(frame, ROTATIONS[self.rotate])
        header = Header()
        header.stamp = stamp.to_msg()
        header.frame_id = self.frame_id
        if self.raw_pub is not None:
            self.raw_pub.publish(bgr_to_image(frame, header, Image))
        if self.comp_pub is not None:
            self.comp_pub.publish(bgr_to_compressed(frame, header, CompressedImage, self.jpeg_quality))
        if self.info_pub is not None:
            self.info.header = header
            self.info_pub.publish(self.info)

    def _tick(self) -> None:
        if self.is_file:
            # Real-time replay: if decoding fell behind the wall clock, skip frames so stamps never go stale.
            elapsed = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
            while (self.frame_index + 1) / self.file_fps < elapsed and self.cap.grab():
                self.frame_index += 1
            ok, frame = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.t0 = self.get_clock().now()
                    self.frame_index = 0
                    return
                self.get_logger().info("end of video file")
                self.timer.cancel()
                return
            stamp = self.t0 + Duration(seconds=self.frame_index / self.file_fps)
            self.frame_index += 1
            self._publish(frame, stamp)
            return
        if self.reader is None:
            ok, frame = self.cap.read()
            if ok:
                self._publish(frame, self.get_clock().now())
            return
        snap = self.reader.get_latest()
        if snap.frame is None or snap.sequence == self.last_seq:
            return
        self.last_seq = snap.sequence
        age = snap.age_seconds or 0.0
        stamp = self.get_clock().now() - Duration(seconds=age)
        self._publish(snap.frame, stamp)

    def destroy_node(self) -> None:
        if self.reader is not None:
            self.reader.stop()
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PhoneCamera()
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
