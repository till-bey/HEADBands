"""frisbee_interceptor: predict where and when the frisbee crosses the catching plane (prediction-only by default).

Subscribes:
  /frisbee/track             frisbee_msgs/FrisbeeTrack
  camera_info (remap)        sensor_msgs/CameraInfo     optional: intrinsics for metric mode
  image (remap)              sensor_msgs/Image          optional: only for the debug overlay
                             (image/compressed, sensor_msgs/CompressedImage, when image_transport = compressed)
  /frisbee/disable           std_msgs/Empty             disarm automatic interception immediately
Publishes:
  /frisbee/intercept         frisbee_msgs/FrisbeeIntercept   every track update + 10 Hz watchdog when data stops
  /frisbee/catch_point       geometry_msgs/PointStamped      valid predictions only (display / logging)
  /frisbee/markers           visualization_msgs/MarkerArray  trajectory, catch point, catching plane, text
  /frisbee/debug/intercept   sensor_msgs/Image               overlay, only while someone subscribes
  /frisbee/catch_target      geometry_msgs/PointStamped      ONLY when armed and every safety check passes
Services:
  ~/enable_interception      std_srvs/SetBool                arm (true) / disarm (false)

Modes (no calibration is invented; see docs/frisbee_interception.md):
  metric - camera intrinsics (CameraInfo or camera.* parameters) + TF from the camera frame to target_frame
           + catch_plane.point/normal in target_frame: 3D straight line -> plane intersection.
  image  - prediction.catch_diameter_px calibrated; optional pixel -> catch-plane homography
           (image_calibration.*) turns the predicted pixel into a point in target_frame.
A catch target is never sent from stale data, an unconfirmed track, an invalid or low-confidence prediction,
a point outside the workspace box, or a jump larger than max_target_step_m.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import ColorRGBA, Empty
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker, MarkerArray

from frisbee_msgs.msg import FrisbeeIntercept, FrisbeeTrack
from frisbee_interception.core.prediction import InterceptPredictor, Intrinsics, Plane, Prediction, PredictorConfig
from frisbee_interception.core.tracking import Measurement
from frisbee_interception.params import declare_section
from frisbee_interception.ros_image import bgr_to_image, compressed_to_bgr, image_to_bgr
from frisbee_interception.viz import draw_overlay


def _quat_to_R(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _vec(values, n=3):
    vals = [float(v) for v in values] if values else []
    return np.array(vals) if len(vals) == n else None


class InterceptorNode(Node):
    def __init__(self) -> None:
        super().__init__("frisbee_interceptor")
        p = self.declare_parameter
        self.target_frame = p("target_frame", "body_link").value
        self.camera_frame_override = p("camera_frame", "").value
        self.cam_param = [float(p(f"camera.{k}", 0.0).value) for k in ("fx", "fy", "cx", "cy")]
        self.plane_point = _vec(p("catch_plane.point", [0.0]).value)
        self.plane_normal = _vec(p("catch_plane.normal", [0.0]).value)
        H = [float(v) for v in p("image_calibration.homography", [0.0]).value]
        self.homography = np.array(H).reshape(3, 3) if len(H) == 9 else None
        self.cal_origin = _vec(p("image_calibration.plane_origin", [0.0]).value)
        self.cal_axis_a = _vec(p("image_calibration.plane_axis_a", [0.0]).value)
        self.cal_axis_b = _vec(p("image_calibration.plane_axis_b", [0.0]).value)
        self.max_data_age = float(p("safety.max_data_age_s", 0.25).value)
        self.track_timeout = float(p("safety.track_timeout_s", 0.5).value)
        self.min_confidence = float(p("safety.min_confidence", 0.5).value)
        self.ws_min = _vec(p("safety.workspace_min", [0.0]).value)
        self.ws_max = _vec(p("safety.workspace_max", [0.0]).value)
        self.max_step = float(p("safety.max_target_step_m", 0.3).value)
        self.min_tti_cmd = float(p("safety.min_time_to_intercept_s", 0.1).value)
        self.enabled = bool(p("interception.enabled", False).value)
        self.publish_debug = bool(p("publish_debug", True).value)
        self.predictor = InterceptPredictor(PredictorConfig.from_dict(declare_section(self, "prediction", PredictorConfig())))

        try:
            from tf2_ros import Buffer, TransformListener

            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
        except ImportError:
            self.tf_buffer = None
            self.get_logger().warn("tf2_ros not available: metric mode disabled")

        self.camera_info: CameraInfo | None = None
        self.last_track: FrisbeeTrack | None = None
        self.last_track_wall = None
        self.last_pred: Prediction | None = None
        self.last_cmd: np.ndarray | None = None
        self.last_image = None

        self.create_subscription(FrisbeeTrack, "/frisbee/track", self._on_track, 10)
        self.create_subscription(CameraInfo, "camera_info", self._on_info, qos_profile_sensor_data)
        self.create_subscription(Empty, "/frisbee/disable", self._on_disable, 10)
        qos1 = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        if p("image_transport", "raw").value == "compressed":
            self.create_subscription(CompressedImage, "image/compressed", self._on_image, qos1)
        else:
            self.create_subscription(Image, "image", self._on_image, qos1)
        self.int_pub = self.create_publisher(FrisbeeIntercept, "/frisbee/intercept", 10)
        self.pt_pub = self.create_publisher(PointStamped, "/frisbee/catch_point", 10)
        self.cmd_pub = self.create_publisher(PointStamped, "/frisbee/catch_target", 10)
        self.mk_pub = self.create_publisher(MarkerArray, "/frisbee/markers", 10)
        self.dbg_pub = self.create_publisher(Image, "/frisbee/debug/intercept", qos_profile_sensor_data)
        self.create_service(SetBool, "~/enable_interception", self._on_enable)
        self.add_on_set_parameters_callback(self._on_params)
        self.create_timer(0.1, self._watchdog)
        self.get_logger().info(
            f"frisbee_interceptor: target_frame={self.target_frame}, interception "
            f"{'ENABLED' if self.enabled else 'disabled (prediction only)'}"
        )
        self._warned: set[str] = set()

    # ------------------------------------------------------------------ arming
    def _on_enable(self, req, res):
        if req.data and (self.ws_min is None or self.ws_max is None):
            res.success, res.message = False, "refused: set safety.workspace_min / workspace_max first"
            return res
        self.enabled = bool(req.data)
        self.last_cmd = None
        res.success, res.message = True, f"automatic interception {'ENABLED' if self.enabled else 'disabled'}"
        self.get_logger().warn(res.message)
        return res

    def _on_disable(self, _msg) -> None:
        if self.enabled:
            self.get_logger().warn("automatic interception DISABLED by /frisbee/disable")
        self.enabled = False
        self.last_cmd = None

    def _on_params(self, params):
        from rcl_interfaces.msg import SetParametersResult

        for prm in params:
            if prm.name == "interception.enabled":
                self.enabled = bool(prm.value)
                self.last_cmd = None
                self.get_logger().warn(f"automatic interception {'ENABLED' if self.enabled else 'disabled'}")
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------ inputs
    def _on_info(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _on_image(self, msg) -> None:
        if self.publish_debug and self.dbg_pub.get_subscription_count() > 0:
            self.last_image = msg

    def _intrinsics(self) -> Intrinsics | None:
        if self.camera_info is not None and self.camera_info.k[0] > 0:
            k = self.camera_info.k
            return Intrinsics(k[0], k[4], k[2], k[5])
        fx, fy, cx, cy = self.cam_param
        return Intrinsics(fx, fy, cx, cy) if fx > 0 and fy > 0 else None

    def _camera_to_target(self, camera_frame: str):
        if self.tf_buffer is None or not camera_frame:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(self.target_frame, camera_frame, Time(), timeout=Duration(seconds=0.0))
        except Exception as exc:  # LookupException, ConnectivityException, ExtrapolationException
            self._warn_once(f"tf:{camera_frame}", f"no TF {camera_frame} -> {self.target_frame}: {exc}")
            return None
        tr = tf.transform
        return _quat_to_R(tr.rotation), np.array([tr.translation.x, tr.translation.y, tr.translation.z])

    def _warn_once(self, key: str, text: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self.get_logger().warn(text)

    # ------------------------------------------------------------------ core
    def _on_track(self, msg: FrisbeeTrack) -> None:
        self.last_track = msg
        self.last_track_wall = self.get_clock().now()
        stamp_age = (self.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9
        if stamp_age > self.max_data_age:
            self._publish_invalid(msg.header, f"stale camera data ({stamp_age:.2f} s old)")
            return
        if not msg.confirmed:
            self._publish_invalid(msg.header, f"no confirmed track ({msg.state})")
            return
        hist = [Measurement(t, u, v, d, c) for t, u, v, d, c in zip(msg.hist_t, msg.hist_u, msg.hist_v, msg.hist_d, msg.hist_confidence)]
        camera_frame = self.camera_frame_override or msg.header.frame_id
        K = self._intrinsics()
        pred: Prediction | None = None
        if K is not None and self.plane_point is not None and self.plane_normal is not None:
            ext = self._camera_to_target(camera_frame)
            if ext is not None:
                pred = self.predictor.predict_metric(hist, K, ext[0], ext[1], Plane(self.plane_point, self.plane_normal))
        if pred is None:
            pp = (K.cx, K.cy) if K is not None else (msg.image_width / 2.0, msg.image_height / 2.0)
            pred = self.predictor.predict_image(hist, self.homography, pp)
            if pred.valid and pred.point is not None:
                pred.point = self._plane_coords_to_target(pred.point)
        # Re-time to this message (a coasting track repeats old measurements: never re-send an old crossing).
        pred = self.predictor.at_time(pred, Time.from_msg(msg.header.stamp).nanoseconds * 1e-9)
        self.last_pred = pred
        self._publish(msg, pred)

    def _plane_coords_to_target(self, ab: np.ndarray) -> np.ndarray | None:
        if self.cal_origin is None or self.cal_axis_a is None or self.cal_axis_b is None:
            self._warn_once("plane", "image_calibration.plane_origin/axis_a/axis_b not set: catch point unavailable")
            return None
        return self.cal_origin + ab[0] * self.cal_axis_a + ab[1] * self.cal_axis_b

    def _in_workspace(self, pt: np.ndarray | None) -> bool:
        if pt is None or self.ws_min is None or self.ws_max is None:
            return False
        return bool(np.all(pt >= self.ws_min) and np.all(pt <= self.ws_max))

    def _publish_invalid(self, header, status: str) -> None:
        out = FrisbeeIntercept()
        out.header.stamp = header.stamp
        out.header.frame_id = self.target_frame
        out.valid = False
        out.status = status
        out.time_to_intercept = out.position_sigma = out.time_sigma = float("nan")
        out.pixel_u = out.pixel_v = float("nan")
        out.interception_enabled = self.enabled
        self.int_pub.publish(out)
        self._publish_markers(None, header)

    def _publish(self, track: FrisbeeTrack, pred: Prediction) -> None:
        out = FrisbeeIntercept()
        out.header.stamp = track.header.stamp
        out.header.frame_id = self.target_frame
        out.mode = pred.mode
        out.valid = bool(pred.valid)
        out.status = "ok" if pred.valid else pred.reason
        out.time_to_intercept = float(pred.time_to_intercept)
        out.position_sigma = float(pred.point_sigma_m)
        out.time_sigma = float(pred.time_sigma_s)
        out.confidence = float(pred.confidence)
        out.pixel_u, out.pixel_v = (pred.pixel if pred.pixel is not None else (float("nan"), float("nan")))
        if pred.valid and not math.isnan(pred.intercept_time):
            it = Time(seconds=pred.intercept_time) if pred.intercept_time > 0 else Time()
            out.intercept_time = it.to_msg()
        pt = pred.point if pred.valid else None
        if pt is not None:
            out.catch_point = Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2]))
        out.in_workspace = self._in_workspace(pt)
        out.interception_enabled = self.enabled
        self.int_pub.publish(out)
        if pt is not None:
            ps = PointStamped()
            ps.header = out.header
            ps.point = out.catch_point
            self.pt_pub.publish(ps)
            self._maybe_command(out, pt)
        self._publish_markers(pred, out.header)
        self._publish_debug(track, pred)

    def _maybe_command(self, out: FrisbeeIntercept, pt: np.ndarray) -> None:
        """Send a catch target to the existing controller interface only if armed and safe."""
        if not self.enabled:
            return
        reasons = []
        if out.confidence < self.min_confidence:
            reasons.append(f"confidence {out.confidence:.2f} < {self.min_confidence:.2f}")
        if not out.in_workspace:
            reasons.append("outside workspace")
        if out.time_to_intercept < self.min_tti_cmd:
            reasons.append("too late")
        if out.mode == "metric" and not (out.position_sigma <= self.predictor.cfg.max_position_sigma_m):
            reasons.append("position uncertainty")
        if self.last_cmd is not None and float(np.linalg.norm(pt - self.last_cmd)) > self.max_step:
            reasons.append(f"target jump {np.linalg.norm(pt - self.last_cmd):.2f} m > {self.max_step:.2f} m")
        if reasons:
            self.get_logger().info("catch target withheld: " + ", ".join(reasons), throttle_duration_sec=1.0)
            return
        cmd = PointStamped()
        cmd.header = out.header
        cmd.point = out.catch_point
        self.cmd_pub.publish(cmd)
        self.last_cmd = pt.copy()

    def _watchdog(self) -> None:
        """Publish an explicit 'invalid' when track data stops (camera dropped, detector crashed)."""
        if self.last_track_wall is None:
            return
        age = (self.get_clock().now() - self.last_track_wall).nanoseconds * 1e-9
        if age > self.track_timeout:
            hdr = SimpleNamespace(stamp=self.get_clock().now().to_msg(), frame_id=self.target_frame)
            self._publish_invalid(hdr, f"no track data for {age:.1f} s (timeout)")
            self.last_track_wall = None
            self.last_cmd = None

    # ------------------------------------------------------------------ visualization
    def _publish_markers(self, pred: Prediction | None, header) -> None:
        ma = MarkerArray()
        valid = pred is not None and pred.valid and pred.point is not None
        color = ColorRGBA(r=0.1, g=0.9, b=0.2, a=0.9) if valid else ColorRGBA(r=0.9, g=0.2, b=0.2, a=0.6)

        def marker(mid, mtype):
            m = Marker()
            m.header.frame_id = self.target_frame
            m.header.stamp = header.stamp
            m.ns, m.id, m.type = "frisbee", mid, mtype
            m.action = Marker.ADD if valid else Marker.DELETE
            m.pose.orientation.w = 1.0
            m.lifetime = Duration(seconds=1.0).to_msg()
            return m

        pt_m = marker(0, Marker.SPHERE)
        txt = marker(2, Marker.TEXT_VIEW_FACING)
        line = marker(1, Marker.LINE_STRIP)
        if valid:
            pt_m.pose.position = Point(x=float(pred.point[0]), y=float(pred.point[1]), z=float(pred.point[2]))
            pt_m.scale.x = pt_m.scale.y = pt_m.scale.z = 0.27
            pt_m.color = color
            txt.pose.position = Point(x=float(pred.point[0]), y=float(pred.point[1]), z=float(pred.point[2]) + 0.3)
            txt.scale.z = 0.1
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            txt.text = f"t-{pred.time_to_intercept:.2f}s  conf {pred.confidence:.2f}"
            if pred.line_points:
                line.points = [Point(x=float(q[0]), y=float(q[1]), z=float(q[2])) for q in pred.line_points]
                line.scale.x = 0.02
                line.color = ColorRGBA(r=0.0, g=0.8, b=1.0, a=0.9)
            else:
                line.action = Marker.DELETE
        ma.markers = [pt_m, line, txt]
        if self.plane_point is not None and self.plane_normal is not None:
            pl = marker(3, Marker.CUBE)
            pl.action = Marker.ADD
            n = self.plane_normal / np.linalg.norm(self.plane_normal)
            # rotate +x onto the plane normal
            axis = np.cross([1.0, 0.0, 0.0], n)
            s, c = np.linalg.norm(axis), float(n[0])
            if s > 1e-9:
                axis = axis / s
                ang = math.atan2(s, c)
                pl.pose.orientation.x, pl.pose.orientation.y, pl.pose.orientation.z = (axis * math.sin(ang / 2)).tolist()
                pl.pose.orientation.w = math.cos(ang / 2)
            pl.pose.position = Point(x=float(self.plane_point[0]), y=float(self.plane_point[1]), z=float(self.plane_point[2]))
            pl.scale.x, pl.scale.y, pl.scale.z = 0.005, 1.2, 1.2
            pl.color = ColorRGBA(r=0.6, g=0.6, b=1.0, a=0.15)
            pl.lifetime = Duration(seconds=0).to_msg()
            ma.markers.append(pl)
        self.mk_pub.publish(ma)

    def _publish_debug(self, track: FrisbeeTrack, pred: Prediction) -> None:
        img_msg = self.last_image
        if img_msg is None or self.dbg_pub.get_subscription_count() == 0:
            return
        try:
            bgr = compressed_to_bgr(img_msg) if isinstance(img_msg, CompressedImage) else image_to_bgr(img_msg)
        except (ValueError, cv2.error):
            return
        hist = [Measurement(t, u, v, d, c) for t, u, v, d, c in zip(track.hist_t, track.hist_u, track.hist_v, track.hist_d, track.hist_confidence)]
        est = {"u": track.u, "v": track.v, "d": track.d, "du": track.du, "dv": track.dv, "dd": track.dd}
        view = SimpleNamespace(history=hist, state=track.state, estimate=lambda: est if track.confirmed else None)
        self.dbg_pub.publish(bgr_to_image(draw_overlay(bgr, [], view, pred), img_msg.header, Image))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InterceptorNode()
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
