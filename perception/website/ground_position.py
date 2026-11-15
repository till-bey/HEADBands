"""Live ground-position dashboard using the reusable white-frisbee camera API."""
from __future__ import annotations

import math
import time

import cv2
import streamlit as st

from camera import STREAM_MODES
from detector import DetectorSettings
from camera import CAMERA_BASE_HEIGHT_M, FrisbeeCamera, pixel_to_ground_robot


def orient_camera_frame(frame, orientation):
    """Keep upright portrait streams; rotate only landscape-shaped input."""
    if frame.shape[1] <= frame.shape[0] or orientation == "As received":
        return frame
    direction = cv2.ROTATE_90_COUNTERCLOCKWISE if orientation == "Portrait (counterclockwise)" else cv2.ROTATE_90_CLOCKWISE
    return cv2.rotate(frame, direction)


def displayed_fov(shape, horizontal_fov_deg, vertical_fov_deg, basis):
    """Convert landscape sensor calibration to upright portrait axes if needed."""
    height, width = shape[:2]
    if basis == "Landscape sensor" and height > width:
        sensor_vertical = vertical_fov_deg
        if sensor_vertical is None:
            sensor_vertical = math.degrees(2 * math.atan(
                width / height * math.tan(math.radians(horizontal_fov_deg) / 2)))
        return sensor_vertical, horizontal_fov_deg
    return horizontal_fov_deg, vertical_fov_deg


def draw_forward_meter(frame, camera_height, tilt_rad, horizontal_fov_deg, vertical_fov_deg=None):
    """Draw ground segment (0, 0) -> (0, 1m), clipping an offscreen origin.

    The arrowhead always marks the true projected one-meter endpoint. If that
    endpoint is outside the image, show a message instead of a misleading arrow.
    """
    annotated = frame.copy()
    height, width = frame.shape[:2]
    fx = width / (2 * math.tan(math.radians(horizontal_fov_deg) / 2))
    fy = fx if vertical_fov_deg is None else height / (2 * math.tan(math.radians(vertical_fov_deg) / 2))
    sin_t, cos_t = math.sin(tilt_rad), math.cos(tilt_rad)
    # Ground point (X=0, Y=1) in camera coordinates: forward depth and down.
    depth = cos_t + camera_height * sin_t
    down = camera_height * cos_t - sin_t
    if depth <= 1e-12:
        return annotated, "The 1 m forward ground point is behind the camera."
    end_y = height / 2 + fy * down / depth
    if not 0 <= end_y <= height - 1:
        return annotated, "The 1 m forward ground point is outside the image at this height, tilt, and FOV."
    # The ground origin below the camera may project below the frame.
    origin_depth = camera_height * sin_t
    origin_y = height / 2 + fy * camera_height * cos_t / origin_depth if origin_depth > 1e-12 else height
    clipped = origin_y > height - 1
    start = (round(width / 2), round(min(height - 1, origin_y)))
    end = (round(width / 2), round(end_y))
    color = (0, 220, 255)  # BGR amber, distinct from the frisbee-center marker
    cv2.arrowedLine(annotated, start, end, color, 2, cv2.LINE_AA, tipLength=0.18)
    label_x = max(0, min(width - 150, end[0] + 12))
    label_y = max(18, min(height - 8, end[1] - 8))
    cv2.putText(annotated, "1 m forward", (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    note = "Arrow tip: 1 m straight ahead on the ground from directly below the camera."
    if clipped:
        note += " The starting point is below the image; the arrow tail is clipped."
    return annotated, note


def render_ground_position(settings: DetectorSettings, *, saved_configuration=None) -> None:
    saved = saved_configuration or {}
    st.subheader("Frisbee ground position")
    st.caption("Uses the rim and ellipse settings in the sidebar and the imported FrisbeeCamera class. Pixel coordinates refer to the upright displayed camera image at full resolution.")
    with st.form("ground_camera_form"):
        url = st.text_input("Ground-position camera URL", value=saved.get("url", ""), key="ground_cfg_url")
        mode = st.selectbox("Ground-position stream format", STREAM_MODES, index=int(saved.get("snapshot", False)), key="ground_cfg_mode")
        connect = st.form_submit_button("Connect ground-position camera")
    if connect:
        previous = st.session_state.pop("ground_position_camera", None)
        if previous is not None:
            previous.close()
        st.session_state.pop("ground_position_frame", None)
        try:
            st.session_state["ground_position_camera"] = FrisbeeCamera(url, settings, mode=mode, portrait=False)
        except ValueError as exc:
            st.error(str(exc))

    orientation = st.selectbox("Camera orientation", ["Portrait (clockwise)", "Portrait (counterclockwise)", "As received"])
    st.caption("Portrait streams are kept upright. Landscape-shaped streams rotate 90° in the selected direction before detection and projection.")
    a, b, c = st.columns(3)
    base_height = a.number_input("Camera base height (m)", min_value=0.000001, value=saved.get("camera_base_height", CAMERA_BASE_HEIGHT_M), step=0.01, key="ground_cfg_base_height")
    robot_z = b.number_input("Height offset (m)", min_value=0.0, value=saved.get("robot_z", 0.0), step=0.01, key="ground_cfg_robot_z")
    tilt_rad = c.number_input("Tilt offset (rad)", value=saved.get("tilt_rad", math.radians(15)), step=0.01, format="%.4f", key="ground_cfg_tilt")
    height = base_height + robot_z
    hfov = st.number_input("Horizontal FOV (°)", min_value=0.000001, max_value=179.999999, value=saved.get("horizontal_fov", 77.32), step=0.01, key="ground_cfg_hfov")
    st.caption(f"Saved mounting height: {base_height:.3f} m base + {robot_z:.3f} m offset = {height:.3f} m. Live ROS body/head readings are not stored in the saved file. Tilt is zero horizontal, positive downward.")
    explicit_vfov = st.checkbox("Specify vertical FOV", value=saved.get("vertical_fov", 61.93) is not None, key="ground_cfg_use_vfov")
    vfov = st.number_input("Vertical FOV (°)", min_value=0.000001, max_value=179.999999, value=saved.get("vertical_fov") or 61.93, step=0.01, key="ground_cfg_vfov") if explicit_vfov else None
    fov_basis = st.selectbox("FOV calibration axes", ["Displayed image", "Landscape sensor"])
    st.caption("Use Displayed image for FOV measured in the upright portrait image. Choose Landscape sensor for landscape camera specifications; horizontal and vertical FOV will be swapped for portrait, deriving vertical FOV from aspect ratio if omitted.")
    st.caption("Flat ground, no camera roll, and an undistorted image are assumed. FOV calibration must match the camera lens and selected axes. Without vertical FOV, square pixels are assumed. Ground origin is below the camera; X right, Y forward; bearing 0 rad forward, positive right.")
    camera = st.session_state.get("ground_position_camera")
    if camera is not None and st.button("Stop ground-position camera"):
        camera.close()
        st.session_state.pop("ground_position_camera", None)
        st.session_state.pop("ground_position_frame", None)
        camera = None
    if camera is None:
        st.info("Connect a camera to see pixel coordinates, angles, and ground position.")
        return
    camera.settings = settings

    @st.fragment(run_every=0.15)
    def render_measurement():
        try:
            frame = camera.read_frame(timeout=0.001)
            st.session_state["ground_position_frame"] = (frame, time.monotonic())
        except TimeoutError:
            cached = st.session_state.get("ground_position_frame")
            if cached is None or time.monotonic() - cached[1] > 1.0:
                st.info("Waiting for a fresh camera frame. Check the URL and network if this persists.")
                return
            frame = cached[0]
        frame = orient_camera_frame(frame, orientation)
        effective_hfov, effective_vfov = displayed_fov(frame.shape, hfov, vfov, fov_basis)
        center = camera.detect_center(frame)
        annotated, marker_note = draw_forward_meter(frame, height, tilt_rad, effective_hfov, effective_vfov)
        if center is not None:
            cv2.circle(annotated, center, 8, (255, 80, 180), 2)
            cv2.putText(annotated, f"Center: {center}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 80, 180), 2)
        preview, results = st.columns([1, 2], gap="large")
        with preview:
            st.image(annotated, channels="BGR", caption=f"Upright camera image · {frame.shape[1]}×{frame.shape[0]}", width=360)
            st.caption(marker_note)
            if effective_vfov is not None:
                st.caption(f"Displayed FOV: horizontal {effective_hfov:.1f}°, vertical {effective_vfov:.1f}°.")
        with results:
            if center is None:
                st.info("No frisbee candidate detected in this frame. Adjust rim edge thresholds or ellipse size limits.")
                return
            px, py = center
            image_height, image_width = frame.shape[:2]
            fx = image_width / (2 * math.tan(math.radians(effective_hfov) / 2))
            fy = fx if effective_vfov is None else image_height / (2 * math.tan(math.radians(effective_vfov) / 2))
            u, v = (px - image_width / 2) / fx, (py - image_height / 2) / fy
            alpha = tilt_rad
            down = math.sin(alpha) + v * math.cos(alpha)
            forward = math.cos(alpha) - v * math.sin(alpha)
            ray_angle = math.atan2(down, math.hypot(u, forward))
            st.markdown("**Camera coordinates and angles**")
            coords = st.columns(2)
            coords[0].metric("Camera pixel X", px)
            coords[1].metric("Camera pixel Y", py)
            angles = st.columns(2)
            angles[0].metric("Camera tilt to ground", f"{tilt_rad:.4f} rad")
            angles[1].metric("Ray angle below ground horizon", f"{ray_angle:.4f} rad")
            try:
                position = pixel_to_ground_robot(px, py, image_width, image_height, robot_z, tilt_rad, effective_hfov, effective_vfov, camera_base_height_m=base_height)
            except ValueError as exc:
                st.warning(str(exc))
                return
            st.markdown("**Ground position**")
            ground = st.columns(2)
            ground[0].metric("Ground bearing", f"{position['bearing_rad']:.4f} rad")
            ground[1].metric("Ground distance", f"{position['distance']:.2f} m")
            ground_xy = st.columns(2)
            ground_xy[0].metric("Ground X (right)", f"{position['x']:.2f} m")
            ground_xy[1].metric("Ground Y (forward)", f"{position['y']:.2f} m")
            st.caption("Ray angle measures how far the detected center is below the horizontal ground plane. Ground bearing is the final horizontal direction to the frisbee.")

    render_measurement()
