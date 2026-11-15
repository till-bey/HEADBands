#!/usr/bin/env python3
"""Camera dashboard with browser controls for navigation and body height."""

import time
from copy import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perception"))
from camera import create_camera, pixel_to_ground_robot

import cv2
import numpy as np

if __package__:
    from .config import parse_args, create_controller
    from .dashboard_server import DashboardServer
else:
    from config import parse_args, create_controller
    from dashboard_server import DashboardServer


def inspect_frame(camera, frame, args, controller):
    """Detect once, projecting the same center that is drawn on the image."""
    center = camera.detect_center(frame)
    position = None
    status = "No frisbee detected"
    if center is not None:
        height, width = frame.shape[:2]
        try:
            position = pixel_to_ground_robot(
                *center, width, height, args.robot_z, args.tilt_rad,
                args.horizontal_fov, args.vertical_fov,
            )
            status = "Tracking"
        except ValueError as exc:
            status = f"Invalid ground projection: {exc}"
    return center, position, controller.compute(position), status


def render_view(frame, center, position, command, status, *, robot_status=None):
    """Keep the image aspect ratio and place telemetry to its right."""
    height, width = frame.shape[:2]
    scale = min(1280 / width, 720 / height, 1.0)
    view = cv2.resize(frame, (round(width * scale), round(height * scale)))
    h, w = view.shape[:2]
    cv2.line(view, (w // 2, 0), (w // 2, h - 1), (150, 150, 150), 1)
    if center is not None:
        point = tuple(round(v * scale) for v in center)
        color = (80, 230, 80) if position is not None else (0, 190, 255)
        cv2.circle(view, point, 18, color, 2)
        cv2.drawMarker(view, point, color, cv2.MARKER_CROSS, 28, 2)
        cv2.putText(view, "Frisbee", (max(0, point[0] - 30), max(20, point[1] - 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    if position is None:
        geometry = ["Distance: -- m", "Angle: -- rad (-- deg)",
                    "Ground X (right): -- m", "Ground Y (forward): -- m"]
    else:
        geometry = [
            f"Distance: {position['distance']:.3f} m",
            f"Angle: {position['bearing_rad']:+.3f} rad ({position['bearing_deg']:+.1f} deg)",
            f"Ground X (right): {position['x']:+.3f} m",
            f"Ground Y (forward): {position['y']:+.3f} m",
        ]
    robot_label = (f"ROS: {robot_status['status']}" if isinstance(robot_status, dict)
                   else "Preview only - no robot commands")
    lines = [
        "NAVIGATION DASHBOARD", robot_label, "", status, "",
        *geometry, "",
        f"Controller: {command.state}",
        f"Proposed forward: {command.linear_speed:.3f} m/s",
        f"Proposed turn: {command.angular_speed:+.3f} rad/s", "",
        "Angle: 0 forward, + right, - left.",
        "ROS turn: + left.", "Ctrl+C in terminal to quit.",
    ]
    panel_width, padding = 400, 24
    font, font_scale = cv2.FONT_HERSHEY_SIMPLEX, 0.58
    max_text_width = panel_width - 2 * padding
    rows = []
    for text in lines:
        if not text:
            rows.append("")
            continue
        # Measure actual text so long status messages and URLs stay inside the panel.
        while text:
            end = len(text)
            while cv2.getTextSize(text[:end], font, font_scale, 1)[0][0] > max_text_width:
                end -= 1
            if end < len(text):
                space = text.rfind(" ", 0, end + 1)
                if space > 0:
                    end = space
            rows.append(text[:end])
            text = text[end:].lstrip()
    panel_height = padding * 2 + sum(28 if row else 14 for row in rows)
    canvas = np.zeros((max(h, panel_height), w + panel_width, 3), dtype=np.uint8)
    canvas[:h, :w] = view
    canvas[:, w:] = (28, 21, 16)
    y = padding
    for index, row in enumerate(rows):
        if row:
            color = (189, 223, 135) if index == 0 else (230, 230, 230)
            cv2.putText(canvas, row, (w + padding, y + 18),
                        font, font_scale, color, 1, cv2.LINE_AA)
            y += 28
        else:
            y += 14
    return canvas


def run_dashboard(display, args, *, robot_control=None):
    """Apply browser requests and process the selected camera on one thread."""
    controller = create_controller(args)
    camera = None
    blank = np.zeros((640, 480, 3), dtype=np.uint8)
    last_frame = blank.copy()
    status = "Enter a camera URL in the browser and click Connect"
    try:
        while True:
            generation = robot_control.target_generation() if robot_control is not None else None
            update = display.take_update()
            if update is not None:
                if robot_control is not None:
                    robot_control.clear_target("Camera settings changed; navigation stopped")
                    generation = robot_control.target_generation()
                action, settings = update
                if camera is not None:
                    camera.close()
                    camera = None
                last_frame = blank.copy()
                status = "Camera disconnected"
                if action == "connect":
                    args = settings
                    controller = create_controller(args)
                    try:
                        camera = create_camera(args)
                        status = "Connecting camera"
                    except (ValueError, RuntimeError) as exc:
                        status = f"Camera error: {exc}"

            center, position = None, None
            command = controller.compute(None)
            observed_at = time.monotonic()
            if camera is not None:
                started_at = observed_at
                try:
                    projection_args = args
                    if robot_control is not None:
                        lift = robot_control.status()
                        if lift.get("lift_fresh") and lift.get("lift_position") is not None:
                            projection_args = copy(args)
                            projection_args.robot_z = lift["lift_position"]
                    last_frame = camera.read_frame(timeout=0.2)
                    center, position, command, status = inspect_frame(camera, last_frame, projection_args, controller)
                    if time.monotonic() - started_at > args.target_timeout:
                        center, position, command = None, None, controller.compute(None)
                        status = "Camera result expired"
                except TimeoutError:
                    status = "Waiting for a fresh camera frame (previous image shown)"
                except (ValueError, RuntimeError, cv2.error) as exc:
                    status = f"Camera error: {exc}"
            robot_status = None
            if robot_control is not None:
                robot_control.update_target(position, observed_at, controller,
                                            target_timeout=args.target_timeout, camera_status=status,
                                            expected_generation=generation)
                robot_status = robot_control.status()
            display.show(render_view(last_frame, center, position, command, status, robot_status=robot_status))
            time.sleep(0.05)
    finally:
        if robot_control is not None:
            robot_control.clear_target("Camera preview stopped; navigation stopped")
        if camera is not None:
            camera.close()


def main(argv=None):
    args = parse_args(argv, description=__doc__, dashboard=True)
    display = DashboardServer(args.port, settings=args)
    try:
        run_dashboard(display, args, robot_control=display.robot_control)
    except KeyboardInterrupt:
        pass
    finally:
        display.close()


if __name__ == "__main__":
    main()
