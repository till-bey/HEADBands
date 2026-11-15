#!/usr/bin/env python3
"""Read the frisbee target, compute P control, and update robot velocities."""

import time
from pathlib import Path
import sys

# Shared camera package also used by the website.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perception"))
from camera import CameraTargetReader, create_camera

if __package__:
    from .config import parse_args, create_controller
else:
    from config import parse_args, create_controller


def control_step(reader, controller, robot, *, target_timeout=0.5):
    """Read the latest target and apply one velocity update through the robot adapter."""
    position, observed_at, camera_status = reader.snapshot()
    if time.monotonic() - observed_at > target_timeout:
        position, camera_status = None, "Camera result expired"

    command = controller.compute(position)
    status = command.state if position is not None else camera_status
    if not robot.set_velocity(command.linear_speed, command.angular_speed):
        status = "Waiting for cmd_vel subscriber"
    return status


def control_loop(reader, controller, robot, *, target_timeout=0.5,
                 prepare_step=None, on_status=None):
    """Shared CLI/dashboard loop; robot hides all ROS operations.

    prepare_step services pending browser requests and returns the current
    (controller, target_timeout) before each step. on_status receives every
    result, while ROS logging only reports changes.
    """
    last_status = None
    while robot.running:
        if prepare_step is not None:
            controller, target_timeout = prepare_step()
        status = control_step(reader, controller, robot, target_timeout=target_timeout)
        if on_status is not None:
            on_status(status)
        if status != last_status:
            robot.log(status)
            last_status = status
        robot.wait(0.05)


def main(argv=None):
    args = parse_args(argv)
    camera = create_camera(args)
    controller = create_controller(args)
    reader = CameraTargetReader(
        camera, robot_z=args.robot_z, tilt_rad=args.tilt_rad,
        horizontal_fov_deg=args.horizontal_fov, vertical_fov_deg=args.vertical_fov,
    )
    # Load ROS only when running navigation, keeping --help usable without ROS.
    if __package__:
        from .navigation import RobotNavigation
    else:
        from navigation import RobotNavigation
    try:
        with RobotNavigation() as robot:
            reader.start()
            control_loop(reader, controller, robot, target_timeout=args.target_timeout)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()


if __name__ == "__main__":
    main()
