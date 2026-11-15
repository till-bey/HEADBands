"""Frisbee detection + tracking + interception prediction (prediction-only by default).

  # phone stream (or a video file) through the phone_camera node
  ros2 launch frisbee_interception frisbee.launch.py source:=http://192.168.1.50:8080/stream.mjpg
  ros2 launch frisbee_interception frisbee.launch.py source:=/workspace/perception/Dataset/Videos/VID_20261114_222519.mp4

  # an existing ROS 2 camera instead (e.g. the robot's Orbbec)
  ros2 launch frisbee_interception frisbee.launch.py use_phone_camera:=false \
      image_topic:=/camera/color/image_raw camera_info_topic:=/camera/color/camera_info

Arguments: source, use_phone_camera, image_topic, camera_info_topic, image_transport, config, rotate,
calibration_file, rviz, loop, catch_trigger.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("frisbee_interception")
    config = LaunchConfiguration("config")
    image_topic = LaunchConfiguration("image_topic")
    info_topic = LaunchConfiguration("camera_info_topic")
    use_phone = LaunchConfiguration("use_phone_camera")
    transport = LaunchConfiguration("image_transport")

    return LaunchDescription([
        DeclareLaunchArgument("source", default_value="", description="phone stream URL or video file"),
        DeclareLaunchArgument("use_phone_camera", default_value="true"),
        DeclareLaunchArgument("image_topic", default_value="/phone_camera/image_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/phone_camera/camera_info"),
        # compressed (JPEG) by default: raw 720p frames (2.7 MB) are mostly dropped by best-effort DDS
        DeclareLaunchArgument("image_transport", default_value="compressed", description="raw|compressed"),
        DeclareLaunchArgument("config", default_value=os.path.join(share, "config", "frisbee.yaml")),
        DeclareLaunchArgument("rotate", default_value="none", description="none|cw|ccw|180 for the phone image"),
        DeclareLaunchArgument("calibration_file", default_value="", description="camera_calibration YAML"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("loop", default_value="false", description="replay a video file in a loop"),
        DeclareLaunchArgument("catch_trigger", default_value="true",
                              description="start frisbee_catch_trigger (disarmed: it only logs would_fire events)"),
        Node(
            package="frisbee_interception", executable="phone_camera", name="phone_camera",
            condition=IfCondition(use_phone),
            parameters=[{
                "source": LaunchConfiguration("source"),
                "rotate": LaunchConfiguration("rotate"),
                "calibration_file": LaunchConfiguration("calibration_file"),
                "loop": PythonExpression(["'", LaunchConfiguration("loop"), "'.lower() == 'true'"]),
                "publish_raw": PythonExpression(["'", transport, "' == 'raw'"]),
                "publish_compressed": PythonExpression(["'", transport, "' == 'compressed'"]),
            }],
        ),
        Node(
            package="frisbee_interception", executable="frisbee_detector", name="frisbee_detector",
            parameters=[config, {"image_transport": transport}],
            remappings=[("image", image_topic), ("image/compressed", [image_topic, "/compressed"])],
        ),
        Node(
            package="frisbee_interception", executable="frisbee_tracker", name="frisbee_tracker",
            parameters=[config],
        ),
        Node(
            package="frisbee_interception", executable="frisbee_interceptor", name="frisbee_interceptor",
            parameters=[config, {"image_transport": transport}],
            remappings=[("image", image_topic), ("image/compressed", [image_topic, "/compressed"]),
                        ("camera_info", info_topic)],
        ),
        Node(
            package="frisbee_interception", executable="frisbee_catch_trigger", name="frisbee_catch_trigger",
            condition=IfCondition(LaunchConfiguration("catch_trigger")),
            parameters=[config],
        ),
        Node(
            package="rviz2", executable="rviz2", name="rviz2", condition=IfCondition(LaunchConfiguration("rviz")),
            arguments=["-d", os.path.join(share, "rviz", "frisbee.rviz")],
        ),
    ])
