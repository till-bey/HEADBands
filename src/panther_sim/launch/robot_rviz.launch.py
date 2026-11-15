"""RViz on the real robot, drawn with the sim's stylized stand-in meshes (the robot's meshes are not on this laptop).

ros2 launch panther_sim robot_rviz.launch.py     # in a robot terminal (docker/run.sh)

The robot publishes no state for the lift or the wheels, so its own TF is split in two and misses those links.
Here a local robot_state_publisher reads the robot's /joint_states (arms, head) and builds the whole tree from
urdf/robot_description.urdf with the lift and wheels fixed (lift at the URDF height), like the stripped URDF on main.
Its description and TF go on /panther_standin/* only, and RViz reads only those: nothing is published on the
robot's own topics. The arm velocity limits are set to the sim's 0.5 rad/s (the robot's are 0), so rqt can use it too:

ros2 run rqt_joint_trajectory_controller rqt_joint_trajectory_controller --ros-args -r /robot_description:=/panther_standin/robot_description
"""
import importlib.util
import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

SHARE = get_package_share_directory("panther_sim")
_spec = importlib.util.spec_from_file_location("mock_sim", os.path.join(os.path.dirname(__file__), "mock_sim.launch.py"))
mock_sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mock_sim)

STANDIN = "/panther_standin"
TOPICS = [(t, STANDIN + t) for t in ("/robot_description", "/tf", "/tf_static")]


def standin_description():
    root = ET.parse(os.path.join(SHARE, "urdf", "robot_description.urdf")).getroot()
    for j in root.findall("joint"):
        if j.get("name") in ["body_joint"] + mock_sim.WHEELS:  # no state for these on the robot
            j.set("type", "fixed")
        lim = j.find("limit")
        if j.get("name") in mock_sim.ARM_HEAD and lim is not None and float(lim.get("velocity", "0")) == 0:
            lim.set("velocity", mock_sim.SIM_ARM_VEL)  # the robot's is 0 and rqt divides by it
    mock_sim.use_meshes(root, "")  # the vendor meshes only exist on the robot
    return ET.tostring(root, encoding="unicode")


def generate_launch_description():
    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher", name="standin_description",
             parameters=[{"robot_description": standin_description()}], remappings=TOPICS),
        Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(SHARE, "config", "panther.rviz")],
             remappings=TOPICS),
    ])
