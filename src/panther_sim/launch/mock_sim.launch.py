"""Panther arms, head, lift and grippers on ros2_control mock hardware, shown in RViz, driven from rqt. No physics.
Always runs on ROS_DOMAIN_ID 77, localhost only, whatever the terminal is set to.

ros2 launch panther_sim mock_sim.launch.py
  urdf:=<path>    robot URDF (default: urdf/robot_description.urdf, the robot's /robot_description)
  meshes:=<dir>   local copy of the robot's mh3_description share directory; without it, stylized stand-ins
"""
import os
import re
import tempfile
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from panther_examples import robot as r
from panther_sim import meshes

SHARE = get_package_share_directory("panther_sim")
LIFT_TOP = r.MAX_LIFT  # the vendor body height is the top of the travel
ARM_HEAD = r.ARMS + r.HEAD
FINGER_CLOSE = 0.7  # rad of stand-in finger swing at gripper 1.0 (closed); looks right, not a vendor value
WHEELS = [f"joint_{w}_{k}" for w in ("fl", "fr", "bl", "br") for k in ("rev", "con")]
# One mock system per robot hardware component, same names, so hardware lifecycle calls (teach.py) work here.
# The lift and grippers are fixed in the vendor URDF; robot_description() frees them.
HARDWARE = {r.LEFT_HARDWARE: r.LEFT_ARM, r.RIGHT_HARDWARE: r.RIGHT_ARM, r.LEFT_GRIPPER_HARDWARE: ["joint_lf"],
            r.RIGHT_GRIPPER_HARDWARE: ["joint_rf"], "mh3_head": r.HEAD, "mh3_movebase": ["body_joint"]}
# The vendor URDF gives every arm joint velocity="0"; rqt divides by it. 0.5 rad/s is our choice;
# the vendor publishes no limit (PR #521 moves the arms at 0.35 rad/s average).
SIM_ARM_VEL = "0.5"
SIM_DOMAIN = "77"
# Loopback only, multicast forced on (lo has none by default). ROS_LOCALHOST_ONLY=1 instead caps Cyclone at a few
# participants on Humble, and the sim has about ten. Same string as docker/run.sh --sim.
SIM_CYCLONE = '<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" multicast="true"/></Interfaces></General></Domain></CycloneDDS>'
# The robot has published mesh paths from both /opt/ros/humble/share and its workspace install.
VENDOR_MESHES = r"file://[^\"]*/share/mh3_description"


def use_meshes(root, mesh_dir):
    """Point the vendor mesh paths at mesh_dir, or at the stylized stand-ins (panther_sim/meshes.py) without one."""
    if not mesh_dir:
        mesh_dir = meshes.write_all(os.path.join(tempfile.gettempdir(), "panther_sim_meshes"))
        for link in root.findall("link"):  # odin1_ros2_driver is not installed here; base_link's stand-in has the sensor
            for v in link.findall("visual"):
                if "odin1_ros2_driver" in v.find("geometry/mesh").get("filename", ""):
                    link.remove(v)
    for m in root.iter("mesh"):
        m.set("filename", re.sub(VENDOR_MESHES, "file://" + os.path.abspath(os.path.expanduser(mesh_dir)), m.get("filename")))


def robot_description(urdf_path, mesh_dir):
    root = ET.parse(urdf_path).getroot()
    for tag in ("ros2_control", "gazebo"):  # vendor blocks load CAN hardware plugins that only exist on the robot
        for e in root.findall(tag):
            root.remove(e)
    use_meshes(root, mesh_dir)
    for j in root.findall("joint"):
        lim = j.find("limit")
        if j.get("name") in ARM_HEAD and lim is not None and float(lim.get("velocity", "0")) == 0:
            lim.set("velocity", SIM_ARM_VEL)
    by_name = {j.get("name"): j for j in root.findall("joint")}
    lift = by_name["body_joint"]
    if lift.find("axis").get("xyz") == "0 0 -1":
        # The vendor joint (axis 0 0 -1, limit 0-0.6) has the direction wrong: 0 is the bottom, so the axis goes
        # up and the origin drops by the full travel to put the vendor height at the top.
        lift.set("type", "prismatic")
        lift.find("axis").set("xyz", "0 0 1")
        lift.find("limit").attrib.update(lower="0", upper=str(LIFT_TOP), velocity=str(r.LIFT_VEL))
        x, y, z = (float(v) for v in lift.find("origin").get("xyz").split())
        lift.find("origin").set("xyz", f"{x} {y} {z - LIFT_TOP}")
    for s in "lr":
        if by_name[f"joint_{s}f"].get("type") != "fixed":
            continue
        # ros2_control rejects fixed joints, but cartesian_follow.py's IK runs through joint_{s}f to link_{s}f.
        # So the vendor joint stays fixed under a new name, and joint_{s}f becomes a 0 (open)..1 (closed)
        # joint on a dummy link that moves nothing itself; the two finger roots mimic it.
        by_name[f"joint_{s}f"].set("name", f"joint_{s}f_mount")
        root.append(ET.fromstring(f'<link name="link_{s}f_cmd"/>'))
        root.append(ET.fromstring(
            f'<joint name="joint_{s}f" type="revolute"><parent link="link_{s}f"/><child link="link_{s}f_cmd"/>'
            '<origin xyz="0 0 0"/><axis xyz="0 0 1"/><limit effort="0" lower="0" upper="1" velocity="1"/></joint>'))
        for finger in (f"joint_{s}f00", f"joint_{s}f10"):
            by_name[finger].set("type", "revolute")
            by_name[finger].find("limit").set("upper", str(FINGER_CLOSE))
            ET.SubElement(by_name[finger], "mimic", joint=f"joint_{s}f", multiplier=str(FINGER_CLOSE))

    for hw, names in HARDWARE.items():
        joints = "".join(
            f'<joint name="{j}"><command_interface name="position"/>'
            f'<state_interface name="position"><param name="initial_value">{LIFT_TOP / 2 if j == "body_joint" else 0.0}</param>'
            '</state_interface><state_interface name="velocity"/></joint>' for j in names)
        if hw == "mh3_movebase":  # wheels are state-only so robot_state_publisher gets a TF for them
            joints += "".join(f'<joint name="{j}"><state_interface name="position"/></joint>' for j in WHEELS)
        root.append(ET.fromstring(f'<ros2_control name="{hw}" type="system"><hardware>'
                                  '<plugin>mock_components/GenericSystem</plugin></hardware>' + joints + "</ros2_control>"))
    return ET.tostring(root, encoding="unicode")


def nodes(context):
    desc = {"robot_description": robot_description(LaunchConfiguration("urdf").perform(context),
                                                   LaunchConfiguration("meshes").perform(context))}
    spawn = lambda *c: Node(package="controller_manager", executable="spawner", arguments=list(c))
    return [
        Node(package="robot_state_publisher", executable="robot_state_publisher", parameters=[desc]),
        Node(package="controller_manager", executable="ros2_control_node",
             parameters=[os.path.join(SHARE, "config", "controllers.yaml")], output="screen",
             remappings=[("~/robot_description", "/robot_description")]),  # Humble listens on its own topic
        # As on the robot: the forward controller owns both arms; the per-arm trajectory controllers
        # are loaded inactive and switched in for a move (cartesian_follow.py, PR #521 driver).
        spawn("joint_state_broadcaster", r.FORWARD, r.HEAD_FORWARD,
              r.LIFT_CONTROLLER, r.LEFT_GRIPPER_CONTROLLER, r.RIGHT_GRIPPER_CONTROLLER),
        spawn("--inactive", r.LEFT_TRAJ, r.RIGHT_TRAJ, "head_joint_trajectory_controller"),
        # The robot's lift and gripper topics and CAN enable/disable services (vendor plugins on the robot).
        Node(package="panther_sim", executable="robot_topics", output="screen"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(SHARE, "config", "panther.rviz")]),
        Node(package="rqt_joint_trajectory_controller", executable="rqt_joint_trajectory_controller"),
    ]


def generate_launch_description():
    return LaunchDescription([
        # rqt_joint_trajectory_controller can miss the controller manager under Fast DDS. The robot runs Cyclone anyway.
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"),
        # Never join the robot's graph, whatever the terminal has: the sim uses the robot's
        # controller names, so on the robot's domain rqt could command the real arms.
        SetEnvironmentVariable("ROS_DOMAIN_ID", SIM_DOMAIN),
        SetEnvironmentVariable("CYCLONEDDS_URI", SIM_CYCLONE),  # replaces a robot setup's NIC pinning
        DeclareLaunchArgument("urdf", default_value=os.path.join(SHARE, "urdf", "robot_description.urdf")),
        DeclareLaunchArgument("meshes", default_value=""),
        OpaqueFunction(function=nodes),
    ])
