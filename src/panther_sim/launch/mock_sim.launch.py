"""Panther arms, head, lift and grippers on ros2_control mock hardware, shown in RViz, driven from rqt. No physics.
Always runs on ROS_DOMAIN_ID 77, localhost only, whatever the terminal is set to.

ros2 launch panther_sim mock_sim.launch.py
  urdf:=<path>    full URDF captured from the robot (default: the stripped one from PR #521)
  meshes:=<dir>   local copy of the robot's /opt/ros/humble/share/mh3_description
"""
import math
import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable, UnsetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SHARE = get_package_share_directory("panther_sim")
ARM_HEAD = [f"joint_{s}a{i}" for s in "lr" for i in range(8)] + ["joint_head0", "joint_head1"]
# Lift (base z axis) and grippers: the vendor URDF has them fixed; robot_description() frees them.
LIFT_GRIPPERS = ["body_joint", "joint_lf", "joint_rf"]
LIFT_VEL = "0.05"   # m/s, the SDK lift limit
LIFT_TOP = 0.8      # m; lift runs 0 (bottom) to 0.8 (SDK, 800 mm travel; top unconfirmed on the robot); the vendor body height is the top
FINGER_CLOSE = 0.7  # rad of stand-in finger swing at gripper 1.0 (closed); looks right, not a vendor value
WHEELS = [f"joint_{w}_{k}" for w in ("fl", "fr", "bl", "br") for k in ("rev", "con")]
# The vendor URDF gives every arm joint velocity="0"; rqt divides by it. 0.5 rad/s is our choice;
# the vendor publishes no limit (PR #521 moves the arms at 0.35 rad/s average).
SIM_ARM_VEL = "0.5"
SIM_DOMAIN = "77"
VENDOR_MESHES ="file:///opt/ros/humble/share/mh3_description"


def _visual(link, geometry, xyz="0 0 0", rpy="0 0 0", rgba="0.75 0.75 0.78 1"):
    v = ET.SubElement(link, "visual")
    ET.SubElement(v, "origin", xyz=xyz, rpy=rpy)
    ET.SubElement(ET.SubElement(v, "geometry"), *geometry)
    m = ET.SubElement(v, "material", name=rgba)
    ET.SubElement(m, "color", rgba=rgba)


def add_primitive_visuals(root):
    """Stand-in for the vendor meshes: each link is a rod from its origin to each child joint."""
    links = {l.get("name"): l for l in root.findall("link")}
    _visual(links["base_link"], ("box", {"size": "0.5 0.42 0.22"}), xyz="0 0 0.17", rgba="0.25 0.25 0.28 1")
    for j in root.findall("joint"):
        parent, child = j.find("parent").get("link"), j.find("child").get("link")
        x, y, z = (float(v) for v in j.find("origin").get("xyz", "0 0 0").split())
        if child.endswith("_con"):  # drive wheel
            _visual(links[child], ("cylinder", {"radius": "0.06", "length": "0.04"}), rpy="1.5708 0 0", rgba="0.1 0.1 0.1 1")
        if j.get("type") == "revolute":
            _visual(links[child], ("sphere", {"radius": "0.035"}), rgba="0.95 0.55 0.1 1")
        length = math.sqrt(x * x + y * y + z * z)
        if length < 0.02 or (parent == "base_link" and child != "body_link"):
            continue  # base box already covers the wheel mounts and sensors
        radius = "0.07" if child == "body_link" else "0.025"
        rpy = f"0 {math.atan2(math.hypot(x, y), z)} {math.atan2(y, x)}"  # turns the cylinder's z axis onto (x, y, z)
        _visual(links[parent], ("cylinder", {"radius": radius, "length": str(length)}), xyz=f"{x/2} {y/2} {z/2}", rpy=rpy)
    _visual(links["link_head1"], ("box", {"size": "0.14 0.2 0.16"}), xyz="0.04 0 0.07", rgba="0.25 0.25 0.28 1")


def robot_description(urdf_path, mesh_dir):
    root = ET.parse(urdf_path).getroot()
    for tag in ("ros2_control", "gazebo"):  # vendor blocks load CAN hardware plugins that only exist on the robot
        for e in root.findall(tag):
            root.remove(e)
    for j in root.findall("joint"):
        lim = j.find("limit")
        if j.get("name") in ARM_HEAD and lim is not None and float(lim.get("velocity", "0")) == 0:
            lim.set("velocity", SIM_ARM_VEL)
    by_name = {j.get("name"): j for j in root.findall("joint")}
    if by_name["body_joint"].get("type") == "fixed":
        # The vendor joint (axis 0 0 -1, limit 0-0.6) has the direction wrong: 0 is the bottom, so the axis goes
        # up and the origin drops by the full travel to put the vendor height at the top.
        lift = by_name["body_joint"]
        lift.set("type", "prismatic")
        lift.find("axis").set("xyz", "0 0 1")
        lift.find("limit").attrib.update(lower="0", upper=str(LIFT_TOP), velocity=LIFT_VEL)
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
    if not any(l.find("visual") is not None for l in root.findall("link")):
        add_primitive_visuals(root)

    joints = "".join(
        f'<joint name="{j}"><command_interface name="position"/>'
        f'<state_interface name="position"><param name="initial_value">{LIFT_TOP / 2 if j == "body_joint" else 0.0}</param>'
        '</state_interface><state_interface name="velocity"/></joint>' for j in ARM_HEAD + LIFT_GRIPPERS)
    # Wheels are state-only so robot_state_publisher gets a TF for them.
    joints += "".join(f'<joint name="{j}"><state_interface name="position"/></joint>' for j in WHEELS)
    root.append(ET.fromstring('<ros2_control name="panther_mock" type="system"><hardware>'
                              '<plugin>mock_components/GenericSystem</plugin></hardware>' + joints + "</ros2_control>"))
    xml = ET.tostring(root, encoding="unicode")
    if mesh_dir:
        xml = xml.replace(VENDOR_MESHES, "file://" + os.path.abspath(os.path.expanduser(mesh_dir)))
    return xml


def nodes(context):
    desc = {"robot_description": robot_description(LaunchConfiguration("urdf").perform(context),
                                                   LaunchConfiguration("meshes").perform(context))}
    spawn = lambda *c: Node(package="controller_manager", executable="spawner", arguments=list(c))
    return [
        Node(package="robot_state_publisher", executable="robot_state_publisher", parameters=[desc]),
        Node(package="controller_manager", executable="ros2_control_node",
             parameters=[os.path.join(SHARE, "config", "controllers.yaml")], output="screen"),
        # As on the robot: the forward controller owns both arms; the per-arm trajectory controllers
        # are loaded inactive and switched in for a move (cartesian_follow.py, PR #521 driver).
        spawn("joint_state_broadcaster", "dual_arm_forward_position_controller", "head_forward_position_controller",
              "lifting_joint_trajectory_controller", "left_gripper_joint_trajectory_controller",
              "right_gripper_joint_trajectory_controller"),
        spawn("--inactive", "left_arm_joint_trajectory_position_controller",
              "right_arm_joint_trajectory_position_controller", "head_joint_trajectory_controller"),
        # The robot's lift and gripper topics (its controllers for them are vendor plugins).
        Node(package="panther_sim", executable="robot_topics", output="screen"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(SHARE, "config", "panther.rviz")]),
        Node(package="rqt_joint_trajectory_controller", executable="rqt_joint_trajectory_controller"),
    ]


def generate_launch_description():
    return LaunchDescription([
        # rqt_joint_trajectory_controller 4.42 never finds the controller manager under Fast DDS
        # (fresh node per lookup, queried before discovery). The robot runs Cyclone anyway.
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"),
        # Never join the robot's graph, whatever the terminal has: the sim uses the robot's
        # controller names, so on the robot's domain rqt could command the real arms.
        SetEnvironmentVariable("ROS_DOMAIN_ID", SIM_DOMAIN),
        SetEnvironmentVariable("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST"),
        UnsetEnvironmentVariable("CYCLONEDDS_URI"),  # robot setups pin it to the robot's NIC
        DeclareLaunchArgument("urdf", default_value=os.path.join(SHARE, "urdf", "panther.urdf")),
        DeclareLaunchArgument("meshes", default_value=""),
        OpaqueFunction(function=nodes),
    ])
