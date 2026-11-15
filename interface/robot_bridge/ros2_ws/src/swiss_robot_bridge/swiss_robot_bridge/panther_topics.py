"""Real UniX AI Panther ROS 2 topic names, services, joints, hardware limits, and frisbee trajectories.

Derived from the `clean_up` branch of `HEADBands` (`src/panther_examples/panther_examples/robot.py`,
`master_controller.py`, `cartesian_follow.py`, `teach.py`, `joint_check.py`, `commands.py`,
`src/panther_sim/panther_sim/robot_topics.py`, `motion/retarget/`, and `perception/detector/`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PANTHER_ROS_DOMAIN_ID = 15
PANTHER_SIM_ROS_DOMAIN_ID = 77
PANTHER_RMW_IMPLEMENTATION = "rmw_cyclonedds_cpp"
PANTHER_SIM_CYCLONEDDS_URI = (
    '<CycloneDDS><Domain><General><Interfaces>'
    '<NetworkInterface name="lo" multicast="true"/>'
    "</Interfaces></General></Domain></CycloneDDS>"
)

# --- Controller & Hardware Component Names (from panther_examples/robot.py) ---
FORWARD_CONTROLLER = "dual_arm_forward_position_controller"
LEFT_TRAJ_CONTROLLER = "left_arm_joint_trajectory_position_controller"
RIGHT_TRAJ_CONTROLLER = "right_arm_joint_trajectory_position_controller"
HEAD_FORWARD_CONTROLLER = "head_forward_position_controller"
LEFT_GRIPPER_CONTROLLER = "left_unix_gripper3_controller"
RIGHT_GRIPPER_CONTROLLER = "right_unix_gripper3_controller"
LIFT_CONTROLLER = "lifting_controller"

LEFT_HARDWARE = "unix_openarm_left"
RIGHT_HARDWARE = "unix_openarm_right"
LEFT_GRIPPER_HARDWARE = "raw_can0"
RIGHT_GRIPPER_HARDWARE = "raw_can1"

# --- Command Topics (Bridge -> Panther Controllers) ---
TOPIC_CMD_VEL = "/cmd_vel"  # geometry_msgs/msg/Twist
TOPIC_PREEMPT_TELEOP = "/preempt_teleop"  # std_msgs/msg/Empty
TOPIC_MASTER_TRAJ = (
    "/master_controller/joint_trajectory"  # trajectory_msgs/msg/JointTrajectory (whole upper body)
)
TOPIC_DUAL_ARM_COMMANDS = (
    f"/{FORWARD_CONTROLLER}/commands"  # std_msgs/msg/Float64MultiArray (16 joints)
)
TOPIC_HEAD_COMMANDS = f"/{HEAD_FORWARD_CONTROLLER}/commands"  # std_msgs/msg/Float64MultiArray
TOPIC_LIFT_COMMANDS = f"/{LIFT_CONTROLLER}/commands"  # sensor_msgs/msg/JointState
TOPIC_LEFT_GRIPPER_CMD = "/left_gripper/gripper_command"  # unix_msgs/msg/GripperCommand
TOPIC_RIGHT_GRIPPER_CMD = "/right_gripper/gripper_command"  # unix_msgs/msg/GripperCommand
TOPIC_LEFT_ARM_TRAJ = (
    f"/{LEFT_TRAJ_CONTROLLER}/joint_trajectory"  # trajectory_msgs/msg/JointTrajectory
)
TOPIC_RIGHT_ARM_TRAJ = (
    f"/{RIGHT_TRAJ_CONTROLLER}/joint_trajectory"  # trajectory_msgs/msg/JointTrajectory
)
TOPIC_PLANNED_PATH = "/cartesian_follow/planned_path"  # nav_msgs/msg/Path (frame body_link)

# --- Services ---
SRV_ULTRASONIC_SWITCH = "/movebase3_drive_controller/ultrasonic_switch"  # std_srvs/srv/SetBool
SRV_LIST_CONTROLLERS = "/controller_manager/list_controllers"  # controller_manager_msgs/srv/ListControllers
SRV_SWITCH_CONTROLLER = "/controller_manager/switch_controller"  # controller_manager_msgs/srv/SwitchController

# --- State, Safety & Sensor Topics (Panther -> Bridge) ---
TOPIC_BATTERY_STATE = "/battery_state_broadcaster/battery_state"  # sensor_msgs/msg/BatteryState
TOPIC_ODOM = "/odom"  # nav_msgs/msg/Odometry
TOPIC_POSE = "/pose"  # geometry_msgs/msg/PoseWithCovarianceStamped
TOPIC_DIAGNOSTICS = "/diagnostics"  # diagnostic_msgs/msg/DiagnosticArray (includes E-Stop entry)
TOPIC_ROBOT_DESCRIPTION = "/robot_description"  # std_msgs/msg/String (latched URDF)
TOPIC_JOINT_STATES = "/joint_states"  # sensor_msgs/msg/JointState
TOPIC_LIFT_STATES = f"/{LIFT_CONTROLLER}/joint_states"  # sensor_msgs/msg/JointState
TOPIC_LEFT_GRIPPER_STATES = "/left_gripper/joint_states"  # sensor_msgs/msg/JointState
TOPIC_RIGHT_GRIPPER_STATES = "/right_gripper/joint_states"  # sensor_msgs/msg/JointState
TOPIC_FRONT_SCAN = "/scan"  # sensor_msgs/msg/LaserScan (real Panther front radar)
TOPIC_SIM_FRONT_SCAN = "/lidar_front/scan"  # sensor_msgs/msg/LaserScan (Gazebo sim front radar)

# --- Orbbec Gemini 2 RGB-D Camera Topics (active while camera driver runs) ---
TOPIC_CAMERA_COLOR_RAW = "/camera/color/image_raw"  # sensor_msgs/msg/Image
TOPIC_CAMERA_COLOR_COMPRESSED = "/camera/color/image_raw/compressed"  # sensor_msgs/msg/CompressedImage
TOPIC_CAMERA_DEPTH_RAW = "/camera/depth/image_raw"  # sensor_msgs/msg/Image

ORBBEC_WS_SETUP_PATH = "/home/unix_ai/orbbec_ws/install/setup.bash"
DEFAULT_ORBBEC_LAUNCH_CMD = (
    "bash -lc 'source /home/unix_ai/orbbec_ws/install/setup.bash "
    "&& exec ros2 launch orbbec_camera gemini2.launch.py'"
)

# --- Joint Names & Safe Mechanical Limits (panther_examples/robot.py + URDF + PR #521) ---
LEFT_ARM_JOINTS: tuple[str, ...] = tuple(f"joint_la{i}" for i in range(8))
RIGHT_ARM_JOINTS: tuple[str, ...] = tuple(f"joint_ra{i}" for i in range(8))
ARM_JOINTS: tuple[str, ...] = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
HEAD_JOINTS: tuple[str, ...] = ("joint_head0", "joint_head1")  # pitch, yaw
MASTER_STEPPED_JOINTS: tuple[str, ...] = HEAD_JOINTS + ("body_joint", "joint_lf", "joint_rf")

LIFT_MIN_M = 0.0
LIFT_MAX_M = 0.80  # MIN_LIFT, MAX_LIFT = 0.0, 0.8 m in panther_examples/robot.py (0 = bottom)
LIFT_MAX_VEL_MPS = 0.05  # LIFT_VEL = 0.05 m/s in panther_examples/robot.py

GRIPPER_MODE_PARALLEL = "parallel"
GRIPPER_OPEN_POS = 0.0
GRIPPER_CLOSED_POS = 1.0
GRIPPER_DEFAULT_VEL = 0.5  # GRIPPER_VEL = 0.5 strokes/s in master_controller.py & joint_check.py
RIGHT_GRIPPER_MAX = 1.0  # panther_examples/robot.py: master_gui closes the right gripper no further

MAX_ARM_JOINT_VEL_RPS = 0.5  # MAX_JOINT_VEL = 0.5 rad/s in cartesian_follow.py

# URDF limits (src/panther_sim/urdf/robot_description.urdf), re-checked locally for generated gestures.
ARM_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "joint_la0": (-1.046, 0.697), "joint_la1": (-3.14, 0.785), "joint_la2": (0.0, 1.57),
    "joint_la3": (-2.79, 2.79), "joint_la4": (-2.23, 0.0), "joint_la5": (-2.79, 2.79),
    "joint_la6": (-0.87, 0.87), "joint_la7": (-0.69, 0.69),
    "joint_ra0": (-0.697, 1.046), "joint_ra1": (-3.14, 0.785), "joint_ra2": (-1.57, 0.0),
    "joint_ra3": (-2.79, 2.79), "joint_ra4": (-2.23, 0.0), "joint_ra5": (-2.79, 2.79),
    "joint_ra6": (-0.87, 0.87), "joint_ra7": (-1.57, 1.57),
}
GESTURE_MAX_DURATION_S = 30.0

# Head bounds. joint_head0 = pitch (+ looks down), joint_head1 = yaw (+ looks left).
# Conservative (PR #521): pitch [-0.34, +0.34], yaw [-1.20, +1.20]. Used until PANTHER_HEAD_FULL_RANGE=1,
# which should only be set once the head command order has been checked on the robot (send a small
# yaw, watch which /joint_states entry moves).
HEAD_PITCH_MIN_RAD = -0.34
HEAD_PITCH_MAX_RAD = 0.34
HEAD_YAW_MIN_RAD = -1.20
HEAD_YAW_MAX_RAD = 1.20
# Full range: URDF/user-guide limits minus 5 deg (pitch -0.348..1.5, yaw +-1.57).
HEAD_FULL_PITCH_MIN_RAD = -0.2607
HEAD_FULL_PITCH_MAX_RAD = 1.4127
HEAD_FULL_YAW_MAX_RAD = 1.4827
HEAD_MAX_VEL_RPS = 0.6  # smoothing speed for head moves
HEAD_HARD_MAX_VEL_RPS = 0.7  # gesture trajectories are refused above this

# Right-arm frisbee throws, one CSV each in HEADBands `motion/retarget/` (rows t,joint_ra0..joint_ra7,joint_rf with
# the gripper 1 closed / 0 open), loaded and checked when the bridge starts. The cloud picks one by name
# (throw_variant, e.g. "throw the frisbee early"); a new version is a new CSV and a line here (and its words in the
# backend's intent_parser.THROW_VARIANT_WORDS). Played like `cartesian_follow --joints FILE --pause 1`, with
# PANTHER_THROW_SPEED scaling the timing (2 = twice as fast).
THROW_DIR = Path(__file__).resolve().parents[6] / "motion" / "retarget"
THROW_FILES: dict[str, str] = {
    "standard": "joints_trajectory_shoulder_rel_3_v6.csv",  # v5 with the wrist -30 deg, -5 at release, 15 at the end
    "early": "joints_trajectory_shoulder_rel_3_v5_early.csv",  # gripper opens 0.2 s earlier
}
DEFAULT_THROW = "standard"
THROW_PAUSE_S = 1.0  # held at the first row before and after the gripper's first command

# Disk pickup ("pick up the disk" = pick_object, object_label "disk"): both arms through the joint-state dumps
# 1.txt..4.txt in HEADBands `motion/pickup/key_points/`, as `ros2 run panther_examples pickup` plays them.
PICKUP_DIR = Path(__file__).resolve().parents[6] / "motion" / "pickup" / "key_points"
PICKUP_PHASE_S = 8.0  # per key pose, at least (slower if a joint would pass 0.9 x MAX_ARM_JOINT_VEL_RPS)
PICKUP_PAUSE_S = 1.0  # held between key poses
PICKUP_LIFT_M = 0.2  # lift height for the whole pickup
PICKUP_SCRAPE_GRIP = 0.5  # right gripper after the scrape (pose 3); closed when the move away (pose 4) starts

PANTHER_COMMAND_TOPICS: tuple[str, ...] = (
    TOPIC_CMD_VEL,
    TOPIC_PREEMPT_TELEOP,
    TOPIC_MASTER_TRAJ,
    TOPIC_HEAD_COMMANDS,
    TOPIC_LIFT_COMMANDS,
    TOPIC_LEFT_GRIPPER_CMD,
    TOPIC_RIGHT_GRIPPER_CMD,
)

PANTHER_SUBSCRIBED_TOPICS: tuple[str, ...] = (
    TOPIC_BATTERY_STATE,
    TOPIC_ODOM,
    TOPIC_DIAGNOSTICS,
    TOPIC_ROBOT_DESCRIPTION,
    TOPIC_JOINT_STATES,
    TOPIC_LIFT_STATES,
    TOPIC_LEFT_GRIPPER_STATES,
    TOPIC_RIGHT_GRIPPER_STATES,
    TOPIC_FRONT_SCAN,
    TOPIC_CAMERA_COLOR_RAW,
    TOPIC_CAMERA_DEPTH_RAW,
)


def can_service(joint: str, verb: str) -> str:
    """CANopen Trigger service of an arm joint, e.g. `/can0/joint_la0/enable` (from `panther_examples/robot.py`)."""
    bus = "can0" if joint.startswith("joint_la") else "can1"
    return f"/{bus}/{joint}/{verb}"


def single_position(msg: Any) -> float | None:
    """Extract a valid single position from a Panther gripper or lift `JointState` message.

    Matches `panther_examples/robot.py`: about half the robot's gripper messages carry `-32.767`
    (`int16 -32767 / 1000`); real values are `0..1` (gripper) and `0..0.8 m` (lift).
    """
    positions = list(getattr(msg, "position", []) or [])
    if not positions:
        return None
    val = float(positions[0])
    return val if val > -1.0 else None
