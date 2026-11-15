"""Panther names and limits shared by the robot scripts and the sim. Source: docs/ros2/README.md."""

LEFT_ARM = [f"joint_la{i}" for i in range(8)]
RIGHT_ARM = [f"joint_ra{i}" for i in range(8)]
ARMS = LEFT_ARM + RIGHT_ARM  # command order of the dual-arm forward controller; /joint_states order differs
HEAD = ["joint_head0", "joint_head1"]  # pitch, yaw

FORWARD = "dual_arm_forward_position_controller"
LEFT_TRAJ = "left_arm_joint_trajectory_position_controller"
RIGHT_TRAJ = "right_arm_joint_trajectory_position_controller"
HEAD_FORWARD = "head_forward_position_controller"

LEFT_HARDWARE = "unix_openarm_left"
RIGHT_HARDWARE = "unix_openarm_right"
LEFT_GRIPPER_HARDWARE = "raw_can0"
RIGHT_GRIPPER_HARDWARE = "raw_can1"

LEFT_GRIPPER_CONTROLLER = "left_unix_gripper3_controller"
RIGHT_GRIPPER_CONTROLLER = "right_unix_gripper3_controller"
LEFT_GRIPPER_CMD_TOPIC = "/left_gripper/gripper_command"
RIGHT_GRIPPER_CMD_TOPIC = "/right_gripper/gripper_command"
LEFT_GRIPPER_STATE_TOPIC = "/left_gripper/joint_states"
RIGHT_GRIPPER_STATE_TOPIC = "/right_gripper/joint_states"

LIFT_CONTROLLER = "lifting_controller"
LIFT_CMD_TOPIC = "/lifting_controller/commands"
LIFT_STATE_TOPIC = "/lifting_controller/joint_states"
# shortcut: 0.8 m is the SDK/user-guide travel; nobody has read the top end stop on the robot yet. Measure, then fix here.
MIN_LIFT, MAX_LIFT = 0.0, 0.8  # m, 0 = bottom
LIFT_VEL = 0.05  # m/s, SDK limit


def can_service(joint, verb):
    """CANopen Trigger service of an arm joint, e.g. /can0/joint_la0/enable."""
    return f"/{'can0' if joint.startswith('joint_la') else 'can1'}/{joint}/{verb}"
