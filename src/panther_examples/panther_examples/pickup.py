#!/usr/bin/env python3
"""Frisbee pickup using a saved SEDS model or Cartesian IK."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as RosPath
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from unix_msgs.msg import GripperCommand

from panther_examples.cartesian_follow import Chain, PlanningError, plan, path_msg, timed
from panther_examples.robot import (
    FORWARD,
    LIFT_CMD_TOPIC,
    LEFT_GRIPPER_CMD_TOPIC,
    LEFT_TRAJ,
    RIGHT_GRIPPER_CMD_TOPIC,
    RIGHT_TRAJ,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PICKUP_MOTION = REPO_ROOT / "motion" / "pickup"
if str(PICKUP_MOTION) not in sys.path:
    sys.path.insert(0, str(PICKUP_MOTION))
from camera_calibration import CameraFloorCalibration  # noqa: E402
from pickup_seds import SEDSModel  # noqa: E402

MAX_JOINT_VEL = 0.5
DEFAULT_HOME = np.array([0.0, -1.0, -0.9, 0.0, -1.2, 0.0, 0.0, 0.0])
DEFAULT_FLOOR_Z = 0.05
DEFAULT_APPROACH_Z = 0.25
DEFAULT_BODY_HEIGHT = 0.4
DEFAULT_HOME_SECONDS = 4.0
DEFAULT_APPROACH_SECONDS = 4.0
DEFAULT_DESCEND_SECONDS = 2.0
DEFAULT_GRIPPER_POSITION = 1.0
DEFAULT_START_FILE = Path("/workspace/motion/pickup/start.txt")
DEFAULT_HARDCODED_PATH = Path("/workspace/motion/pickup/hardcoded_pickup.csv")
DEFAULT_KEY_POINTS = Path("/workspace/motion/pickup/pickup_key_points.txt")
DEFAULT_KEY_POINT_DIR = Path("/workspace/motion/pickup/key_points")


def load_home_joints(path: Path, arm: str) -> np.ndarray:
    values = {}
    prefix = "ra" if arm == "right" else "la"
    pattern = re.compile(rf"^({prefix}[0-7])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$")
    if not path.is_file():
        raise SystemExit(f"home position file not found: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        match = pattern.match(line)
        if match:
            values[match.group(1)] = float(match.group(2))
    missing = [f"{prefix}{index}" for index in range(8) if f"{prefix}{index}" not in values]
    if missing:
        raise SystemExit(f"{path}: missing home joints: {', '.join(missing)}")
    return np.array([values[f"{prefix}{index}"] for index in range(8)], dtype=float)


def load_hardcoded_path(path: Path) -> np.ndarray:
    if not path.is_file():
        raise SystemExit(f"hardcoded path not found: {path}")
    rows = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    if rows.shape[1] != 7 or len(rows) < 2 or np.any(np.diff(rows[:, 0]) <= 0):
        raise SystemExit(f"{path}: expected increasing t,x,y,z,roll,pitch,yaw CSV")
    return rows


def load_joint_keyframes(path: Path) -> list[dict[str, float]]:
    text = path.read_text(encoding="utf-8").splitlines()
    keyframes = []
    index = 0
    while index < len(text):
        if text[index].strip() != "name:":
            index += 1
            continue
        names = []
        index += 1
        while index < len(text) and text[index].strip().startswith("-"):
            names.append(text[index].strip()[1:].strip())
            index += 1
        while index < len(text) and text[index].strip() != "position:":
            index += 1
        if index >= len(text):
            break
        positions = []
        index += 1
        while index < len(text) and text[index].strip().startswith("-"):
            positions.append(float(text[index].strip()[1:].strip()))
            index += 1
        if len(names) == len(positions) and names:
            keyframes.append(dict(zip(names, positions)))
    if not keyframes:
        raise SystemExit(f"{path}: no ROS joint-state keyframes found")
    return keyframes


def load_joint_pose(path: Path) -> dict[str, float]:
    frames = load_joint_keyframes(path)
    if len(frames) != 1:
        raise SystemExit(f"{path}: expected exactly one joint-state snapshot")
    return frames[0]


def keyframe_joints(chain: Chain, keyframe: dict[str, float]) -> np.ndarray:
    missing = [name for name in chain.names if name not in keyframe]
    if missing:
        raise SystemExit(f"keyframe is missing joints: {', '.join(missing)}")
    return np.array([keyframe[name] for name in chain.names], dtype=float)


def adjust_final_z(chain: Chain, joints: np.ndarray, z: float | None) -> np.ndarray:
    if z is None:
        return joints
    import PyKDL as kdl

    current = chain.fk(joints)
    target = kdl.Frame(current.M, kdl.Vector(current.p.x(), current.p.y(), z))
    adjusted = chain.ik(target, joints)
    if adjusted is None:
        raise SystemExit(f"could not adjust final gripper Z to {z:.3f} m without losing IK reachability")
    return adjusted


def plan_joint_keyframes(chain: Chain, current: np.ndarray, targets: list[np.ndarray], time_scale: float):
    times = np.arange(len(targets), dtype=float) * 4.0 * time_scale
    return timed(chain, targets, times, current, MAX_JOINT_VEL)


def target_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.target_xy is not None:
        return np.asarray(args.target_xy, dtype=float)
    if args.calibration is None:
        raise SystemExit("--target-pixel requires --calibration")
    return CameraFloorCalibration.load(args.calibration).pixel_to_robot(args.target_pixel)


def call_service(node, client, request):
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError(f"service unavailable: {client.srv_name}")
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    if future.result() is None:
        raise RuntimeError(f"service failed: {client.srv_name}")
    return future.result()


def switch_controllers(node, activate: list[str], deactivate: list[str]) -> None:
    if not activate and not deactivate:
        return
    client = node.create_client(SwitchController, "/controller_manager/switch_controller")
    request = SwitchController.Request(
        activate_controllers=activate,
        deactivate_controllers=deactivate,
        strictness=SwitchController.Request.STRICT,
        activate_asap=True,
    )
    request.timeout.sec = 5
    result = call_service(node, client, request)
    if not result.ok:
        raise RuntimeError(f"controller switch failed: activate={activate}, deactivate={deactivate}")


def controller_states(node) -> dict[str, int]:
    result = call_service(node, node.create_client(ListControllers, "/controller_manager/list_controllers"), ListControllers.Request())
    return {controller.name: controller.state for controller in result.controller}


def send_joint_goal(node, action_client, joint_names: list[str], positions: np.ndarray, seconds: float) -> None:
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = joint_names
    point = JointTrajectoryPoint(positions=positions.tolist())
    point.time_from_start = Duration(sec=int(seconds), nanosec=int((seconds % 1.0) * 1e9))
    goal.trajectory.points = [point]
    if not action_client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError("arm trajectory action server unavailable")
    goal_future = action_client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, goal_future)
    handle = goal_future.result()
    if handle is None or not handle.accepted:
        raise RuntimeError("home trajectory was rejected")
    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result().result
    if result.error_code != 0:
        raise RuntimeError(f"home trajectory failed: {result.error_string}")


def publish_lift(node, height: float, seconds: float = 1.0) -> None:
    publisher = node.create_publisher(JointState, LIFT_CMD_TOPIC, 10)
    message = JointState(name=["body_joint"], position=[float(height)])
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        publisher.publish(message)
        rclpy.spin_once(node, timeout_sec=0.05)


def publish_gripper(node, topic: str, position: float) -> None:
    publisher = node.create_publisher(GripperCommand, topic, 10)
    message = GripperCommand()
    message.mode = "parallel"
    message.command.name = [""]
    message.command.position = [float(position)]
    message.command.velocity = [0.5]
    message.command.effort = [0.5]
    publisher.publish(message)
    rclpy.spin_once(node, timeout_sec=0.1)


def ramp_gripper(node, topic: str, start: float, end: float, seconds: float) -> None:
    publisher = node.create_publisher(GripperCommand, topic, 10)
    steps = max(2, int(np.ceil(seconds * 20.0)))
    for value in np.linspace(start, end, steps):
        message = GripperCommand()
        message.mode = "parallel"
        message.command.name = [""]
        message.command.position = [float(value)]
        message.command.velocity = [0.5]
        message.command.effort = [0.5]
        publisher.publish(message)
        rclpy.spin_once(node, timeout_sec=max(0.01, seconds / steps))


def plan_with_retiming(chain, poses, times, start_joints):
    last_error = None
    for scale in (1.0, 2.0, 4.0, 8.0):
        try:
            planned = plan(chain, poses, times * scale, start_joints, MAX_JOINT_VEL, start_joints)
            if scale > 1.0:
                print(f"joint-speed limit required retiming trajectory by {scale:.1f}x")
            return planned
        except PlanningError as error:
            last_error = error
            if "would reach" not in str(error):
                raise
    raise last_error


def send_dual_goals(node, left_client, right_client, left_goal, right_goal):
    if not left_client.wait_for_server(timeout_sec=5.0) or not right_client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError("one or both arm trajectory action servers are unavailable")
    left_future = left_client.send_goal_async(left_goal)
    right_future = right_client.send_goal_async(right_goal)
    rclpy.spin_until_future_complete(node, left_future)
    rclpy.spin_until_future_complete(node, right_future)
    left_handle, right_handle = left_future.result(), right_future.result()
    if left_handle is None or right_handle is None or not left_handle.accepted or not right_handle.accepted:
        raise RuntimeError("one or both scrape trajectory goals were rejected")
    left_result = left_handle.get_result_async()
    right_result = right_handle.get_result_async()
    rclpy.spin_until_future_complete(node, left_result)
    rclpy.spin_until_future_complete(node, right_result)
    results = (left_result.result().result, right_result.result().result)
    if any(result.error_code != 0 for result in results):
        raise RuntimeError("scrape trajectory failed: " + "; ".join(result.error_string for result in results if result.error_string))


def make_joint_goal(chain: Chain, positions: np.ndarray, seconds: float) -> FollowJointTrajectory.Goal:
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = chain.names
    point = JointTrajectoryPoint(positions=positions.tolist())
    point.time_from_start = Duration(sec=int(seconds), nanosec=int((seconds % 1.0) * 1e9))
    goal.trajectory.points = [point]
    return goal


def clamp_keyframe(chain: Chain, joints: np.ndarray, side: str, phase: str) -> np.ndarray:
    clipped = np.clip(joints, chain.lo, chain.hi)
    for index, (value, adjusted, lower, upper) in enumerate(zip(joints, clipped, chain.lo, chain.hi)):
        if value != adjusted:
            print(f"WARNING: {phase} {side} {chain.names[index]}={value:.6f} exceeds URDF limits "
                  f"[{lower:.6f}, {upper:.6f}]; clamping to {adjusted:.6f}")
    return clipped


def run_keypoint_phase(node, state, urdf, phase: str, key_point_dir: Path, body_height: float, time_scale: float, dry_run: bool, lower_before: bool = True, lower_after: bool = False, lower_height: float | None = None, scrape_grip: float = 0.5, gripper_close_seconds: float = 2.0) -> None:
    phase_index = {
        "scrape_prepare_position": "1.txt",
        "scrape_prepare": "2.txt",
        "scrape_execute": "3.txt",
        "scrape_move_away": "4.txt",
    }[phase]
    keyframe = load_joint_pose(key_point_dir / phase_index)
    left_chain = Chain(urdf, "body_link", "link_lf")
    right_chain = Chain(urdf, "body_link", "link_rf")
    left_joints = clamp_keyframe(left_chain, keyframe_joints(left_chain, keyframe), "left", phase)
    right_joints = clamp_keyframe(right_chain, keyframe_joints(right_chain, keyframe), "right", phase)
    path_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_publisher(RosPath, f"/sim_frisbee_pickup/{phase}/left", path_qos).publish(path_msg(left_chain, np.array([left_joints]), node.get_clock().now().to_msg()))
    node.create_publisher(RosPath, f"/sim_frisbee_pickup/{phase}/right", path_qos).publish(path_msg(right_chain, np.array([right_joints]), node.get_clock().now().to_msg()))
    print(f"{phase}: key point {phase_index}, body height {body_height:.3f}")
    print(f"left joints: {left_joints.tolist()}")
    print(f"right joints: {right_joints.tolist()}")
    if dry_run:
        print(f"dry-run: {phase} joint goal prepared; paths published under /sim_frisbee_pickup/{phase}")
        return
    states = controller_states(node)
    switch_controllers(node, [LEFT_TRAJ, RIGHT_TRAJ], [FORWARD] if states.get(FORWARD) == "active" else [])
    print(f"{phase}: using {LEFT_TRAJ} and {RIGHT_TRAJ}; {FORWARD} released")
    left_client = ActionClient(node, FollowJointTrajectory, f"/{LEFT_TRAJ}/follow_joint_trajectory")
    right_client = ActionClient(node, FollowJointTrajectory, f"/{RIGHT_TRAJ}/follow_joint_trajectory")
    try:
        if lower_before:
            publish_lift(node, body_height)
        if phase in ("scrape_prepare_position", "scrape_prepare"):
            publish_gripper(node, LEFT_GRIPPER_CMD_TOPIC, 0.0)
            publish_gripper(node, RIGHT_GRIPPER_CMD_TOPIC, 0.0)
        seconds = 4.0 * time_scale
        send_dual_goals(node, left_client, right_client, make_joint_goal(left_chain, left_joints, seconds), make_joint_goal(right_chain, right_joints, seconds))
        if lower_after:
            publish_lift(node, body_height if lower_height is None else lower_height)
        if phase == "scrape_execute":
            publish_gripper(node, RIGHT_GRIPPER_CMD_TOPIC, scrape_grip)
            print(f"scrape_execute complete: right gripper at {scrape_grip:.2f}, left gripper open")
        elif phase == "scrape_prepare_position":
            print("scrape_prepare_position complete: both grippers open")
        elif phase == "scrape_prepare":
            print("scrape_prepare complete: both grippers open")
        else:
            ramp_gripper(node, RIGHT_GRIPPER_CMD_TOPIC, scrape_grip, DEFAULT_GRIPPER_POSITION, gripper_close_seconds)
            print("scrape_move_away complete: right gripper fully closed, left gripper open")
    finally:
        switch_controllers(node, [FORWARD], [LEFT_TRAJ, RIGHT_TRAJ])


def pickup_path(model: SEDSModel | None, start: np.ndarray, target_xy: np.ndarray, floor_z: float, approach_height: float, approach_seconds: float, vertical_seconds: float, descend_seconds: float, pickup_rpy: tuple[float, float, float]) -> np.ndarray:
    approach_target = np.array([target_xy[0], target_xy[1], floor_z + approach_height])
    if model is not None:
        approach = model.generate_trajectory(start, target_xy, floor_z + approach_height, approach_seconds, 0.05, pickup_rpy)
    else:
        safe_z = max(float(start[2]), float(approach_target[2]))
        vertical_seconds = min(vertical_seconds, max(0.05, approach_seconds - 0.05))
        horizontal_seconds = max(0.1, approach_seconds - vertical_seconds)
        def segment(begin, end, duration):
            count = max(2, int(np.ceil(duration / 0.05)) + 1)
            segment_times = np.linspace(0.0, duration, count)
            segment_positions = np.linspace(begin, end, count)
            return np.column_stack([
                segment_times,
                segment_positions,
                np.repeat(np.asarray(pickup_rpy)[None, :], count, axis=0),
            ])
        lift = segment(start, np.array([start[0], start[1], safe_z]), vertical_seconds)
        translate = segment(lift[-1, 1:4], np.array([target_xy[0], target_xy[1], safe_z]), horizontal_seconds)
        approach = np.vstack([lift, translate[1:]])
        approach[:, 0] = np.arange(len(approach)) * 0.05
    descend_count = max(2, int(np.ceil(descend_seconds / 0.05)) + 1)
    descend_times = np.linspace(0.05, descend_seconds, descend_count)
    descend_z = np.linspace(approach[-1, 3], floor_z, descend_count)
    descend = np.column_stack([
        approach[-1, 0] + descend_times,
        np.full(descend_count, target_xy[0]),
        np.full(descend_count, target_xy[1]),
        descend_z,
         np.repeat(np.asarray(pickup_rpy)[None, :], descend_count, axis=0),
    ])
    rows = np.vstack([approach, descend])
    return rows


def requested_path_msg(rows: np.ndarray, stamp) -> RosPath:
    message = RosPath()
    message.header.frame_id = "body_link"
    message.header.stamp = stamp
    for row in rows:
        pose = PoseStamped()
        pose.header = message.header
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = row[1:4]
        pose.pose.orientation.w = 1.0
        message.poses.append(pose)
    return message


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("start", "hardcoded", "fetch", "scrape_sequence", "scrape_prepare_position", "scrape_prepare", "scrape_execute", "scrape_move_away"), nargs="?", default="fetch", help="start home, follow hardcoded path, fetch, or play pickup key points")
    parser.add_argument("--model", type=Path, help="saved SEDS model used only with --use-seds")
    parser.add_argument("--use-seds", action="store_true", help="opt into the learned SEDS approach; default is deterministic Cartesian pickup")
    target = parser.add_mutually_exclusive_group(required=False)
    target.add_argument("--target-xy", type=float, nargs=2, metavar=("X", "Y"))
    target.add_argument("--target-pixel", type=float, nargs=2, metavar=("U", "V"))
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--start-file", type=Path, default=DEFAULT_START_FILE, help="home joint file with la0..la7 and ra0..ra7")
    parser.add_argument("--hardcoded-path", type=Path, default=DEFAULT_HARDCODED_PATH)
    parser.add_argument("--key-points", type=Path, default=DEFAULT_KEY_POINTS, help="ROS joint-state snapshots used by scrape phases")
    parser.add_argument("--key-point-dir", type=Path, default=DEFAULT_KEY_POINT_DIR, help="directory containing 1.txt through 4.txt pickup poses")
    parser.add_argument("--left-z", type=float, help="left scrape hand Z in body_link metres")
    parser.add_argument("--right-z", type=float, help="right scrape hand Z in body_link metres")
    parser.add_argument("--time-scale", type=float, default=1.0, help="multiply trajectory durations; 2.0 makes motion twice as slow")
    parser.add_argument("--between-delay", type=float, default=1.0, help="seconds to pause between combined scrape phases")
    parser.add_argument("--scrape-grip", type=float, default=0.5, help="right gripper position after scrape_execute")
    parser.add_argument("--gripper-close-seconds", type=float, default=2.0, help="time to close right gripper fully during move-away")
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--body-height", type=float, default=None, help="lift height in metres; start defaults to 0.4, fetch defaults to 0.0")
    parser.add_argument("--lower-body-height", type=float, default=0.0, help="body height after hand preparation before scrape execution")
    parser.add_argument("--floor-z", type=float, default=DEFAULT_FLOOR_Z, help="hand clearance above base floor in metres; converted to body_link")
    parser.add_argument("--approach-height", type=float, default=DEFAULT_APPROACH_Z, help="height above floor before descending")
    parser.add_argument("--home-joints", type=float, nargs=8, default=None, metavar="Q", help="override start-file with exactly 8 values")
    parser.add_argument("--start-xyz", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Cartesian hand start in body_link; defaults to FK(home-joints)")
    parser.add_argument("--home-seconds", type=float, default=DEFAULT_HOME_SECONDS)
    parser.add_argument("--approach-seconds", type=float, default=6.0, help="vertical plus lateral approach time; tuned CSV uses 6 seconds")
    parser.add_argument("--vertical-seconds", type=float, default=2.0, help="initial vertical stage time; tuned CSV uses 2 seconds")
    parser.add_argument("--descend-seconds", type=float, default=DEFAULT_DESCEND_SECONDS)
    parser.add_argument("--dry-run", action="store_true", help="publish the plan to RViz without moving or closing the gripper")
    args = parser.parse_args()
    if args.phase == "fetch" and args.target_xy is None and args.target_pixel is None:
        parser.error("fetch requires --target-xy or --target-pixel")
    if args.phase == "fetch" and args.use_seds and (args.model is None or not args.model.is_file()):
        raise SystemExit(f"model not found: {args.model}")
    if args.body_height is None:
        args.body_height = 0.2 if args.phase in ("scrape_sequence", "scrape_prepare_position", "scrape_prepare", "scrape_execute", "scrape_move_away") else DEFAULT_BODY_HEIGHT if args.phase == "start" else 0.0
    if not 0.0 <= args.body_height <= 0.8:
        raise SystemExit("--body-height must be between 0 and 0.8 metres")
    if not 0.0 <= args.lower_body_height <= 0.8:
        raise SystemExit("--lower-body-height must be between 0 and 0.8 metres")
    if args.floor_z < 0 or args.approach_height <= 0:
        raise SystemExit("floor clearance must be non-negative and approach height must be positive")
    if args.time_scale <= 0:
        raise SystemExit("--time-scale must be positive")
    if not 0.0 <= args.scrape_grip <= DEFAULT_GRIPPER_POSITION:
        raise SystemExit(f"--scrape-grip must be between 0 and {DEFAULT_GRIPPER_POSITION}")
    if args.gripper_close_seconds <= 0:
        raise SystemExit("--gripper-close-seconds must be positive")

    rclpy.init()
    node = rclpy.create_node("sim_frisbee_pickup")
    state: dict[str, object] = {}
    node.create_subscription(JointState, "/joint_states", lambda message: state.setdefault("joints", dict(zip(message.name, message.position))), 10)
    urdf: dict[str, str] = {}
    from std_msgs.msg import String
    node.create_subscription(String, "/robot_description", lambda message: urdf.setdefault("xml", message.data), QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    for _ in range(100):
        if "joints" in state and "xml" in urdf:
            break
        rclpy.spin_once(node, timeout_sec=0.1)
    if "joints" not in state or "xml" not in urdf:
        raise RuntimeError("did not receive /joint_states and /robot_description; start the mock sim first")

    chain = Chain(urdf["xml"], "body_link", f"link_{args.arm[0]}f")
    base_chain = Chain(urdf["xml"], "base_footprint", "body_link")
    import PyKDL as kdl
    home = load_home_joints(args.start_file, args.arm) if args.home_joints is None else np.asarray(args.home_joints, dtype=float)
    if np.any(home < chain.lo) or np.any(home > chain.hi):
        raise SystemExit(f"--home-joints outside URDF limits: {chain.lo}..{chain.hi}")
    arm_controller = LEFT_TRAJ if args.arm == "left" else RIGHT_TRAJ
    gripper_topic = LEFT_GRIPPER_CMD_TOPIC if args.arm == "left" else RIGHT_GRIPPER_CMD_TOPIC
    states = controller_states(node)
    if args.phase == "start":
        if args.dry_run:
            print(f"start dry-run: body height={args.body_height:.3f}, home joints={home.tolist()}")
            return
        switch_controllers(node, [arm_controller], [FORWARD] if states.get(FORWARD) == "active" else [])
        action_client = ActionClient(node, FollowJointTrajectory, f"/{arm_controller}/follow_joint_trajectory")
        try:
            publish_lift(node, args.body_height)
            publish_gripper(node, gripper_topic, 0.0)
            send_joint_goal(node, action_client, chain.names, home, args.home_seconds)
            print(f"start complete: body height={args.body_height:.3f}, arm={args.arm} home, gripper open")
        finally:
            switch_controllers(node, [FORWARD], [arm_controller])
            node.destroy_node()
            rclpy.shutdown()
        return

    keypoint_phases = ("scrape_prepare_position", "scrape_prepare", "scrape_execute", "scrape_move_away")
    if args.phase == "scrape_sequence" or args.phase in keypoint_phases:
        phases = keypoint_phases if args.phase == "scrape_sequence" else (args.phase,)
        if args.phase == "scrape_sequence" and not args.dry_run:
            publish_lift(node, args.body_height)
        for index, phase in enumerate(phases):
            run_keypoint_phase(
                node,
                state,
                urdf["xml"],
                phase,
                args.key_point_dir,
                args.body_height,
                args.time_scale,
                args.dry_run,
                lower_before=args.phase != "scrape_sequence",
                lower_after=args.phase == "scrape_sequence" and phase == "scrape_prepare_position",
                lower_height=args.lower_body_height,
                scrape_grip=args.scrape_grip,
                gripper_close_seconds=args.gripper_close_seconds,
            )
            if not args.dry_run and index < len(phases) - 1:
                if args.between_delay < 0:
                    raise SystemExit("--between-delay must be non-negative")
                deadline = time.monotonic() + args.between_delay
                while time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        rclpy.shutdown()
        return

    if args.phase in ("scrape_prepare", "scrape_from_ground"):
        keyframes = load_joint_keyframes(args.key_points)
        if len(keyframes) < 3:
            raise SystemExit(f"{args.key_points}: expected at least 3 keyframes")
        left_chain = Chain(urdf["xml"], "body_link", "link_lf")
        right_chain = Chain(urdf["xml"], "body_link", "link_rf")
        left_joints = np.array([state["joints"][name] for name in left_chain.names], dtype=float)
        right_joints = np.array([state["joints"][name] for name in right_chain.names], dtype=float)
        if args.phase == "scrape_prepare":
            left_targets = [keyframe_joints(left_chain, keyframes[0]), keyframe_joints(left_chain, keyframes[1])]
            right_targets = [keyframe_joints(right_chain, keyframes[0]), keyframe_joints(right_chain, keyframes[1])]
        else:
            left_targets = [adjust_final_z(left_chain, keyframe_joints(left_chain, keyframes[2]), args.left_z)]
            right_targets = [adjust_final_z(right_chain, keyframe_joints(right_chain, keyframes[2]), args.right_z)]
        left_plan = plan_joint_keyframes(left_chain, left_joints, left_targets, args.time_scale)
        right_plan = plan_joint_keyframes(right_chain, right_joints, right_targets, args.time_scale)
        left_qs, left_times, left_velocities = left_plan
        right_qs, right_times, right_velocities = right_plan
        path_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_publisher(RosPath, "/sim_frisbee_pickup/left_scrape_path", path_qos).publish(path_msg(left_chain, left_qs, node.get_clock().now().to_msg()))
        node.create_publisher(RosPath, "/sim_frisbee_pickup/right_scrape_path", path_qos).publish(path_msg(right_chain, right_qs, node.get_clock().now().to_msg()))
        left_target = left_chain.fk(left_qs[-1])
        right_target = right_chain.fk(right_qs[-1])
        print(f"{args.phase} joint keyframes: {args.key_points}")
        print(f"left final FK: xyz={list(left_target.p)}, rpy_deg={np.degrees(left_target.M.GetRPY()).tolist()}")
        print(f"right final FK: xyz={list(right_target.p)}, rpy_deg={np.degrees(right_target.M.GetRPY()).tolist()}")
        left_actual = left_chain.fk(left_qs[-1])
        right_actual = right_chain.fk(right_qs[-1])
        print(f"left planned FK: xyz={list(left_actual.p)}, rpy_deg={np.degrees(left_actual.M.GetRPY()).tolist()}")
        print(f"right planned FK: xyz={list(right_actual.p)}, rpy_deg={np.degrees(right_actual.M.GetRPY()).tolist()}")
        if args.dry_run:
            print(f"dry-run: {args.phase} paths published on /sim_frisbee_pickup/left_scrape_path and /sim_frisbee_pickup/right_scrape_path")
            try:
                rclpy.spin(node)
            except KeyboardInterrupt:
                pass
            return
        states = controller_states(node)
        switch_controllers(node, [LEFT_TRAJ, RIGHT_TRAJ], [FORWARD] if states.get(FORWARD) == "active" else [])
        left_client = ActionClient(node, FollowJointTrajectory, f"/{LEFT_TRAJ}/follow_joint_trajectory")
        right_client = ActionClient(node, FollowJointTrajectory, f"/{RIGHT_TRAJ}/follow_joint_trajectory")

        def make_goal(chain, qs, velocities, times):
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = chain.names
            for joints, speed, timestamp in zip(qs, velocities, times):
                point = JointTrajectoryPoint(positions=joints.tolist(), velocities=speed.tolist())
                point.time_from_start = Duration(sec=int(timestamp), nanosec=int((timestamp % 1) * 1e9))
                goal.trajectory.points.append(point)
            return goal

        try:
            publish_lift(node, args.body_height)
            publish_gripper(node, LEFT_GRIPPER_CMD_TOPIC, 0.0)
            publish_gripper(node, RIGHT_GRIPPER_CMD_TOPIC, 0.0)
            send_dual_goals(node, left_client, right_client, make_goal(left_chain, left_qs, left_velocities, left_times), make_goal(right_chain, right_qs, right_velocities, right_times))
            if args.phase == "scrape_from_ground":
                publish_gripper(node, RIGHT_GRIPPER_CMD_TOPIC, DEFAULT_GRIPPER_POSITION)
                print("scrape_from_ground complete: right gripper closed, left gripper open")
            else:
                print("scrape_prepare complete: both grippers open")
        finally:
            switch_controllers(node, [FORWARD], [LEFT_TRAJ, RIGHT_TRAJ])
            node.destroy_node()
            rclpy.shutdown()
        return
	
    if args.phase == "hardcoded":
        cartesian_rows = load_hardcoded_path(args.hardcoded_path)
        target_xy = None
    else:
        target_xy = target_from_args(args)
    body_frame = base_chain.fk([args.body_height])
    floor_z_body = args.floor_z - float(body_frame.p[2])
    live_joints = np.array([state["joints"][name] for name in chain.names], dtype=float)
    current_rotation = chain.fk(live_joints).M
    start_joints = live_joints
    start_position = np.array(list(chain.fk(start_joints).p), dtype=float)
    if args.start_xyz is not None:
        start_position = np.asarray(args.start_xyz, dtype=float)
        start_frame = kdl.Frame(current_rotation, kdl.Vector(*start_position))
        start_joints = chain.ik(start_frame, home)
        if start_joints is None:
            raise SystemExit(f"--start-xyz is not reachable from the home seed: {start_position.tolist()}")
    if args.phase == "fetch":
        model = SEDSModel.load(args.model) if args.use_seds else None
        cartesian_rows = pickup_path(model, start_position, target_xy, floor_z_body, args.approach_height, args.approach_seconds, args.vertical_seconds, args.descend_seconds, (0.0, 0.0, 0.0))
    path_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    path_publisher = node.create_publisher(RosPath, "/cartesian_follow/planned_path", path_qos)
    requested_publisher = node.create_publisher(RosPath, "/sim_frisbee_pickup/requested_path", path_qos)
    print(f"body_link origin above base floor: {float(body_frame.p[2]):.3f} m; requested floor clearance: {args.floor_z:.3f} m")
    if target_xy is not None:
        print(f"target in body_link: x={target_xy[0]:.3f}, y={target_xy[1]:.3f}, floor z={floor_z_body:.3f} m")
    else:
        print(f"following hardcoded path: {args.hardcoded_path}")
    requested_publisher.publish(requested_path_msg(cartesian_rows, node.get_clock().now().to_msg()))
    if args.phase == "hardcoded":
        poses = [kdl.Frame(kdl.Rotation.RPY(*row[4:7]), kdl.Vector(*row[1:4])) for row in cartesian_rows]
    else:
        path_rotation = current_rotation
        poses = [kdl.Frame(path_rotation, kdl.Vector(*row[1:4])) for row in cartesian_rows]
    times = cartesian_rows[:, 0]
    times = times * args.time_scale
    try:
        qs, times, velocities = plan_with_retiming(chain, poses, times, start_joints)
    except PlanningError as error:
        if error.partial_qs is not None and len(error.partial_qs) > 0:
            path_publisher.publish(path_msg(chain, error.partial_qs, node.get_clock().now().to_msg()))
            print(f"Published reachable IK prefix: {len(error.partial_qs)} waypoints")
        else:
            path_publisher.publish(requested_path_msg(cartesian_rows, node.get_clock().now().to_msg()))
        if 0 <= error.waypoint < len(cartesian_rows):
            failed = cartesian_rows[error.waypoint]
            print(f"failed requested pose: t={failed[0]:.2f}, x={failed[1]:.3f}, y={failed[2]:.3f}, z={failed[3]:.3f}")
        print(f"IK refused the pickup path at waypoint {error.waypoint}")
        print(f"IK refused the pickup path: {error}")
        print("The requested Cartesian path was published to /cartesian_follow/planned_path for RViz inspection.")
        if args.dry_run:
            try:
                rclpy.spin(node)
            except KeyboardInterrupt:
                pass
            return
        raise
    path_publisher.publish(path_msg(chain, qs, node.get_clock().now().to_msg()))
    print(f"target body_link XY: {target_xy.tolist()}; floor Z: {floor_z_body:.3f}; body height: {args.body_height:.3f}")
    print(f"planned {len(qs)} Cartesian waypoints; approach clearance: {args.approach_height:.3f} m")
    if args.dry_run:
        print("dry-run: plan published on /cartesian_follow/planned_path; press Ctrl-C to exit")
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        return

    switch_controllers(node, [arm_controller], [FORWARD] if states.get(FORWARD) == "active" else [])
    action_client = ActionClient(node, FollowJointTrajectory, f"/{arm_controller}/follow_joint_trajectory")
    try:
        publish_lift(node, args.body_height)
        publish_gripper(node, gripper_topic, 0.0)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = chain.names
        for q, velocity, timestamp in zip(qs, velocities, times):
            point = JointTrajectoryPoint(positions=q.tolist(), velocities=velocity.tolist())
            point.time_from_start = Duration(sec=int(timestamp), nanosec=int((timestamp % 1) * 1e9))
            goal.trajectory.points.append(point)
        goal_future = action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, goal_future)
        handle = goal_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("pickup trajectory was rejected")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)
        result = result_future.result().result
        if result.error_code != 0:
            raise RuntimeError(f"pickup trajectory failed: {result.error_string}")
        publish_gripper(node, gripper_topic, DEFAULT_GRIPPER_POSITION)
        print("pickup complete: gripper closed")
    finally:
        switch_controllers(node, [FORWARD], [arm_controller])
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
