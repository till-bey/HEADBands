#!/usr/bin/env python3
"""Disk pickup from the floor: plays the joint-space key poses motion/pickup/key_points/1.txt..4.txt on both arms.

Lift at --body-height (0.19 m) throughout. 1 prepare position (grippers open); 2 prepare; 3 scrape, right gripper to --scrape-grip;
4 move away, right gripper closed at its start. The web interface plays the same sequence through the bridge
("pick up the disk"; swiss_robot_bridge.local_safety.pickup_points).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint
from unix_msgs.msg import GripperCommand

from panther_examples.cartesian_follow import Chain
from panther_examples.robot import (
    FORWARD,
    LEFT_GRIPPER_CMD_TOPIC,
    LEFT_TRAJ,
    LIFT_CMD_TOPIC,
    RIGHT_GRIPPER_CMD_TOPIC,
    RIGHT_TRAJ,
)

DEFAULT_KEY_POINT_DIR = Path("/workspace/motion/pickup/key_points")
PHASES = ("1.txt", "2.txt", "3.txt", "4.txt")


def load_joint_pose(path: Path) -> dict[str, float]:
    """One `ros2 topic echo /joint_states --once` dump -> {joint: position}."""
    lists: dict[str, list[str]] = {}
    key = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.endswith(":") and not line.startswith(" "):
            key = line[:-1]
            lists[key] = []
        elif line.startswith("- ") and key is not None:
            lists[key].append(line[2:].strip())
    names, positions = lists.get("name", []), lists.get("position", [])
    if not names or len(names) != len(positions):
        raise SystemExit(f"{path}: expected one joint-state snapshot with name and position lists")
    return dict(zip(names, map(float, positions)))


def arm_joints(chain: Chain, pose: dict[str, float], label: str) -> np.ndarray:
    missing = [name for name in chain.names if name not in pose]
    if missing:
        raise SystemExit(f"{label}: missing joints {', '.join(missing)}")
    q = np.array([pose[name] for name in chain.names])
    clipped = np.clip(q, chain.lo, chain.hi)
    for name, a, b in zip(chain.names, q, clipped):
        if a != b:
            print(f"WARNING: {label} {name}={a:.4f} outside URDF limits, clamped to {b:.4f}")
    return clipped


def call_service(node, client, request):
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError(f"service unavailable: {client.srv_name}")
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    if future.result() is None:
        raise RuntimeError(f"service failed: {client.srv_name}")
    return future.result()


def switch_controllers(node, activate: list[str], deactivate: list[str]) -> None:
    request = SwitchController.Request(
        activate_controllers=activate,
        deactivate_controllers=deactivate,
        strictness=SwitchController.Request.STRICT,
        activate_asap=True,
    )
    request.timeout.sec = 5
    if not call_service(node, node.create_client(SwitchController, "/controller_manager/switch_controller"), request).ok:
        raise RuntimeError(f"controller switch failed: activate={activate}, deactivate={deactivate}")


def forward_active(node) -> bool:
    result = call_service(node, node.create_client(ListControllers, "/controller_manager/list_controllers"), ListControllers.Request())
    return any(c.name == FORWARD and c.state == "active" for c in result.controller)


def spin_for(node, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)


def publish_lift(node, height: float) -> None:
    publisher = node.create_publisher(JointState, LIFT_CMD_TOPIC, 10)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        publisher.publish(JointState(name=["body_joint"], position=[float(height)]))
        rclpy.spin_once(node, timeout_sec=0.05)


def publish_gripper(node, topic: str, position: float) -> None:
    message = GripperCommand()
    message.mode = "parallel"
    message.command.name = [""]
    message.command.position = [float(position)]
    message.command.velocity = [0.5]
    message.command.effort = [0.5]
    node.create_publisher(GripperCommand, topic, 10).publish(message)
    rclpy.spin_once(node, timeout_sec=0.1)


def make_goal(names: list[str], positions: np.ndarray, seconds: float) -> FollowJointTrajectory.Goal:
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = names
    point = JointTrajectoryPoint(positions=positions.tolist())
    point.time_from_start = Duration(sec=int(seconds), nanosec=int((seconds % 1.0) * 1e9))
    goal.trajectory.points = [point]
    return goal


def run_dual_goals(node, clients, goals) -> None:
    if not all(client.wait_for_server(timeout_sec=5.0) for client in clients):
        raise RuntimeError("arm trajectory action servers unavailable")
    sent = [client.send_goal_async(goal) for client, goal in zip(clients, goals)]
    for future in sent:
        rclpy.spin_until_future_complete(node, future)
    handles = [future.result() for future in sent]
    if any(h is None or not h.accepted for h in handles):
        raise RuntimeError("pickup trajectory goal rejected")
    results = [h.get_result_async() for h in handles]
    for future in results:
        rclpy.spin_until_future_complete(node, future)
    errors = [f.result().result.error_string for f in results if f.result().result.error_code != 0]
    if errors:
        raise RuntimeError("pickup trajectory failed: " + "; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-point-dir", type=Path, default=DEFAULT_KEY_POINT_DIR)
    parser.add_argument("--body-height", type=float, default=0.19, help="lift height for the whole pickup (m)")
    parser.add_argument("--scrape-grip", type=float, default=0.5, help="right gripper after the scrape (0 open..1 closed)")
    parser.add_argument("--gripper-close-seconds", type=float, default=0.5, help="right gripper close ramp before the move away")
    parser.add_argument("--phase-seconds", type=float, default=8.0, help="time per key pose")
    parser.add_argument("--between-delay", type=float, default=1.0, help="pause between key poses (s)")
    parser.add_argument("--dry-run", action="store_true", help="load and check the key poses, do not move")
    args = parser.parse_args()
    if not 0.0 <= args.body_height <= 0.8:
        raise SystemExit("--body-height must be between 0 and 0.8 m")
    if not 0.0 <= args.scrape_grip <= 1.0 or args.phase_seconds <= 0 or args.between_delay < 0 or args.gripper_close_seconds <= 0:
        raise SystemExit("--scrape-grip must be in [0, 1], --phase-seconds positive, --between-delay non-negative")

    rclpy.init()
    node = rclpy.create_node("disk_pickup")
    urdf: dict[str, str] = {}
    node.create_subscription(String, "/robot_description", lambda m: urdf.setdefault("xml", m.data),
                             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    for _ in range(100):
        if "xml" in urdf:
            break
        rclpy.spin_once(node, timeout_sec=0.1)
    if "xml" not in urdf:
        raise RuntimeError("no /robot_description; start the robot or the mock sim first")
    chains = (Chain(urdf["xml"], "body_link", "link_lf"), Chain(urdf["xml"], "body_link", "link_rf"))
    poses = []
    for name in PHASES:
        pose = load_joint_pose(args.key_point_dir / name)
        poses.append([arm_joints(chain, pose, f"{name} {side}") for chain, side in zip(chains, ("left", "right"))])
    if args.dry_run:
        for name, (left, right) in zip(PHASES, poses):
            print(f"{name}: left {np.round(left, 3).tolist()} right {np.round(right, 3).tolist()}")
        return

    switch_controllers(node, [LEFT_TRAJ, RIGHT_TRAJ], [FORWARD] if forward_active(node) else [])
    clients = [ActionClient(node, FollowJointTrajectory, f"/{c}/follow_joint_trajectory") for c in (LEFT_TRAJ, RIGHT_TRAJ)]
    try:
        publish_lift(node, args.body_height)
        publish_gripper(node, LEFT_GRIPPER_CMD_TOPIC, 0.0)
        publish_gripper(node, RIGHT_GRIPPER_CMD_TOPIC, 0.0)
        for index, (name, joints) in enumerate(zip(PHASES, poses)):
            if index:
                spin_for(node, args.between_delay)
            if index == 3:
                for value in np.linspace(args.scrape_grip, 1.0, max(2, round(args.gripper_close_seconds / 0.05)))[1:]:
                    publish_gripper(node, RIGHT_GRIPPER_CMD_TOPIC, value)  # gentle close, then the move away
                    spin_for(node, 0.05)
            run_dual_goals(node, clients, [make_goal(c.names, q, args.phase_seconds) for c, q in zip(chains, joints)])
            print(f"{name} reached")
            if index == 2:
                publish_gripper(node, RIGHT_GRIPPER_CMD_TOPIC, args.scrape_grip)
        print("pickup complete: right gripper closed")
    finally:
        switch_controllers(node, [FORWARD], [LEFT_TRAJ, RIGHT_TRAJ])
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
