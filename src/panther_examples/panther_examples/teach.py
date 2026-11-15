"""Record dual-arm and gripper motion by hand, adjust lift height, and play back.

Run one step at a time inside the Docker container:

  ros2 run panther_examples teach                  # print status of controllers, hardware, and lift
  ros2 run panther_examples teach release           # deactivate arm & gripper holding controllers
  ros2 run panther_examples teach disable --yes     # drop torque on both arms and grippers
  ros2 run panther_examples teach adjust-height     # disable arms, interactively adjust base height, then re-enable
  ros2 run panther_examples teach record --seconds 10  # record both arms & grippers
  ros2 run panther_examples teach action [--height 0.35]  # convert recording -> action (with optional lift height)
  ros2 run panther_examples teach play --yes [--file <path>] # play back lift + dual arms + grippers

disable drops torque. play moves the robot.
Neither runs without --yes.
"""

import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from unix_msgs.msg import GripperCommand

LEFT_ARM = [
    "joint_la0", "joint_la1", "joint_la2", "joint_la3",
    "joint_la4", "joint_la5", "joint_la6", "joint_la7",
]
RIGHT_ARM = [
    "joint_ra0", "joint_ra1", "joint_ra2", "joint_ra3",
    "joint_ra4", "joint_ra5", "joint_ra6", "joint_ra7",
]
ARMS = LEFT_ARM + RIGHT_ARM

FORWARD = "dual_arm_forward_position_controller"
LEFT_TRAJ = "left_arm_joint_trajectory_position_controller"
RIGHT_TRAJ = "right_arm_joint_trajectory_position_controller"

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
MIN_LIFT = 0.0
MAX_LIFT = 0.8

SERVICE_TIMEOUT_SEC = 20.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def out_dir():
    """Resolve the output directory for recordings, next to the workspace root."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "panther_examples").is_dir():
            path = parent / "teach_out"
            path.mkdir(parents=True, exist_ok=True)
            return path
    path = Path("/workspace/teach_out")
    path.mkdir(parents=True, exist_ok=True)
    return path


def service_name(joint, verb):
    """Build the CANopen service name for an arm joint."""
    bus = "can0" if joint.startswith("joint_la") else "can1"
    return f"/{bus}/{joint}/{verb}"


def hardware_state(name):
    """Parse the state label for a hardware component."""
    proc = subprocess.run(
        ["ros2", "control", "list_hardware_components"],
        check=False, text=True, capture_output=True)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout)
    current = None
    for line in plain.splitlines():
        stripped = line.strip()
        if stripped.startswith("name:"):
            current = stripped.split(":", 1)[1].strip()
        elif current == name and "label=" in stripped:
            return stripped.split("label=", 1)[1].strip().strip("'\"")
    return "unknown"


def call_trigger(service):
    """Call a std_srvs/srv/Trigger service and return True on success."""
    print(f"CALL     {service}  (timeout {SERVICE_TIMEOUT_SEC:.0f}s)")
    try:
        proc = subprocess.run(
            ["ros2", "service", "call", service, "std_srvs/srv/Trigger", "{}"],
            check=False, text=True, capture_output=True,
            timeout=SERVICE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT  {service}")
        return None
    text = "\n".join(
        part.strip() for part in (proc.stdout, proc.stderr) if part and part.strip())
    if text:
        print(text)
    compact = text.replace(" ", "")
    if "success=True" in compact or "success:true" in compact.lower():
        return True
    print(f"FAIL     {service}")
    return False


def parse_controller_states(text):
    """Parse controller names and their lifecycle states."""
    states = {}
    known = {"active", "inactive", "unconfigured", "finalized"}
    plain = re.sub(r"\x1b\[[0-9;]*m", "", text)
    for line in plain.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[-1] not in known:
            continue
        states[parts[0].split("[", 1)[0]] = parts[-1]
    return states


def controller_states():
    proc = subprocess.run(
        ["ros2", "control", "list_controllers"],
        check=False, text=True, capture_output=True)
    text = proc.stdout if proc.stdout.strip() else proc.stderr
    return proc.returncode, proc.stdout, proc.stderr, parse_controller_states(text)


def make_gripper_msg(position, velocity=0.1, effort=0.5):
    """Construct unix_msgs/GripperCommand message."""
    msg = GripperCommand()
    msg.mode = "parallel"
    msg.command.name = [""]
    msg.command.position = [float(position)]
    msg.command.velocity = [float(velocity)]
    msg.command.effort = [float(effort)]
    return msg


def make_lift_msg(height):
    """Construct sensor_msgs/JointState for the lifting controller."""
    clamped = max(MIN_LIFT, min(MAX_LIFT, float(height)))
    msg = JointState()
    msg.name = ["body_joint"]
    msg.position = [clamped]
    msg.velocity = []
    msg.effort = []
    return msg


def stream_lift_command(pub, node, height, duration_sec=1.0):
    """Stream lift position command at 20 Hz to ensure continuous reception."""
    deadline = time.monotonic() + 1.0
    while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    msg = make_lift_msg(height)
    end = time.monotonic() + max(0.2, duration_sec)
    while rclpy.ok() and time.monotonic() < end:
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_status():
    """Print current controller, hardware, and lift state."""
    print("=== Panther Teach Status (Dual Arm + Gripper + Lift) ===\n")

    print("1. Controllers")
    code, stdout, stderr, states = controller_states()
    if code != 0:
        print(stderr.strip() or "ros2 control list_controllers failed")
        return 1
    for name in (FORWARD, LEFT_TRAJ, RIGHT_TRAJ, LEFT_GRIPPER_CONTROLLER, RIGHT_GRIPPER_CONTROLLER, LIFT_CONTROLLER):
        print(f"   {name}: {states.get(name, 'not listed')}")
    if states.get(FORWARD) == "active":
        print("   >>> The forward controller is holding the arms. Run 'release' first.")
    print()

    print("2. Hardware Components")
    for hw in (LEFT_HARDWARE, RIGHT_HARDWARE, LEFT_GRIPPER_HARDWARE, RIGHT_GRIPPER_HARDWARE):
        print(f"   {hw}: {hardware_state(hw)}")
    print()

    print("Pipeline:")
    print("  1) release        – deactivate arm & gripper holding controllers")
    print("  2) disable --yes  – drop torque on both arms and grippers (hold arms!)")
    print("  3) adjust-height  – disable arms, adjust base height interactively, re-enable arms")
    print("  4) record         – subscribe to dual arms & grippers for N seconds")
    print("  5) action         – bundle trajectory (optional: --height <meters>)")
    print("  6) play --yes     – execute: moves lift first (if specified), then plays trajectories")
    if stdout.strip():
        print()
        print(stdout.rstrip())
    return 0


def step_release():
    """Deactivate arm forward position controller and gripper controllers."""
    code, _stdout, stderr, states = controller_states()
    if code != 0:
        print(stderr.strip() or "ros2 control list_controllers failed")
        return 1

    to_deactivate = []
    for ctrl in (FORWARD, LEFT_GRIPPER_CONTROLLER, RIGHT_GRIPPER_CONTROLLER):
        if states.get(ctrl) == "active":
            to_deactivate.append(ctrl)

    if not to_deactivate:
        print("No holding controllers active. Nothing to release.")
        return 0

    print(f"Deactivating: {', '.join(to_deactivate)}")
    proc = subprocess.run(
        ["ros2", "control", "switch_controllers", "--deactivate"] + to_deactivate,
        check=False)
    if proc.returncode != 0:
        return proc.returncode

    _code, _stdout, _stderr, states = controller_states()
    for name in to_deactivate:
        print(f"{name}: {states.get(name, 'not listed')}")
    return 0


def step_disable():
    """Disable (drop torque on) left and right arm joints and grippers."""
    print("Disabling both arms and grippers. The arms will drop – hold them!")

    step_release()

    # Deactivate gripper hardware components
    for ghw in (LEFT_GRIPPER_HARDWARE, RIGHT_GRIPPER_HARDWARE):
        print(f"Deactivating {ghw}...")
        subprocess.run(
            ["ros2", "control", "set_hardware_component_state", ghw, "inactive"],
            check=False)

    failed = []
    for joint in ARMS:
        if not call_trigger(service_name(joint, "disable")):
            failed.append(joint)

    if failed:
        print("Note: CAN disable returned False for: " + " ".join(failed))
        print("If forward controller was released, verify if drives are already de-energized.")
    else:
        print("All 16 arm joints accepted disable.")

    print("Move arms and grippers by hand, then run: record")
    return 0


def step_enable():
    """Re-enable all CAN joints and restore the forward controller to hold arms."""
    print("Re-enabling CAN joints on left and right arms...")
    for joint in ARMS:
        call_trigger(service_name(joint, "enable"))

    for ghw in (LEFT_GRIPPER_HARDWARE, RIGHT_GRIPPER_HARDWARE):
        print(f"Re-activating {ghw}...")
        subprocess.run(
            ["ros2", "control", "set_hardware_component_state", ghw, "active"],
            check=False)

    print(f"Activating {FORWARD} to hold arms at current pose...")
    subprocess.run(
        ["ros2", "control", "switch_controllers", "--activate", FORWARD,
         LEFT_GRIPPER_CONTROLLER, RIGHT_GRIPPER_CONTROLLER],
        check=False)
    print("Arms and grippers are re-enabled and holding.")
    return 0


def step_adjust_height(node, step_size=0.05):
    """Disable arms, let user adjust base height interactively, then re-enable arms."""
    print("\n=== Base Height Interactive Adjustment ===")
    print("1. Disabling both arms so they can be positioned freely...")
    step_disable()

    current_height = [0.2]  # fallback estimate

    def on_lift_state(msg):
        if msg.position:
            current_height[0] = float(msg.position[0])

    sub = node.create_subscription(JointState, LIFT_STATE_TOPIC, on_lift_state, 10)
    lift_pub = node.create_publisher(JointState, LIFT_CMD_TOPIC, 10)

    # Wait for DDS discovery
    deadline = time.monotonic() + 1.5
    while lift_pub.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.05)

    print(f"\nBoth arms are now disabled and free to move.")
    print(f"Base lift travel range: {MIN_LIFT:.2f} m to {MAX_LIFT:.2f} m.")
    print(f"Current detected height: {current_height[0]:.3f} m\n")
    print("Controls:")
    print(f"  w / + : raise base by +{step_size:.2f} m")
    print(f"  s / - : lower base by -{step_size:.2f} m")
    print("  <number> : set exact height in meters (e.g. 0.35)")
    print("  q / enter / done : finish adjustment and re-enable arms\n")

    target_height = current_height[0]

    while True:
        try:
            line = input(f"[Height: {target_height:.3f} m] Command > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting adjustment...")
            break

        if line in ("q", "done", "exit", ""):
            print("Finishing height adjustment...")
            break
        elif line in ("w", "+", "up"):
            target_height = min(MAX_LIFT, target_height + step_size)
        elif line in ("s", "-", "down"):
            target_height = max(MIN_LIFT, target_height - step_size)
        else:
            try:
                val = float(line)
                target_height = max(MIN_LIFT, min(MAX_LIFT, val))
            except ValueError:
                print("Unrecognized command. Use w/s, a number (0.0 - 0.8), or q to finish.")
                continue

        print(f"Setting lift height to: {target_height:.3f} m")
        stream_lift_command(lift_pub, node, target_height, duration_sec=0.5)

    node.destroy_subscription(sub)
    node.destroy_publisher(lift_pub)

    # Re-enable arms
    print("\n2. Re-enabling arms to lock them at the new height...")
    step_enable()
    print(f"Base height successfully set to ~{target_height:.3f} m with arms re-enabled.\n")
    return 0


def step_record(node, seconds):
    """Subscribe to /joint_states and gripper topics, saving synchronized dual-arm data."""
    print(f"Recording dual arms and grippers for {seconds:.0f}s. Move them now.")
    left_set = set(LEFT_ARM)
    right_set = set(RIGHT_ARM)

    latest_arm = {}
    latest_left_gripper = {"pos": 0.0}
    latest_right_gripper = {"pos": 0.0}
    latest_lift = {"height": None}

    def on_arm_state(msg):
        latest_arm["msg"] = msg

    def on_left_gripper(msg):
        if msg.position:
            latest_left_gripper["pos"] = float(msg.position[0])

    def on_right_gripper(msg):
        if msg.position:
            latest_right_gripper["pos"] = float(msg.position[0])

    def on_lift_state(msg):
        if msg.position:
            latest_lift["height"] = float(msg.position[0])

    node.create_subscription(JointState, "/joint_states", on_arm_state, 10)
    node.create_subscription(JointState, LEFT_GRIPPER_STATE_TOPIC, on_left_gripper, 10)
    node.create_subscription(JointState, RIGHT_GRIPPER_STATE_TOPIC, on_right_gripper, 10)
    node.create_subscription(JointState, LIFT_STATE_TOPIC, on_lift_state, 10)

    samples = []
    start = time.monotonic()
    next_save = start

    while rclpy.ok() and time.monotonic() < start + seconds:
        rclpy.spin_once(node, timeout_sec=0.05)
        msg = latest_arm.get("msg")
        if msg is None or time.monotonic() < next_save:
            continue

        found = {name: pos for name, pos in zip(msg.name, msg.position)}
        has_left = all(name in found for name in LEFT_ARM)
        has_right = all(name in found for name in RIGHT_ARM)

        if not (has_left and has_right):
            continue

        samples.append({
            "t": round(time.monotonic() - start, 3),
            "left": [found[name] for name in LEFT_ARM],
            "right": [found[name] for name in RIGHT_ARM],
            "left_gripper": round(latest_left_gripper["pos"], 4),
            "right_gripper": round(latest_right_gripper["pos"], 4),
            "lift_height": round(latest_lift["height"], 4) if latest_lift["height"] is not None else None,
        })
        next_save = time.monotonic() + 0.05  # ~20 Hz

    if not samples:
        print("Recorded nothing. Verify that /joint_states is publishing.")
        return 1

    path = out_dir() / "recording.json"
    path.write_text(json.dumps({
        "joints_left": LEFT_ARM,
        "joints_right": RIGHT_ARM,
        "recorded_lift_height": latest_lift["height"],
        "samples": samples,
    }, indent=2) + "\n")
    print(f"Wrote {len(samples)} dual-arm samples to {path}")
    return 0


def _downsample(samples):
    """Keep one sample every ~100 ms, always including first and last."""
    kept = []
    next_t = 0.0
    for sample in samples:
        if sample["t"] + 1e-6 < next_t:
            continue
        kept.append(sample)
        next_t = sample["t"] + 0.1
    if kept[-1] is not samples[-1]:
        kept.append(samples[-1])
    return kept


def step_action(target_height=None):
    """Convert raw recording into dual-arm trajectory action JSON with optional lift height."""
    path = out_dir() / "recording.json"
    if not path.is_file():
        # Fallback to recording_right.json if recording.json does not exist
        fallback = out_dir() / "recording_right.json"
        if fallback.is_file():
            print(f"Dual-arm recording not found; using right-arm fallback {fallback}")
            path = fallback
        else:
            print(f"No recording found at {path}. Run record first.")
            return 1

    recording = json.loads(path.read_text())
    samples = recording.get("samples", [])
    if not samples:
        print("Recording has no samples.")
        return 1

    points = _downsample(samples)
    is_dual = "left" in points[0] and "right" in points[0]

    # Lift height determination: explicit CLI flag takes precedence, else recorded height
    initial_lift = target_height
    if initial_lift is None:
        initial_lift = recording.get("recorded_lift_height")
    if initial_lift is not None:
        initial_lift = max(MIN_LIFT, min(MAX_LIFT, float(initial_lift)))

    if is_dual:
        left_pts = []
        right_pts = []
        for sample in points:
            t_from_start = round(0.5 + sample["t"], 3)
            left_pts.append({
                "time_from_start": t_from_start,
                "positions": sample["left"],
                "gripper": sample.get("left_gripper", 0.0),
            })
            right_pts.append({
                "time_from_start": t_from_start,
                "positions": sample["right"],
                "gripper": sample.get("right_gripper", 0.0),
            })

        action = {
            "note": "Dual-arm and gripper trajectory.",
            "initial_lift_height": initial_lift,
            "left": {
                "joint_names": LEFT_ARM,
                "points": left_pts,
            },
            "right": {
                "joint_names": RIGHT_ARM,
                "points": right_pts,
            },
        }
        dest = out_dir() / "action.json"
        dest.write_text(json.dumps(action, indent=2) + "\n")
        duration = left_pts[-1]["time_from_start"]
        print(f"Wrote {len(points)} dual-arm waypoints ({duration:.1f} s) to {dest}")
    else:
        # Single arm format
        right_pts = []
        for sample in points:
            right_pts.append({
                "time_from_start": round(0.5 + sample["t"], 3),
                "positions": sample["position"],
                "gripper": sample.get("gripper", 0.0),
            })
        action = {
            "note": "Right-arm trajectory.",
            "initial_lift_height": initial_lift,
            "joint_names": RIGHT_ARM,
            "points": right_pts,
        }
        dest = out_dir() / "action_right.json"
        dest.write_text(json.dumps(action, indent=2) + "\n")
        duration = right_pts[-1]["time_from_start"]
        print(f"Wrote {len(points)} waypoints ({duration:.1f} s) to {dest}")

    if initial_lift is not None:
        print(f"Target initial lift height: {initial_lift:.3f} m")
    print(f"Inspect {dest}, then run: play --yes")
    return 0


def _goal_from_points(joint_names, points):
    """Build a FollowJointTrajectory.Goal from a list of trajectory points."""
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(joint_names)
    for point in points:
        msg = JointTrajectoryPoint()
        msg.positions = [float(v) for v in point["positions"]]
        seconds = float(point["time_from_start"])
        msg.time_from_start.sec = int(seconds)
        msg.time_from_start.nanosec = int(round((seconds % 1) * 1e9))
        if msg.time_from_start.nanosec >= 1_000_000_000:
            msg.time_from_start.sec += 1
            msg.time_from_start.nanosec -= 1_000_000_000
        goal.trajectory.points.append(msg)
    return goal


def step_play(node, action_file=None):
    """Execute trajectory: sets base height first, then plays dual-arm trajectories."""
    if action_file is None:
        if (out_dir() / "action.json").is_file():
            action_file = out_dir() / "action.json"
        elif (out_dir() / "action_right.json").is_file():
            action_file = out_dir() / "action_right.json"
        else:
            print(f"No action file found in {out_dir()}. Run action first.")
            return 1
    else:
        action_file = Path(action_file)

    print(f"Loading action sequence from: {action_file}")
    action = json.loads(action_file.read_text())

    is_dual = "left" in action and "right" in action

    # 1. Base height positioning (if initial_lift_height is specified)
    initial_lift = action.get("initial_lift_height")
    if initial_lift is not None:
        target_h = max(MIN_LIFT, min(MAX_LIFT, float(initial_lift)))
        print(f"Moving base lift to target height: {target_h:.3f} m...")
        lift_pub = node.create_publisher(JointState, LIFT_CMD_TOPIC, 10)
        stream_lift_command(lift_pub, node, target_h, duration_sec=3.0)
        node.destroy_publisher(lift_pub)
        print("Base height positioning complete.")

    # 2. Re-enable arm joints and gripper hardware
    print("Re-enabling CAN drives...")
    active_arms = ARMS if is_dual else RIGHT_ARM
    for joint in active_arms:
        call_trigger(service_name(joint, "enable"))

    for ghw in ([LEFT_GRIPPER_HARDWARE, RIGHT_GRIPPER_HARDWARE] if is_dual else [RIGHT_GRIPPER_HARDWARE]):
        subprocess.run(
            ["ros2", "control", "set_hardware_component_state", ghw, "active"],
            check=False)

    # 3. Switch controllers: deactivate forward, activate trajectory and gripper controllers
    activate_list = [RIGHT_TRAJ, RIGHT_GRIPPER_CONTROLLER]
    if is_dual:
        activate_list.extend([LEFT_TRAJ, LEFT_GRIPPER_CONTROLLER])

    print(f"Switching off {FORWARD}, activating: {', '.join(activate_list)}")
    proc = subprocess.run([
        "ros2", "control", "switch_controllers",
        "--deactivate", FORWARD,
        "--activate", *activate_list,
    ], check=False)
    if proc.returncode != 0:
        print("Controller switch failed. Nothing was sent.")
        return proc.returncode

    # 4. Publishers and Action Clients
    right_gripper_pub = node.create_publisher(GripperCommand, RIGHT_GRIPPER_CMD_TOPIC, 10)
    left_gripper_pub = node.create_publisher(GripperCommand, LEFT_GRIPPER_CMD_TOPIC, 10) if is_dual else None

    right_client = ActionClient(node, FollowJointTrajectory, f"/{RIGHT_TRAJ}/follow_joint_trajectory")
    left_client = ActionClient(node, FollowJointTrajectory, f"/{LEFT_TRAJ}/follow_joint_trajectory") if is_dual else None

    if not right_client.wait_for_server(timeout_sec=5.0):
        print(f"No action server for {RIGHT_TRAJ}")
        return 1
    if is_dual and not left_client.wait_for_server(timeout_sec=5.0):
        print(f"No action server for {LEFT_TRAJ}")
        return 1

    # Prepare goals
    if is_dual:
        right_points = action["right"]["points"]
        left_points = action["left"]["points"]
        right_goal = _goal_from_points(action["right"]["joint_names"], right_points)
        left_goal = _goal_from_points(action["left"]["joint_names"], left_points)
        duration = max(right_points[-1]["time_from_start"], left_points[-1]["time_from_start"])
    else:
        right_points = action["points"]
        left_points = []
        right_goal = _goal_from_points(action["joint_names"], right_points)
        left_goal = None
        duration = right_points[-1]["time_from_start"]

    print(f"Sending trajectory ({duration:.1f} s)...")

    # Dispatch goals
    right_future = right_client.send_goal_async(right_goal)
    left_future = left_client.send_goal_async(left_goal) if is_dual else None

    rclpy.spin_until_future_complete(node, right_future, timeout_sec=5.0)
    right_handle = right_future.result() if right_future.done() else None
    if right_handle is None or not right_handle.accepted:
        print("Right arm goal rejected!")
        return 1

    left_handle = None
    if is_dual:
        rclpy.spin_until_future_complete(node, left_future, timeout_sec=5.0)
        left_handle = left_future.result() if left_future.done() else None
        if left_handle is None or not left_handle.accepted:
            print("Left arm goal rejected!")
            return 1

    print("Goal(s) accepted. Streaming trajectory playback...")

    # Thread to stream gripper setpoints matching the trajectory timeline
    stop_gripper = threading.Event()

    def stream_grippers():
        start_time = time.monotonic()
        r_idx = 0
        l_idx = 0
        while not stop_gripper.is_set():
            elapsed = time.monotonic() - start_time
            if r_idx < len(right_points):
                while r_idx + 1 < len(right_points) and right_points[r_idx + 1]["time_from_start"] <= elapsed:
                    r_idx += 1
                right_gripper_pub.publish(make_gripper_msg(right_points[r_idx].get("gripper", 0.0)))

            if is_dual and l_idx < len(left_points):
                while l_idx + 1 < len(left_points) and left_points[l_idx + 1]["time_from_start"] <= elapsed:
                    l_idx += 1
                left_gripper_pub.publish(make_gripper_msg(left_points[l_idx].get("gripper", 0.0)))

            if r_idx >= len(right_points) and (not is_dual or l_idx >= len(left_points)):
                break
            time.sleep(0.05)

    gripper_thread = threading.Thread(target=stream_grippers, daemon=True)
    gripper_thread.start()

    # Wait for completion
    res_right_fut = right_handle.get_result_async()
    res_left_fut = left_handle.get_result_async() if is_dual else None

    rclpy.spin_until_future_complete(node, res_right_fut)
    if is_dual:
        rclpy.spin_until_future_complete(node, res_left_fut)

    stop_gripper.set()
    gripper_thread.join(timeout=1.0)

    # Publish final gripper states
    right_gripper_pub.publish(make_gripper_msg(right_points[-1].get("gripper", 0.0)))
    if is_dual:
        left_gripper_pub.publish(make_gripper_msg(left_points[-1].get("gripper", 0.0)))

    print("Playback completed.")

    # 5. Restore initial controller state
    print(f"Switching back to {FORWARD}...")
    subprocess.run([
        "ros2", "control", "switch_controllers",
        "--activate", FORWARD,
        "--deactivate", *[c for c in activate_list if "gripper" not in c],
    ], check=False)

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "step",
        nargs="?",
        default="status",
        choices=("status", "release", "disable", "enable", "adjust-height", "record", "action", "play"),
    )
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="recording duration (default 10 s)")
    parser.add_argument("--height", "--lift-height", type=float, default=None,
                        help="target base/lift height in meters (0.0 to 0.8) for action sequence")
    parser.add_argument("--file", type=str, default=None,
                        help="path to custom action JSON file for play (e.g. teach_out/action_right.json)")
    parser.add_argument("--yes", action="store_true",
                        help="required for disable and play (safety gate)")
    args = parser.parse_args()

    if args.step in ("disable", "play") and not args.yes:
        print(f"'{args.step}' does nothing without --yes (safety gate)")
        return 1

    if args.step == "status":
        return step_status()
    if args.step == "release":
        return step_release()
    if args.step == "disable":
        return step_disable()
    if args.step == "enable":
        return step_enable()
    if args.step == "action":
        return step_action(target_height=args.height)

    # Steps requiring an active rclpy node
    rclpy.init()
    node = Node("panther_teach")
    try:
        if args.step == "adjust-height":
            return step_adjust_height(node)
        if args.step == "record":
            return step_record(node, args.seconds)
        return step_play(node, action_file=args.file)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
