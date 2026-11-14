"""Record a right-arm motion by hand, then play it back.

Run one step at a time inside the Docker container:

  ros2 run panther_examples teach               # print status
  ros2 run panther_examples teach release        # deactivate the forward controller
  ros2 run panther_examples teach disable --yes  # drop torque on right arm joints
  ros2 run panther_examples teach record --seconds 10
  ros2 run panther_examples teach action         # convert recording -> trajectory
  ros2 run panther_examples teach play --yes     # play back via the trajectory controller

disable drops torque on right arm joints. play moves the right arm.
Neither runs unless you pass --yes.
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

RIGHT_ARM = [
    "joint_ra0", "joint_ra1", "joint_ra2", "joint_ra3",
    "joint_ra4", "joint_ra5", "joint_ra6", "joint_ra7",
]
FORWARD = "dual_arm_forward_position_controller"
RIGHT_TRAJ = "right_arm_joint_trajectory_position_controller"
RIGHT_HARDWARE = "unix_openarm_right"
# The CANopen driver waits this long for the status word, then answers.
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
    """Build the CANopen service name for a right-arm joint."""
    return f"/can1/{joint}/{verb}"


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
    print(
        f"CALL     {service}  "
        f"(timeout {SERVICE_TIMEOUT_SEC:.0f}s)"
    )
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


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_status():
    """Print current controller and hardware state for the right arm."""
    print("=== Right-arm teach status ===\n")

    print("1. CAN enable/disable services (right arm)")
    for joint in (RIGHT_ARM[0], RIGHT_ARM[-1]):
        for verb in ("disable", "enable"):
            print(f"   {service_name(joint, verb)}")
    print()

    print("2. Controllers")
    code, stdout, stderr, states = controller_states()
    if code != 0:
        print(stderr.strip() or "ros2 control list_controllers failed")
        return 1
    for name in (FORWARD, RIGHT_TRAJ):
        print(f"   {name}: {states.get(name, 'not listed')}")
    if states.get(FORWARD) == "active":
        print("   >>> The forward controller is holding the arms. Run 'release' first.")
    print()

    print("3. Hardware")
    print(f"   {RIGHT_HARDWARE}: {hardware_state(RIGHT_HARDWARE)}")
    print()

    print("Pipeline:")
    print("  1) release       – deactivate forward controller")
    print("  2) disable --yes – drop torque on right arm (hold the arm!)")
    print("  3) record        – subscribe to /joint_states for N seconds")
    print("  4) action        – convert recording to a trajectory file")
    print("  5) play --yes    – play back via FollowJointTrajectory action")
    if stdout.strip():
        print()
        print(stdout.rstrip())
    return 0


def step_release():
    """Deactivate the dual-arm forward position controller so it stops holding."""
    code, _stdout, stderr, states = controller_states()
    if code != 0:
        print(stderr.strip() or "ros2 control list_controllers failed")
        return 1
    state = states.get(FORWARD, "not listed")
    print(f"{FORWARD}: {state}")
    if state != "active":
        print("Nothing to release.")
        return 0
    proc = subprocess.run(
        ["ros2", "control", "switch_controllers", "--deactivate", FORWARD],
        check=False)
    if proc.returncode != 0:
        return proc.returncode
    _code, _stdout, _stderr, states = controller_states()
    print(f"{FORWARD}: {states.get(FORWARD, 'not listed')}")
    return 0


def step_disable():
    """Disable (drop torque on) every right-arm joint via the CANopen services."""
    print("Disabling right arm joints. The arm will drop – hold it!")
    hw = hardware_state(RIGHT_HARDWARE)
    print(f"{RIGHT_HARDWARE}: {hw}")
    if hw != "active":
        print(f"Hardware is not active ({hw}). Cannot disable joints.")
        return 1

    failed = []
    for joint in RIGHT_ARM:
        if not call_trigger(service_name(joint, "disable")):
            failed.append(joint)
    if failed:
        print("Still enabled or unanswered: " + " ".join(failed))
        return 1
    print("All 8 right-arm joints accepted disable. Move the arm by hand, then record.")
    return 0


def step_record(node, seconds):
    """Subscribe to /joint_states and save right-arm positions to a JSON file."""
    print(f"Recording /joint_states for {seconds:.0f}s. Move the right arm now.")
    right_set = set(RIGHT_ARM)
    latest = {}

    def on_state(msg):
        latest["msg"] = msg

    node.create_subscription(JointState, "/joint_states", on_state, 10)
    samples = []
    start = time.monotonic()
    next_save = start

    while rclpy.ok() and time.monotonic() < start + seconds:
        rclpy.spin_once(node, timeout_sec=0.05)
        msg = latest.get("msg")
        if msg is None or time.monotonic() < next_save:
            continue
        found = {
            name: position
            for name, position in zip(msg.name, msg.position)
            if name in right_set
        }
        if len(found) != len(RIGHT_ARM):
            continue
        samples.append({
            "t": round(time.monotonic() - start, 3),
            "position": [found[name] for name in RIGHT_ARM],
        })
        next_save = time.monotonic() + 0.05  # ~20 Hz

    if not samples:
        print("Recorded nothing.")
        return 1

    path = out_dir() / "recording_right.json"
    path.write_text(json.dumps({
        "joints": RIGHT_ARM,
        "samples": samples,
    }, indent=2) + "\n")
    print(f"Wrote {len(samples)} samples to {path}")
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


def step_action():
    """Convert the raw recording into a FollowJointTrajectory-ready JSON."""
    path = out_dir() / "recording_right.json"
    if not path.is_file():
        print(f"No recording at {path}. Run record first.")
        return 1
    recording = json.loads(path.read_text())
    samples = recording["samples"]
    if not samples:
        print("Recording has no samples.")
        return 1
    points = _downsample(samples)
    trajectory_points = []
    for sample in points:
        trajectory_points.append({
            # Add 0.5 s offset so the first point is not a hard step
            "time_from_start": round(0.5 + sample["t"], 3),
            "positions": sample["position"],
        })
    action = {
        "note": (
            "Right-arm trajectory. The first point is 0.5 s out so playback "
            "is not a step from the current pose."
        ),
        "joint_names": RIGHT_ARM,
        "points": trajectory_points,
    }
    dest = out_dir() / "action_right.json"
    dest.write_text(json.dumps(action, indent=2) + "\n")
    duration = action["points"][-1]["time_from_start"]
    print(f"Wrote {len(trajectory_points)} points, {duration:.1f} s, to {dest}")
    print("First positions: " + " ".join(
        f"{v:.3f}" for v in action["points"][0]["positions"]))
    print("Last positions:  " + " ".join(
        f"{v:.3f}" for v in action["points"][-1]["positions"]))
    print("Inspect the file, then: play --yes")
    return 0


def _goal_from_action(action):
    """Build a FollowJointTrajectory.Goal from the action JSON."""
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(action["joint_names"])
    for point in action["points"]:
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


def step_play(node):
    """Activate the right-arm trajectory controller and execute the recording."""
    path = out_dir() / "action_right.json"
    if not path.is_file():
        print(f"No action at {path}. Run action first.")
        return 1
    action = json.loads(path.read_text())

    # Re-enable joints first (they were disabled for recording)
    print("Re-enabling right arm joints...")
    for joint in RIGHT_ARM:
        call_trigger(service_name(joint, "enable"))

    # Switch controllers: deactivate forward, activate right trajectory
    print(f"Switching off {FORWARD}, on {RIGHT_TRAJ}.")
    proc = subprocess.run([
        "ros2", "control", "switch_controllers",
        "--deactivate", FORWARD,
        "--activate", RIGHT_TRAJ,
    ], check=False)
    if proc.returncode != 0:
        print("Controller switch failed. Nothing was sent.")
        return proc.returncode

    # Send the trajectory via the FollowJointTrajectory action
    client = ActionClient(
        node, FollowJointTrajectory,
        f"/{RIGHT_TRAJ}/follow_joint_trajectory")
    if not client.wait_for_server(timeout_sec=5.0):
        print(f"No action server for {RIGHT_TRAJ}")
        return 1

    goal = _goal_from_action(action)
    duration = action["points"][-1]["time_from_start"]
    print(f"Sending {len(action['points'])} points, {duration:.1f} s...")

    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    handle = future.result() if future.done() else None
    if handle is None or not handle.accepted:
        print("Goal rejected!")
        return 1

    print("Goal accepted. Playing back...")
    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result()
    print(f"Finished. Status={result.status}")

    # Switch back to the forward controller
    print(f"Switching back to {FORWARD}.")
    subprocess.run([
        "ros2", "control", "switch_controllers",
        "--activate", FORWARD,
        "--deactivate", RIGHT_TRAJ,
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
        choices=("status", "release", "disable", "record", "action", "play"),
    )
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="recording duration (default 10 s)")
    parser.add_argument("--yes", action="store_true",
                        help="required for disable and play (safety gate)")
    args = parser.parse_args()

    if args.step in ("disable", "play") and not args.yes:
        print(f"'{args.step}' does nothing without --yes (safety gate)")
        return 1

    # Steps that don't need a ROS node
    if args.step == "status":
        return step_status()
    if args.step == "release":
        return step_release()
    if args.step == "disable":
        return step_disable()
    if args.step == "action":
        return step_action()

    # Steps that need rclpy
    rclpy.init()
    node = Node("panther_teach_right")
    try:
        if args.step == "record":
            return step_record(node, args.seconds)
        return step_play(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
