"""Record a loose-arm motion, then turn it into a trajectory.

Run one step at a time. With no arguments this only prints status.

  python3 src/panther_examples/panther_examples/teach.py
  python3 src/panther_examples/panther_examples/teach.py release
  python3 src/panther_examples/panther_examples/teach.py one-joint
  python3 src/panther_examples/panther_examples/teach.py disable --yes
  python3 src/panther_examples/panther_examples/teach.py record --seconds 10
  python3 src/panther_examples/panther_examples/teach.py action
  python3 src/panther_examples/panther_examples/teach.py play --yes

disable drops torque on every arm joint. play moves the arms.
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

LEFT = [
    "joint_la0", "joint_la1", "joint_la2", "joint_la3",
    "joint_la4", "joint_la5", "joint_la6", "joint_la7",
]
RIGHT = [
    "joint_ra0", "joint_ra1", "joint_ra2", "joint_ra3",
    "joint_ra4", "joint_ra5", "joint_ra6", "joint_ra7",
]
ARMS = LEFT + RIGHT
FORWARD = "dual_arm_forward_position_controller"
LEFT_TRAJ = "left_arm_joint_trajectory_position_controller"
RIGHT_TRAJ = "right_arm_joint_trajectory_position_controller"
LEFT_HARDWARE = "unix_openarm_left"
RIGHT_HARDWARE = "unix_openarm_right"
# The CANopen driver waits this long for the status word, then answers.
DRIVE_SWITCH_SEC = 5.0
SERVICE_TIMEOUT_SEC = 20.0


def out_dir():
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
    bus = "can0" if joint.startswith("joint_la") else "can1"
    return f"/{bus}/{joint}/{verb}"


def hardware_for(joint):
    return LEFT_HARDWARE if joint.startswith("joint_la") else RIGHT_HARDWARE


def hardware_state(name):
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
    """Call a Trigger service and wait past the drive's own 5s state switch."""
    print(
        f"CALL     {service}  "
        f"(drive waits {DRIVE_SWITCH_SEC:.0f}s, this waits {SERVICE_TIMEOUT_SEC:.0f}s)"
    )
    try:
        proc = subprocess.run(
            ["ros2", "service", "call", service, "std_srvs/srv/Trigger", "{}"],
            check=False, text=True, capture_output=True, timeout=SERVICE_TIMEOUT_SEC)
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
    """Humble prints `name  type  state`. Older builds print `name[type] state`."""
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


class Teach(Node):
    def __init__(self):
        super().__init__("panther_teach")

    def record_samples(self, seconds):
        latest = {}

        def on_state(msg):
            latest["msg"] = msg

        self.create_subscription(JointState, "/joint_states", on_state, 10)
        samples = []
        start = time.monotonic()
        next_save = start
        missing = set(ARMS)
        while rclpy.ok() and time.monotonic() < start + seconds:
            rclpy.spin_once(self, timeout_sec=0.05)
            msg = latest.get("msg")
            if msg is None or time.monotonic() < next_save:
                continue
            found = {
                name: position
                for name, position in zip(msg.name, msg.position)
                if name in ARMS
            }
            missing -= set(found)
            if len(found) != len(ARMS):
                continue
            samples.append({
                "t": round(time.monotonic() - start, 3),
                "position": [found[name] for name in ARMS],
            })
            next_save = time.monotonic() + 0.05
        return samples, sorted(missing)


def step_status():
    print("1. CAN enable/disable services")
    for joint in (LEFT[0], RIGHT[0]):
        for verb in ("disable", "enable"):
            print(f"   {service_name(joint, verb)}")
    print("   (one-joint calls these. This step does not.)")
    print()
    print("2. Arm controllers")
    code, stdout, stderr, states = controller_states()
    if code != 0:
        print(stderr.strip() or "ros2 control list_controllers failed")
        return 1
    for name in (FORWARD, LEFT_TRAJ, RIGHT_TRAJ):
        print(f"   {name}: {states.get(name, 'not listed')}")
    if states.get(FORWARD) == "active":
        print("   The forward controller is holding the arms. Run release first.")
    for name in (LEFT_HARDWARE, RIGHT_HARDWARE):
        print(f"   {name}: {hardware_state(name)}")
    print("   Leave the arm hardware active. Turning it off quick-stops every joint,")
    print("   and disable cannot run after that quick stop.")
    print()
    print("Next: release, then one-joint on joint_la7.")
    if stdout.strip():
        print()
        print(stdout.rstrip())
    return 0


def step_release():
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


def _require_active(hardware):
    state = hardware_state(hardware)
    print(f"{hardware}: {state}")
    if state == "active":
        return True
    print(f"disable was not called. {hardware} is {state}.")
    print("Turning that component off quick-stops every joint on the arm.")
    print("From quick stop, disable has no legal transition and returns success=False.")
    print("Turning it back on runs init and homing on all 8 joints. Stay clear of the arm, then:")
    print(f"  ros2 control set_hardware_component_state {hardware} active")
    return False


def step_one_joint(joint):
    if joint not in ARMS:
        print(f"{joint} is not an arm joint")
        return 1
    hardware = hardware_for(joint)
    print(f"Hold {joint}. Disabling drops its torque.")
    if not _require_active(hardware):
        return 1
    disabled = call_trigger(service_name(joint, "disable"))
    if not disabled:
        print("Enable was not called.")
        print("The drive refused the transition. It may already be in quick stop.")
        return 1
    time.sleep(2.0)
    enabled = call_trigger(service_name(joint, "enable"))
    if not enabled:
        print(f"{joint} is still disabled.")
        return 1
    print(f"{joint} accepted disable and enable.")
    return 0


def step_disable():
    print("Disabling every arm joint. Both arms will drop. Hold them.")
    if not _require_active(LEFT_HARDWARE) or not _require_active(RIGHT_HARDWARE):
        return 1
    failed = []
    for joint in ARMS:
        if not call_trigger(service_name(joint, "disable")):
            failed.append(joint)
    if failed:
        print("Still enabled or unanswered: " + " ".join(failed))
        return 1
    print("All 16 arm joints accepted disable. Move them by hand, then record.")
    return 0


def step_record(node, seconds):
    print(f"Recording /joint_states for {seconds:.0f} s. Move the arms now.")
    samples, missing = node.record_samples(seconds)
    if missing:
        print("No samples for: " + " ".join(missing))
    if not samples:
        print("Recorded nothing.")
        return 1
    path = out_dir() / "recording.json"
    path.write_text(json.dumps({
        "joints": ARMS,
        "samples": samples,
    }, indent=2) + "\n")
    print(f"Wrote {len(samples)} samples to {path}")
    return 0


def _downsample(samples):
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


def _side(points, start, end):
    trajectory = []
    for sample in points:
        # Half a second before the recorded start, so playback is not a step.
        trajectory.append({
            "time_from_start": round(0.5 + sample["t"], 3),
            "positions": sample["position"][start:end],
        })
    return trajectory


def step_action():
    path = out_dir() / "recording.json"
    if not path.is_file():
        print(f"No recording at {path}. Run record first.")
        return 1
    recording = json.loads(path.read_text())
    samples = recording["samples"]
    if not samples:
        print("Recording has no samples.")
        return 1
    points = _downsample(samples)
    action = {
        "note": (
            "Start playback from the pose in the first sample. "
            "The first point is 0.5 s out so the arms are not stepped."
        ),
        "left": {"joint_names": LEFT, "points": _side(points, 0, 8)},
        "right": {"joint_names": RIGHT, "points": _side(points, 8, 16)},
    }
    dest = out_dir() / "action.json"
    dest.write_text(json.dumps(action, indent=2) + "\n")
    duration = action["left"]["points"][-1]["time_from_start"]
    print(f"Wrote {len(points)} points, {duration:.1f} s, to {dest}")
    print("First left positions: " + " ".join(
        f"{value:.3f}" for value in action["left"]["points"][0]["positions"]))
    print("Last left positions:  " + " ".join(
        f"{value:.3f}" for value in action["left"]["points"][-1]["positions"]))
    print("Inspect that file. play --yes sends it.")
    return 0


def _goal_from_side(side):
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(side["joint_names"])
    for point in side["points"]:
        msg = JointTrajectoryPoint()
        msg.positions = [float(value) for value in point["positions"]]
        seconds = float(point["time_from_start"])
        msg.time_from_start.sec = int(seconds)
        msg.time_from_start.nanosec = int(round((seconds % 1) * 1e9))
        if msg.time_from_start.nanosec >= 1_000_000_000:
            msg.time_from_start.sec += 1
            msg.time_from_start.nanosec -= 1_000_000_000
        goal.trajectory.points.append(msg)
    return goal


def _submit(node, client, goal, label):
    if not client.wait_for_server(timeout_sec=5.0):
        print(f"No action server for {label}")
        return None
    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    handle = future.result() if future.done() else None
    if handle is None or not handle.accepted:
        print(f"Rejected {label}")
        return None
    print(f"Accepted {label}")
    return handle.get_result_async()


def step_play(node):
    path = out_dir() / "action.json"
    if not path.is_file():
        print(f"No action at {path}. Run action first.")
        return 1
    action = json.loads(path.read_text())
    print(f"Switching off {FORWARD} and on the two trajectory controllers.")
    proc = subprocess.run([
        "ros2", "control", "switch_controllers",
        "--deactivate", FORWARD,
        "--activate", LEFT_TRAJ, RIGHT_TRAJ,
    ], check=False)
    if proc.returncode != 0:
        print("Controllers were not switched. Nothing was sent.")
        return proc.returncode
    left = ActionClient(
        node, FollowJointTrajectory, f"/{LEFT_TRAJ}/follow_joint_trajectory")
    right = ActionClient(
        node, FollowJointTrajectory, f"/{RIGHT_TRAJ}/follow_joint_trajectory")
    pending = (
        ("left arm", _submit(node, left, _goal_from_side(action["left"]), "left arm")),
        ("right arm", _submit(node, right, _goal_from_side(action["right"]), "right arm")),
    )
    if any(future is None for _label, future in pending):
        return 1
    for label, future in pending:
        rclpy.spin_until_future_complete(node, future)
        result = future.result()
        print(f"Finished {label} status={result.status}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        nargs="?",
        default="status",
        choices=("status", "release", "one-joint", "disable", "record", "action", "play"),
    )
    parser.add_argument("--joint", default="joint_la7")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="required for disable and play",
    )
    args = parser.parse_args()

    if args.step in ("disable", "play") and not args.yes:
        print(f"{args.step} does nothing without --yes")
        return 1

    if args.step == "one-joint":
        return step_one_joint(args.joint)
    if args.step == "disable":
        return step_disable()
    if args.step in ("status", "release", "action"):
        return {
            "status": step_status,
            "release": step_release,
            "action": step_action,
        }[args.step]()

    rclpy.init()
    node = Teach()
    try:
        if args.step == "record":
            return step_record(node, args.seconds)
        return step_play(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
