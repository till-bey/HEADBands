#!/usr/bin/env python3
"""One window for the whole robot: arms, head, grippers, lift and base.

  ros2 run panther_examples master_gui        # same on the robot and in the sim (docker/run.sh [--sim])

It drives the robot's own controllers, nothing to install on the robot:
- Arms: left_ and right_arm_joint_trajectory_position_controller, switched in for dual_arm_forward_position_controller
  (as cartesian_follow does). Head: head_forward_position_controller, fed interpolated positions at 50 Hz because it
  jumps to each command. Grippers: /left_gripper and /right_gripper/gripper_command. Closing the window switches the
  controllers back as they were.
- Sliders are sent on release, timed so no joint goes over MAX_JOINT_VEL (grippers: a stroke per second). The right
  gripper closes no further than RIGHT_GRIPPER_MAX (0.67), also when a file asks for more.
- Lift: /lifting_controller/commands on release (the lift has no ros2_control command interface).
- Base: hold a button to publish /cmd_vel at 10 Hz; releasing it, or leaving the window, stops the base.
- Record: samples the 20 joints and the lift at 10 Hz from the robot's state (however they move) into
  teach_out/gui_<date>_<time>.csv, rows t,joint_la0..joint_rf,body_joint under a "# t,<joint names>" header.
- Play: any CSV with that header and any subset of those joints (e.g. motion/retarget/joints_*.csv). The lift goes
  to the first row's height first and stays there; then the joints go to the first row at MAX_JOINT_VEL and follow
  the file at its own timing. Joints not in the file hold. Asks first if the file is faster than 1.25 x MAX_JOINT_VEL.
- Record by hand: pick left, right or both arms. That arm's trajectory controller is switched off, then its torque
  drops: recover, 1 s, then disable, one joint at a time and retried up to 3 times (plain disable answers False
  on the robot, and back-to-back recover/disable left joints stiff). Hold it, move it, it is
  recorded as above. Pressing again restarts the arm's hardware component (unix_openarm_left/right), the only thing
  that turns a disabled joint back on (enable and recover answer False), then switches the arm's trajectory controller
  back in, holding the arm where you left it.
  Closing the window while an arm is free locks it first.
- Stop: holds every joint where it is, stops the lift and the base.
"""
import argparse
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions
from builtin_interfaces.msg import Duration
from controller_manager_msgs.srv import ListControllers, SetHardwareComponentState, SwitchController
from geometry_msgs.msg import Twist
from lifecycle_msgs.msg import State
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from unix_msgs.msg import GripperCommand

from panther_examples.robot import (HEAD, HEAD_FORWARD, LEFT_ARM, LEFT_HARDWARE, RIGHT_HARDWARE, LEFT_GRIPPER_CMD_TOPIC, LEFT_GRIPPER_CONTROLLER,
                                    LEFT_GRIPPER_STATE_TOPIC, LEFT_TRAJ, LIFT_CMD_TOPIC, LIFT_STATE_TOPIC, LIFT_VEL,
                                    MAX_JOINT_VEL, MAX_LIFT, MIN_LIFT, RIGHT_ARM, RIGHT_GRIPPER_CMD_TOPIC,
                                    RIGHT_GRIPPER_CONTROLLER, RIGHT_GRIPPER_MAX, RIGHT_GRIPPER_STATE_TOPIC, RIGHT_TRAJ, can_service,
                                    single_position)
from panther_examples.teach import make_gripper_msg, out_dir

ARM_TRAJ = {LEFT_TRAJ: LEFT_ARM, RIGHT_TRAJ: RIGHT_ARM}
ARM_HARDWARE = {LEFT_TRAJ: LEFT_HARDWARE, RIGHT_TRAJ: RIGHT_HARDWARE}
OURS = [LEFT_TRAJ, RIGHT_TRAJ, HEAD_FORWARD, LEFT_GRIPPER_CONTROLLER, RIGHT_GRIPPER_CONTROLLER]  # active while the GUI runs
GRIPPERS = ["joint_lf", "joint_rf"]
GRIPPER_CMD = {"joint_lf": LEFT_GRIPPER_CMD_TOPIC, "joint_rf": RIGHT_GRIPPER_CMD_TOPIC}
JOINTS = LEFT_ARM + RIGHT_ARM + HEAD + GRIPPERS
COLUMNS = JOINTS + ["body_joint"]  # recorded and playable
FAST = 1.25 * MAX_JOINT_VEL  # play asks first above this; a 10 Hz recording of a slider move reads up to ~0.55 rad/s
LIMIT_SLACK = 0.05  # a file this far past a limit is refused; closer, it is clamped (recorded states sit on the limits)
HAND = {"right": [RIGHT_TRAJ], "left": [LEFT_TRAJ], "both": [LEFT_TRAJ, RIGHT_TRAJ]}  # arms freed by "Record by hand"
GROUPS = {"Left arm": LEFT_ARM, "Right arm": RIGHT_ARM, "Head (pitch, yaw)": HEAD, "Grippers (0 open, 1 closed)": GRIPPERS}
# shortcut: gripper speed is sent as strokes/s, as the sim reads it; the robot's units are undocumented. Check on the
# robot before closing a gripper on anything.
GRIPPER_VEL = 1.0  # strokes/s
# shortcut: base speeds are our choice, not vendor limits; raise them once tried on the robot.
BASE_LIN, BASE_ANG = 0.2, 0.4  # m/s, rad/s
BASE_BUTTONS = {"⟲": (0, 0, 1), "↑": (1, 0, 0), "⟳": (0, 0, -1),
                "←": (0, 1, 0), "↓": (-1, 0, 0), "→": (0, -1, 0)}  # (x, y, yaw) direction, 3 per row


def move_time(now, targets):
    """Seconds for one trajectory point from `now` to `targets` so no joint goes over its speed limit."""
    return max([0.3] + [abs(q - now.get(n, q)) / (GRIPPER_VEL if n in GRIPPERS else MAX_JOINT_VEL)
                        for n, q in targets.items()])


def duration(t):
    return Duration(sec=int(t), nanosec=int(t % 1 * 1e9))


def write_csv(path, names, rows):
    with open(path, "w") as f:
        f.write(f"# t,{','.join(names)}  (s, rad, grippers 1 closed / 0 open, body_joint m) recorded by master_gui\n")
        f.writelines(",".join(f"{v:.6f}" for v in r) + "\n" for r in rows)


def read_csv(path):
    """(joint names, rows [t, q...]) from a file whose first "# t,..." comment names the columns."""
    names, rows = None, []
    for line in open(path):
        line = line.strip()
        if line.startswith("#"):
            head = line[1:].split()[0].split(",") if line[1:].split() else []
            if names is None and head[:1] == ["t"]:
                names = head[1:]
        elif line:
            rows.append([float(v) for v in line.split(",")])
    if not names:
        raise ValueError("no '# t,<joint names>' header line")
    return names, rows


def check(names, rows, limits):
    """Raise ValueError if the file cannot be played."""
    unknown = set(names) - set(COLUMNS)
    if unknown or len(set(names)) != len(names):
        raise ValueError(f"unknown or repeated columns {sorted(unknown) or names}")
    if not rows or any(len(r) != len(names) + 1 for r in rows):
        raise ValueError(f"needs rows of {len(names) + 1} values")
    if any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise ValueError("time must strictly increase")
    for i, n in enumerate(names, 1):
        if n in GRIPPERS:
            continue  # clamped instead: files close the grippers to 1.0, the right one stops at RIGHT_GRIPPER_MAX
        lo, hi = limits[n]
        bad = [r[i] for r in rows if not lo - LIMIT_SLACK <= r[i] <= hi + LIMIT_SLACK]
        if bad:
            raise ValueError(f"{n} reaches {bad[0]:.3f}, limits {lo:.3f}..{hi:.3f}")


def peak_speed(names, rows):
    """(rad/s, joint) of the fastest arm or head joint between rows."""
    return max(((abs(b[i] - a[i]) / (b[0] - a[0]), n) for i, n in enumerate(names, 1) if n in LEFT_ARM + RIGHT_ARM + HEAD
                for a, b in zip(rows, rows[1:])), default=(0.0, None))


def play_frames(names, rows, now, limits):
    """(times, frames) for a file: frame 0 is `now` at t=0, then a move to the first row at MAX_JOINT_VEL, then the
    file's timing. Joints not in the file hold their `now` position; positions are clamped to the limits."""
    idx = {n: i for i, n in enumerate(names, 1)}
    frames = [{n: min(max(r[idx[n]] if n in idx else now[n], limits[n][0]), limits[n][1]) for n in JOINTS} for r in rows]
    shift = move_time(now, frames[0]) - rows[0][0]
    return [0.0] + [r[0] + shift for r in rows], [{n: now[n] for n in JOINTS}] + frames


def plan(times, frames):
    """Split a move onto the robot's controllers. times[0] is 0 and frames[0] is where every joint is now.
    Returns ({arm controller: JointTrajectory}, head keyframes [(t, [pitch, yaw])], {gripper: [(t, target, speed)]}),
    t in seconds from now. A gripper command is sent at the start of each segment it moves in."""
    arms = {c: JointTrajectory(joint_names=names, points=[
        JointTrajectoryPoint(positions=[f[n] for n in names], time_from_start=duration(t))
        for t, f in zip(times[1:], frames[1:])]) for c, names in ARM_TRAJ.items()}
    head = [(t, [f[n] for n in HEAD]) for t, f in zip(times, frames)]
    grip = {g: [(t0, b[g], abs(b[g] - a[g]) / (t1 - t0)) for t0, t1, a, b in zip(times, times[1:], frames, frames[1:])
                if abs(b[g] - a[g]) > 0.005] for g in GRIPPERS}
    return arms, head, grip


def holders(states, joints, ours):
    """Active controllers, other than `ours`, that hold a command interface of `joints`."""
    return [c.name for c in states.values() if c.state == "active" and c.name not in ours
            and any(i.split("/")[0] in joints for i in c.claimed_interfaces)]


def call(node, srv, name, req, timeout=10.0):
    cli = node.create_client(srv, name)
    if not cli.wait_for_service(timeout_sec=5.0):
        raise RuntimeError(f"{name} not found")
    f = cli.call_async(req)
    rclpy.spin_until_future_complete(node, f, timeout_sec=timeout)
    if f.result() is None:
        raise RuntimeError(f"{name} did not answer")
    return f.result()


def controllers(node):
    return {c.name: c for c in call(node, ListControllers, "/controller_manager/list_controllers",
                                    ListControllers.Request()).controller}


def trigger_all(node, services, timeout=20.0):
    """Call std_srvs/Trigger services one by one, as teach does; return those that failed or did not answer."""
    failed = []
    for name in services:
        cli = node.create_client(Trigger, name)
        ok = False
        if cli.wait_for_service(timeout_sec=5.0):
            f = cli.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(node, f, timeout_sec=timeout)  # teach allows 20 s per CAN call
            ok = f.done() and f.result().success
        node.destroy_client(cli)
        if not ok:
            failed.append(name)
    return failed


def free_arms(node, ctrls, rounds=3):
    """Torque off on the arms' joints: recover, 1 s, disable, repeated for the joints that refuse. On the robot a plain
    disable answers False, and recover then disable back to back left all but one joint stiff (2026-11-15); with ~1 s
    between them it works. Returns the disable services still refusing."""
    joints = [j for c in ctrls for j in ARM_TRAJ[c]]
    for _ in range(rounds):
        trigger_all(node, [can_service(j, "recover") for j in joints])
        end = time.monotonic() + 1.0  # shortcut: 1 s is the gap the manual test had; shorten once the drive state is read
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
        failed = trigger_all(node, [can_service(j, "disable") for j in joints])
        joints = [f.split("/")[2] for f in failed]
        if not joints:
            break
    return failed


def set_hardware(node, name, label, state_id):
    req = SetHardwareComponentState.Request(name=name, target_state=State(id=state_id, label=label))
    # shortcut: 30 s is a guess for a CANopen arm to come back up; time it on the robot
    if not call(node, SetHardwareComponentState, "/controller_manager/set_hardware_component_state", req, 30.0).ok:
        raise RuntimeError(f"{name} would not go {label}")


def lock_arms(node, ctrls):
    """Torque back on for arms freed by hand. On the robot neither enable nor recover brings back a disabled joint;
    restarting the arm's hardware component does (tested 2026-11-15). Then the arm's trajectory controller comes in,
    holding where the arm is. Raises if a step fails: the arm may still be limp."""
    for c in ctrls:
        set_hardware(node, ARM_HARDWARE[c], "inactive", State.PRIMARY_STATE_INACTIVE)
        set_hardware(node, ARM_HARDWARE[c], "active", State.PRIMARY_STATE_ACTIVE)
    switch(node, ctrls, [])


def switch(node, activate, deactivate):
    if not (activate or deactivate):
        return
    req = SwitchController.Request(activate_controllers=activate, deactivate_controllers=deactivate,
                                   strictness=SwitchController.Request.STRICT, activate_asap=True)
    req.timeout.sec = 5
    if not call(node, SwitchController, "/controller_manager/switch_controller", req).ok:
        raise RuntimeError(f"switch_controller failed (activate {activate}, deactivate {deactivate})")


class App:
    def __init__(self, node, root, limits, now):
        self.node, self.root, self.now, self.base = node, root, now, None
        self.limits = {**limits, "body_joint": (MIN_LIFT, MAX_LIFT)}
        self.recording, self.rec_t0, self.waiting, self.hand = None, 0.0, None, None
        self.head_keys, self.grip_cmds = None, []  # pending head interpolation and gripper commands (monotonic time)
        self.arm_pub = {c: node.create_publisher(JointTrajectory, f"/{c}/joint_trajectory", 10) for c in ARM_TRAJ}
        self.head_pub = node.create_publisher(Float64MultiArray, f"/{HEAD_FORWARD}/commands", 10)
        self.grip_pub = {g: node.create_publisher(GripperCommand, t, 10) for g, t in GRIPPER_CMD.items()}
        self.lift_pub = node.create_publisher(JointState, LIFT_CMD_TOPIC, 10)
        self.base_pub = node.create_publisher(Twist, "/cmd_vel", 10)
        root.title("Panther master")
        self.vars = {}
        for col, (title, names) in enumerate(GROUPS.items()):
            frame = tk.LabelFrame(root, text=title)
            frame.grid(row=0, column=col, sticky="n", padx=4, pady=4)
            for n in names:
                self.vars[n] = self.slider(frame, n, *limits[n], self.send_joints)
        frame = tk.LabelFrame(root, text="Lift (m, 0 = bottom)")
        frame.grid(row=1, column=0, sticky="nw", padx=4, pady=4)
        self.lift = self.slider(frame, "body_joint", MIN_LIFT, MAX_LIFT, self.send_lift)
        frame = tk.LabelFrame(root, text="Base (hold)")
        frame.grid(row=1, column=1, sticky="nw", padx=4, pady=4)
        for i, (label, d) in enumerate(BASE_BUTTONS.items()):
            b = tk.Button(frame, text=label, width=4, font=("TkDefaultFont", 16))
            b.grid(row=i // 3, column=i % 3)
            b.bind("<ButtonPress-1>", lambda _e, d=d: self.drive(d))
            b.bind("<ButtonRelease-1>", lambda _e: self.drive(None))
        root.bind("<FocusOut>", lambda _e: self.drive(None))
        frame = tk.LabelFrame(root, text="Record / play")
        frame.grid(row=1, column=2, columnspan=2, sticky="nw", padx=4, pady=4)
        self.rec_button = tk.Button(frame, text="● Record", width=16, command=self.toggle_record)
        self.rec_button.grid(row=0, column=0)
        tk.Button(frame, text="▶ Play file…", width=16, command=self.play).grid(row=0, column=1)
        tk.Button(frame, text="■ Stop all", width=16, command=self.stop, fg="red").grid(row=1, column=0, columnspan=2)
        self.status = tk.Label(frame, text="", wraplength=380, justify="left")
        self.hand_arm = tk.StringVar(value="right")
        tk.OptionMenu(frame, self.hand_arm, *HAND).grid(row=2, column=0, sticky="ew")
        self.hand_button = tk.Button(frame, text="✋ Record by hand", width=16, command=self.toggle_hand)
        self.hand_button.grid(row=2, column=1)
        self.status.grid(row=3, column=0, columnspan=2, sticky="w")
        self.root.after(20, self.spin)
        self.root.after(20, self.motion_tick)
        self.root.after(100, self.base_tick)

    def slider(self, parent, name, lo, hi, on_release):
        var = tk.DoubleVar(value=min(max(self.now[name], lo), hi))
        s = tk.Scale(parent, label=name, from_=lo, to=hi, resolution=0.01, variable=var, orient="horizontal", length=220)
        s.pack()
        s.bind("<ButtonRelease-1>", lambda _e: on_release())
        s.bind("<KeyRelease>", lambda _e: on_release())  # arrow keys on a focused slider
        return var

    def send_joints(self):
        targets = {n: v.get() for n, v in self.vars.items()}
        self.execute([0.0, move_time(self.now, targets)], [{n: self.now[n] for n in JOINTS}, targets])

    def execute(self, times, frames):
        arms, head, grip = plan(times, frames)
        for c, traj in arms.items():
            self.arm_pub[c].publish(traj)
        t0 = time.monotonic()
        self.head_keys = [(t0 + t, q) for t, q in head]
        self.grip_cmds = sorted((t0 + t, g, q, v) for g, cmds in grip.items() for t, q, v in cmds)

    def motion_tick(self):
        now = time.monotonic()
        if self.head_keys:
            ts = [k[0] for k in self.head_keys]
            self.head_pub.publish(Float64MultiArray(data=[float(np.interp(now, ts, [k[1][i] for k in self.head_keys]))
                                                          for i in range(len(HEAD))]))
            if now >= ts[-1]:
                self.head_keys = None
        while self.grip_cmds and self.grip_cmds[0][0] <= now:
            _, g, q, v = self.grip_cmds.pop(0)
            self.grip_pub[g].publish(make_gripper_msg(q, velocity=min(max(v, 0.05), GRIPPER_VEL)))
        self.root.after(20, self.motion_tick)

    def send_lift(self):
        self.lift_pub.publish(JointState(name=[""], position=[self.lift.get()], velocity=[LIFT_VEL]))

    def say(self, text):
        self.status.config(text=text)

    def toggle_record(self):
        if self.recording is None:
            self.recording, self.rec_t0 = [], time.monotonic()
            self.rec_button.config(text="■ Stop recording")
            self.say("recording: move the arms and head with the sliders")
            self.record_tick()
            return
        rows, self.recording = self.recording, None
        self.rec_button.config(text="● Record")
        path = out_dir() / time.strftime("gui_%Y%m%d_%H%M%S.csv")
        write_csv(path, COLUMNS, rows)
        self.say(f"{len(rows)} samples ({rows[-1][0]:.1f} s) saved to {path}" if rows else "nothing recorded")

    def toggle_hand(self):
        if self.hand is None and self.recording is not None:
            return self.say("stop the slider recording first")
        if self.hand is not None:
            if self.recording is not None:  # not on a retry after a failed lock
                self.toggle_record()
            saved = self.status.cget("text")
            self.lock()
            return self.say(f"{saved}\n{self.status.cget('text')}")
        arm, ctrls = self.hand_arm.get(), HAND[self.hand_arm.get()]
        if not messagebox.askokcancel("Torque off", f"The {arm} arm{'s' if arm == 'both' else ''} will go limp now. "
                                      "Hold it, then OK."):
            return
        self.stop()
        try:
            switch(self.node, [], ctrls)  # first, so nothing pulls the arm back while it is moved
        except RuntimeError as e:
            return self.say(f"not started: {e}")
        self.hand = ctrls
        self.say("dropping torque, one joint at a time…")
        self.root.update_idletasks()
        failed = free_arms(self.node, ctrls)
        self.hand_button.config(text="🔒 Stop and lock")
        self.toggle_record()
        note = (f"\nno torque drop confirmed for: {' '.join(sorted({f.split('/')[2] for f in failed}))}. If the arm is "
                "still stiff, press Stop and lock.") if failed else ""
        self.say(f"recording: move the {arm} arm{'s' if arm == 'both' else ''} by hand, then Stop and lock{note}")

    def lock(self):
        self.say("restarting the arm hardware to turn torque back on…")
        self.root.update_idletasks()
        try:
            lock_arms(self.node, self.hand)
        except RuntimeError as e:
            return self.say(f"NOT LOCKED, the arm may still be limp: hold it. {e}. Press Stop and lock to retry.")
        self.hand = None
        self.hand_button.config(text="✋ Record by hand")
        for n, v in self.vars.items():
            v.set(self.now[n])
        self.say("arms locked where they are")

    def record_tick(self):
        if self.recording is None:
            return
        self.recording.append([time.monotonic() - self.rec_t0] + [self.now[n] for n in COLUMNS])
        self.root.after(100, self.record_tick)

    def play(self, path=None):
        path = path or filedialog.askopenfilename(initialdir=out_dir(), filetypes=[("CSV", "*.csv"), ("all", "*")])
        if not path:
            return
        try:
            names, rows = read_csv(path)
            check(names, rows, self.limits)
        except (OSError, ValueError) as e:
            self.say(f"not played: {e}")
            return
        speed, joint = peak_speed(names, rows)
        if speed > FAST and not messagebox.askyesno(
                "Fast file", f"{joint} reaches {speed:.2f} rad/s (sliders stop at {MAX_JOINT_VEL}). Play anyway?"):
            return
        if "body_joint" not in names:
            return self.start_play(path, names, rows)
        lift = min(max(rows[0][1 + names.index("body_joint")], MIN_LIFT), MAX_LIFT)
        self.lift.set(lift)
        self.send_lift()
        self.waiting = (path, names, rows, lift, time.monotonic() + abs(lift - self.now["body_joint"]) / LIFT_VEL + 5.0)
        self.say(f"lift going to {lift:.3f} m first")
        self.wait_lift()

    def wait_lift(self):
        if self.waiting is None:  # Stop pressed
            return
        path, names, rows, lift, deadline = self.waiting
        if abs(self.now["body_joint"] - lift) < 0.01:
            self.waiting = None
            return self.start_play(path, names, rows)
        if time.monotonic() > deadline:
            self.waiting = None
            return self.say(f"not played: lift at {self.now['body_joint']:.3f} m, did not reach {lift:.3f} m")
        self.root.after(100, self.wait_lift)

    def start_play(self, path, names, rows):
        times, frames = play_frames(names, rows, self.now, self.limits)
        self.execute(times, frames)
        for n, q in frames[-1].items():  # so the next slider release starts from the end pose
            self.vars[n].set(q)
        self.say(f"playing {path} ({times[-1]:.1f} s)")

    def stop(self):
        self.waiting = None
        # Humble's trajectory controller ignores an empty trajectory, so: reach where it is now, in 0.3 s
        self.execute([0.0, 0.3], [{n: self.now[n] for n in JOINTS}] * 2)
        self.head_pub.publish(Float64MultiArray(data=[self.now[n] for n in HEAD]))
        for g in GRIPPERS:
            self.grip_pub[g].publish(make_gripper_msg(self.now[g], velocity=GRIPPER_VEL))
        self.lift_pub.publish(JointState(name=[""], position=[self.now["body_joint"]], velocity=[LIFT_VEL]))
        self.drive(None)
        for n, v in self.vars.items():
            v.set(self.now[n])
        self.lift.set(self.now["body_joint"])
        self.say("stopped")

    def drive(self, direction):
        was_driving, self.base = self.base, direction
        if direction is None and was_driving:
            self.base_pub.publish(Twist())

    def base_tick(self):
        if self.base:
            msg = Twist()
            msg.linear.x, msg.linear.y, msg.angular.z = (BASE_LIN * self.base[0], BASE_LIN * self.base[1],
                                                         BASE_ANG * self.base[2])
            self.base_pub.publish(msg)
        self.root.after(100, self.base_tick)

    def spin(self):
        for _ in range(50):  # /joint_states alone comes at 100 Hz
            rclpy.spin_once(self.node, timeout_sec=0)
        self.root.after(20, self.spin)


def store_single(now, joint, msg):
    q = single_position(msg)
    if q is not None:
        now[joint] = q


def start(node):
    """Read the robot, take the upper body. Returns (limits, now, activated, deactivated)."""
    got, now = {}, {}
    node.create_subscription(String, "/robot_description", lambda m: got.setdefault("urdf", m.data),
                             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    node.create_subscription(JointState, "/joint_states", lambda m: now.update(zip(m.name, m.position)), 10)
    # the robot's /joint_states has no lift or gripper; they have their own state topics
    for topic, joint in ((LIFT_STATE_TOPIC, "body_joint"), (LEFT_GRIPPER_STATE_TOPIC, "joint_lf"),
                         (RIGHT_GRIPPER_STATE_TOPIC, "joint_rf")):
        node.create_subscription(JointState, topic, lambda m, j=joint: store_single(now, j, m), 10)
    for _ in range(50):
        missing = [n for n in JOINTS + ["body_joint"] if n not in now] + ([] if "urdf" in got else ["/robot_description"])
        if not missing:
            break
        rclpy.spin_once(node, timeout_sec=0.1)
    else:
        sys.exit(f"no state for {missing}: wrong ROS_DOMAIN_ID or RMW?")
    urdf = {j.get("name"): j.find("limit") for j in ET.fromstring(got["urdf"]).findall("joint")}
    limits = {n: (float(urdf[n].get("lower")), float(urdf[n].get("upper"))) for n in LEFT_ARM + RIGHT_ARM + HEAD}
    limits.update({"joint_lf": (0.0, 1.0), "joint_rf": (0.0, RIGHT_GRIPPER_MAX)})

    states = controllers(node)
    missing = [c for c in OURS if c not in states]
    if missing:
        raise RuntimeError(f"{missing} not loaded")
    activate = [c for c in OURS if states[c].state != "active"]
    deactivate = holders(states, JOINTS, OURS)  # dual_arm_forward_position_controller, usually
    switch(node, activate, deactivate)
    return limits, now, activate, deactivate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    # keep the context alive on Ctrl-C so the controllers can be switched back
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("master_gui")
    try:
        limits, now, activate, deactivate = start(node)
    except RuntimeError as e:
        sys.exit(f"nothing moved: {e}")
    root = tk.Tk()
    app = App(node, root, limits, now)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        app.drive(None)
        if app.hand:  # torque back on before anything else; the window is gone, so not app.lock()
            try:
                lock_arms(node, app.hand)
            except RuntimeError as e:
                print(f"ARM STILL LIMP, hold it: {e}. Its torque is off; turn it on with recover once a controller holds it.")
        switch(node, deactivate, activate)  # put the controllers back as they were
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
