# panther_examples

Robot scripts. Run them in the container (`docker/run.sh`, or `docker/run.sh --sim` against the mock sim) after `colcon build && source install/setup.bash`. Joint, controller, hardware and topic names and the lift range live in [`robot.py`](panther_examples/robot.py); change them there, not in the scripts.

| Command | What it does |
| --- | --- |
| `ros2 run panther_examples joint_check [--yes]` | Moves every joint (arms, head, grippers, lift) a small step and back, checks the measured position follows. Without `--yes` it prints the plan only |
| `ros2 run panther_examples teach ...` | Record both arms and grippers moved by hand, adjust the lift, play back. Steps in the [script docstring](panther_examples/teach.py) |
| `ros2 run panther_examples cartesian_follow ...` | Cartesian path for one arm, IK with PyKDL (below) |
| `ros2 run panther_examples master_gui` | One window for both arms, head, grippers, lift and base ([below](#master-gui)) |
| `ros2 run panther_examples commands` | Scratchpad of single commands (gripper, base, head, dual arm); edit `main()` |
| `ros2 run panther_examples pickup ...` | Home, approach, descend, and gripper pickup routine |

# Simulation frisbee pickup

Start the mock simulation in one terminal:

```bash
docker/run.sh --sim
colcon build --symlink-install
source install/setup.bash
ros2 launch panther_sim mock_sim.launch.py
```

In a second `docker/run.sh --sim` terminal, first raise the body, open the
gripper, and place the arm in its home pose:

```bash
ros2 run panther_examples pickup start \
	--arm right \
	--body-height 0.40 \
	--home-joints 0 -1 0 0 0 0 0 0
```

Then run the fetch as a separate dry-run. It uses the live arm state left by
the start command and lowers the body to the fetch height:

```bash
ros2 run panther_examples pickup fetch \
	--arm right \
	--target-xy 0.65 0.15 \
	--body-height 0.0 \
	--floor-z 0.05 \
	--approach-height 0.25 \
	--vertical-seconds 2.0 \
	--approach-seconds 6.0 \
	--descend-seconds 2.0 \
	--dry-run
```

If the default home forward-kinematics position is not a good pickup start,
override it in `body_link` Cartesian coordinates:

```bash
--start-xyz 0.45 0.00 0.70
```

The legacy `--start-xyz` option is still available for fetch diagnostics, but
the normal start position is now the live state produced by `pickup start`.

To execute the tuned Cartesian CSV before the generated fetch, run:

```bash
ros2 run panther_examples pickup hardcoded \
	--arm right \
	--hardcoded-path /workspace/motion/pickup/hardcoded_pickup.csv \
	--body-height 0.0
```

The home joints are loaded from `/workspace/motion/pickup/start.txt` by
default. The full intended sequence is therefore `start`, `hardcoded`, then
`fetch`; each command reads the live joint state from the previous command.

### Current pickup sequence

The current operational sequence is:

```bash
ros2 run panther_examples pickup start \
	--arm right \
	--body-height 0.4

ros2 run panther_examples pickup start \
	--arm left \
	--body-height 0.4

ros2 run panther_examples pickup start \
	--arm left \
	--body-height 0.0

ros2 run panther_examples cartesian_follow \
	--arm right \
	--path /workspace/motion/pickup/hardcoded_pickup.csv

ros2 run panther_examples pickup fetch \
	--arm right \
	--target-xy 0.55 0.0 \
	--body-height 0.0 \
	--floor-z 0.15 \
	--approach-height 0.2
```

Adjust `--target-xy`, `--floor-z`, and `--approach-height` in the final
command for each detected object and desired hand clearance.

### Current scrape pickup sequence

```bash
ros2 run panther_examples pickup start \
	--arm right \
	--body-height 0.4

ros2 run panther_examples pickup start \
	--arm left \
	--body-height 0.0

ros2 run panther_examples pickup scrape_prepare_position \
	--body-height 0.2 \
	--key-point-dir /workspace/motion/pickup/key_points \
	--time-scale 2.0

ros2 run panther_examples pickup scrape_prepare \
	--body-height 0.2 \
	--key-point-dir /workspace/motion/pickup/key_points \
	--time-scale 2.0

ros2 run panther_examples pickup scrape_execute \
	--body-height 0.2 \
	--key-point-dir /workspace/motion/pickup/key_points \
	--time-scale 2.0

ros2 run panther_examples pickup scrape_move_away \
	--body-height 0.2 \
	--key-point-dir /workspace/motion/pickup/key_points \
	--time-scale 2.0
```

The dry-run publishes `/cartesian_follow/planned_path` for RViz and does not
move the arm or close the gripper. Remove `--dry-run` only after inspecting the
path. For a camera pixel target, replace `--target-xy` with:

```bash
--calibration /workspace/motion/pickup/models/camera_floor.json \
--target-pixel 1622 720
```

During a failed dry-run, compare these RViz `Path` topics:

- `/sim_frisbee_pickup/requested_path`: the full requested Cartesian path
- `/cartesian_follow/planned_path`: the reachable IK prefix

The terminal also prints the first failed waypoint and its requested XYZ pose.

The four phases play `key_points/1.txt` through `key_points/4.txt` directly in
joint space. The combined sequence first moves to `key_points/1.txt`, then
lowers the body to `--lower-body-height` before continuing with the remaining
poses. `scrape_execute` moves the right gripper to `--scrape-grip`
(default `0.5`); `scrape_move_away` closes it continuously to full closure over
`--gripper-close-seconds` (default `2.0`). The left gripper stays open.

Run all four phases with one command and a one-second pause between them:

```bash
ros2 run panther_examples pickup scrape_sequence \
	--key-point-dir /workspace/motion/pickup/key_points \
	--body-height 0.2 \
	--lower-body-height 0.0 \
	--scrape-grip 0.5 \
	--gripper-close-seconds 2.0 \
	--time-scale 2.0 \
	--between-delay 1.0
```

Use `--between-delay 0.0` for no pause, or increase it to leave more time
between key poses.

The default fetch planner is deterministic Cartesian motion: vertical stage,
lateral stage, then descent. To opt into the learned SEDS approach instead,
provide `--use-seds` and a model:

```bash
ros2 run panther_examples pickup \
	--use-seds \
	--model /workspace/motion/pickup/models/pickup_seds_g8.npz \
	--arm right \
	--target-xy 0.65 0.15 \
	--body-height 0.40 \
	--floor-z 0.05 \
	--approach-height 0.25 \
	--dry-run
```

`start` defaults to body height 0.4 m; `fetch` defaults to body height 0.0 m.
The fetch command lowers the body, approaches above the target, descends to
`--floor-z`, and closes the gripper. `--floor-z` is the hand's test clearance
above the base floor in metres. The node converts it into `body_link`, where
it may correctly be negative because `body_link` is above the floor.
`--body-height` is the separate torso/lift height.

The pickup node can run against the real robot as well as the mock simulation.
Use `--dry-run` first, verify the RViz path and reachable-prefix output, keep
people clear of both arms, and confirm the active ROS domain before removing
`--dry-run`.

The generated fetch keeps the wrist orientation from the live arm state by
default. The hardcoded CSV uses its own orientation columns. Override the
rough default home pose with eight values using `--home-joints`.

If the path fails at a later waypoint, the node reports the waypoint index and
publishes the reachable IK prefix to RViz. The planner automatically retries
speed-limit failures with 2x, 4x, and 8x retiming before giving up.

If IK rejects a waypoint during `--dry-run`, the requested Cartesian path is
still published on `/cartesian_follow/planned_path` as a straight marker path.
That lets you inspect the target in RViz even though it is not executable. The
printed `floor z` is the target height in `body_link`; a high `--body-height`
can make the floor unreachable because the arm has to reach farther downward.

# Master GUI

[`master_gui.py`](panther_examples/master_gui.py) opens one window with a slider for every arm, head and gripper joint, a lift slider, hold-to-drive base buttons, and record/play. It drives the robot's own controllers, so nothing is installed on the robot, and it runs the same in the sim and on the robot:

```bash
ros2 run panther_examples master_gui
ros2 launch panther_sim robot_rviz.launch.py   # on the robot, in a second terminal, to watch it in RViz
```

- **Arms:** `left_` and `right_arm_joint_trajectory_position_controller`, switched in for `dual_arm_forward_position_controller` as `cartesian_follow` does. Closing the window, or Ctrl-C, switches them back.
- **Head:** `head_forward_position_controller` jumps to each command, so the GUI sends it interpolated positions at 50 Hz.
- **Grippers:** `/left_gripper` and `/right_gripper/gripper_command`.
- A slider is sent when you release it, timed so no joint goes over 0.5 rad/s (grippers: a full stroke per second).
- The lift has no `ros2_control` command interface, so its slider publishes `/lifting_controller/commands` at 0.05 m/s.
- The base buttons publish `/cmd_vel` at 0.2 m/s or 0.4 rad/s while held, and a stop on release or when the window loses focus.
- **Record** samples the 20 joints and the lift at 10 Hz from the robot's state, however they move, into `teach_out/gui_<date>_<time>.csv`: rows `t,joint_la0..joint_rf,body_joint` under a `# t,<joint names>` header.
- **Play file…** takes any CSV with that header and any subset of those joints, so the `motion/retarget/joints_*.csv` files work too. The lift goes to the first row's height and stays there. Then the joints go to the first row at 0.5 rad/s and follow the file at its own timing, and joints not in the file hold. Values more than 0.05 past a limit are refused. Above 0.62 rad/s it asks first.
- **✋ Record by hand** (pick `right`, `left` or `both` next to it): switches that arm's trajectory controller off, then drops torque on each of its joints with `recover` then `disable`. On this robot, plain `disable` answers `False` and leaves the torque on. **Hold the arm before pressing OK.** Move it by hand while it records, as above. **🔒 Stop and lock** turns torque back on by restarting the arm's hardware component (`unix_openarm_left` / `right`, inactive then active), then switches the trajectory controller back in, holding the arm where you left it. On this robot, `enable` and `recover` cannot bring back a disabled joint; only that restart, or rebooting the robot, does (tested 2026-11-15). Closing the window with an arm free locks it first.
- **Stop all** holds every joint where it is and stops the lift and the base.

Typical use: set the lift with its slider, then either Record and move the arms and head with the sliders, or Record by hand. Stop, then Play file… to replay it.

Tested in the sim (2026-11-15), which runs the robot's controller names and topics. Not tried on the robot: the gripper `velocity` units (sent as strokes per second, as the sim reads them; check before closing a gripper on anything), whether the arm stays exactly where you left it when Stop and lock restarts its hardware (try small moves first), and the base speeds.

# Cartesian trajectory following (PyKDL)

[`cartesian_follow.py`](panther_examples/cartesian_follow.py) moves one arm's gripper along a fixed Cartesian trajectory from a CSV file. The robot has no Cartesian controller. The script solves IK on your machine with PyKDL and sends one `FollowJointTrajectory` goal to the robot's own `<arm>_arm_joint_trajectory_position_controller`. Nothing is installed on the robot.

```bash
ros2 run panther_examples cartesian_follow --arm left --path motion/example_path.csv --dry-run   # plan + check, no motion
ros2 run panther_examples cartesian_follow --arm left --path motion/example_path.csv             # move
```

## Trajectory file

One row per waypoint: `t,x,y,z,roll,pitch,yaw`, in s, m and rad. Lines starting with `#` are ignored. Time must strictly increase.

- **Frame:** the pose is `link_lf` (left) or `link_rf` (right), the gripper base (not the fingertips), expressed in `body_link`. `body_link` rides on the lift, so a trajectory stays valid at any lift height. The URDF has the lift fixed, so `base_footprint` coordinates would be wrong on the robot.
- **Spacing:** sample densely, about 0.1 s apart. The controller interpolates in joint space between points, so the hand only follows a straight line or a curve at the waypoints.
- **Start configuration (optional):** a line `# start_joints: q0 q1 ... q7` gives arm joint angles known to reach the first pose. The planner tries that configuration first. The arm is redundant, so the same hand pose has several arm configurations, and a path may only work from some of them.
- **Phone recordings:** [`motion/to_path/to_path.py`](../../motion/to_path/to_path.py) turns a phone recording (a frisbee throw) into a path for `--arm right`. Run it as `python3 motion/to_path/to_path.py RECORDING_DIR OUT.csv` (ROS sourced: it imports this package's planner). It shrinks and slows the motion and starts the hand at `START`. It ignores the phone's rotation: the gripper keeps one orientation, fingers along body +y and the gripper's plane in the body's y-z plane. A wrist flick turns the hand about body z from `FLICK_DEG[0]` to `FLICK_DEG[1]` over the last `FLICK_PART` of the throw, its speed ramping up to the release. A `SLOW_DOWN_S` deceleration follows, so the arm is not stopped dead at full speed. It picks the arm configuration needing the lowest peak joint speed, and checks the written file the way `cartesian_follow.py` will plan it. Each file's first line gives the `--max-joint-vel` it needs; the move to the start always runs at 0.5 rad/s or less. Tune `START`, `SCALE`, `SLOWDOWN`, `VEL_CAP` and `MIN_MARGIN` at the top of the file. The 20 throws in `utils/frisbee_throw_trajs/flatten/` are converted in `utils/frisbee_throw_trajs/paths/`.
- **Arm recordings (retargeting):** [`motion/retarget/`](../../motion/retarget/) holds elbow and wrist positions relative to the shoulder. First, [`visualize.py`](../../motion/retarget/visualize.py) picks each recording's start and release into `selections.csv`. Then [`retarget.py`](../../motion/retarget/retarget.py) copies the human arm's joint angles, not the hand path, onto the right arm. The shoulder (`ra1`–`ra3`) and elbow (`ra4`) are solved in closed form from the upper-arm and forearm directions. `ra2` only swings the arm outward, so `ra0` yaws it inward when it crosses the body. The wrist gets the same flick as `to_path`. Time is stretched to `VEL_CAP`, and angles outside the limits are clipped. Each `joints_*.csv` header gives the slowdown and how far the arm ends up from the human pose. The recording's axes are taken as body_link's (`TO_BODY`; not verified). Run the result with `--arm right --joints FILE`.
- **Joint files:** `--joints FILE` instead of `--path`: rows `t,q0..q7` in the arm's joint order (`joint_ra0..7`), no IK. The same speed check, approach move, controller switch and report apply; a joint outside its limits is refused.
- **Example:** [`motion/example_path.csv`](../../motion/example_path.csv) is a 5 cm vertical circle in front of the left shoulder, in 10 s. It starts with the elbow bent. Starting with the arm hanging straight is a singularity, which IK cannot leave smoothly.

## Seeing the plan in RViz

The script publishes the planned hand path on `/cartesian_follow/planned_path` (`nav_msgs/Path`, frame `body_link`, latched). The sim's RViz config shows it as a cyan line ("Planned hand path"). With `--dry-run` the script stays up so the path stays visible; press Ctrl-C to exit. After a real move, RViz keeps the last path drawn. The path is the forward kinematics of the IK solutions, so it shows what will be executed, not just the CSV.

## What it checks

Before sending anything, it refuses if any of these fail:

1. Every waypoint is reachable within the URDF joint limits. It uses KDL `ChainIkSolverPos_NR_JL`, seeded from the previous waypoint; for the first waypoint it tries the current joints plus 50 random seeds and keeps the closest solution.
2. The peak joint speed of the cubic spline the controller will actually run is at most 0.5 rad/s (`--max-joint-vel`). This includes the move from the current pose to the first waypoint, which is timed automatically.

Spline velocities are monotone: zero where a joint changes direction. So the controller's interpolation never overshoots a waypoint, and a joint resting on its limit stays there.

During the move it records `/joint_states`, then prints any position-limit or speed violation, the peak joint speed, and the largest distance between the hand and the plan. Exit codes: 0 = limits respected, 1 = refused (nothing sent), 2 = moved but a limit was exceeded.

Not checked: acceleration, collisions (between the arms, or with the torso), the gripper.

## Controller switching

This copies PR #521. If `dual_arm_forward_position_controller` is active, it is deactivated and the arm's trajectory controller is activated for the move. Both are put back afterwards, including after Ctrl-C, which also cancels the goal. The sim starts the controllers the same way as the robot (see [`../mock-sim`](../mock-sim/README.md)), so this path is tested.

## Sim test (2026-11-14, mock hardware)

| Case | Result |
| --- | --- |
| `example_path.csv` from the arm hanging straight | moved; peak 0.500 rad/s, hand within 0.5 mm of plan, controllers restored |
| same circle 4× faster | moved; needs only 0.34 rad/s |
| same circle 20× faster | refused: `joint_la7` would reach 1.71 rad/s |
| circle shifted 1 m forward | refused: waypoint 0 out of reach |

Mock hardware follows commands exactly, so the sim checks the plan, not how well the real arm tracks it.

## On the robot

- **Always `--dry-run` first.** It only reads `/robot_description` and `/joint_states`.
- **Speed limit:** 0.5 rad/s is our choice; the vendor publishes no limit. PR #521 moves the arms at 0.35 rad/s average.
- **Lift:** a `body_link` trajectory follows the lift. If the lift moves during a run, the hand moves with it.
