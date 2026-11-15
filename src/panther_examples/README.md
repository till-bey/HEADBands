# panther_examples

Robot scripts. Run them in the container (`docker/run.sh`, or `docker/run.sh --sim` against the mock sim) after `colcon build && source install/setup.bash`. Joint, controller, hardware and topic names and the lift range live in [`robot.py`](panther_examples/robot.py); change them there, not in the scripts.

| Command | What it does |
| --- | --- |
| `ros2 run panther_examples joint_check [--yes]` | Moves every joint (arms, head, grippers, lift) a small step and back, checks the measured position follows. Without `--yes` it prints the plan only |
| `ros2 run panther_examples teach ...` | Record both arms and grippers moved by hand, adjust the lift, play back. Steps in the [script docstring](panther_examples/teach.py) |
| `ros2 run panther_examples cartesian_follow ...` | Cartesian path for one arm, IK with PyKDL (below) |
| `ros2 run panther_examples master_gui` | One window for both arms, head, grippers, lift and base ([below](#master-gui)) |
| `ros2 run panther_examples commands` | Scratchpad of single commands (gripper, base, head, dual arm); edit `main()` |
| `ros2 run panther_examples pickup` | Picks a disk up from the floor with both arms ([below](#disk-pickup)) |

# Disk pickup

[`pickup.py`](panther_examples/pickup.py) picks a disk up from the floor by playing the four joint-space key poses in
[`motion/pickup/key_points/`](../../motion/pickup/key_points/) (`ros2 topic echo /joint_states --once` dumps) on both
arms:

1. lift to `--body-height` (0.19 m, kept for the whole pickup), both grippers open, move to `1.txt`
2. `2.txt`
3. `3.txt`, then the right gripper to `--scrape-grip` (0.5)
4. right gripper closed over `--gripper-close-seconds` (0.5 s), then `4.txt` (move away)

Each pose takes `--phase-seconds` (8 s) with `--between-delay` (1 s) between them. Joints past the URDF limits are clamped with a warning.

```bash
ros2 run panther_examples pickup --dry-run   # load and check the key poses only
ros2 run panther_examples pickup
```

From the web interface, say or type **"pick up the disk"**: the bridge plays the same sequence as one trajectory
(`swiss_robot_bridge.local_safety.pickup_points`), after confirmation.

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
