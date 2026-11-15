# panther_examples

Robot scripts. Run them in the container (`docker/run.sh`, or `docker/run.sh --sim` against the mock sim) after `colcon build && source install/setup.bash`. Joint, controller, hardware and topic names and the lift range live in [`robot.py`](panther_examples/robot.py); change them there, not in the scripts.

| Command | What it does |
| --- | --- |
| `ros2 run panther_examples joint_check [--yes]` | Moves every joint (arms, head, grippers, lift) a small step and back, checks the measured position follows. Without `--yes` it prints the plan only |
| `ros2 run panther_examples teach ...` | Record both arms and grippers moved by hand, adjust the lift, play back. Steps in the [script docstring](panther_examples/teach.py) |
| `ros2 run panther_examples cartesian_follow ...` | Cartesian path for one arm, IK with PyKDL (below) |
| `ros2 run panther_examples commands` | Scratchpad of single commands (gripper, base, head, dual arm); edit `main()` |

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
