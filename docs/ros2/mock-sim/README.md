# Panther mock sim: rqt_joint_trajectory_controller + RViz2

Arms, head, lift and grippers on `ros2_control` mock hardware (`mock_components/GenericSystem`): commanded position = reported position, no physics, no driving base. Use it to try trajectories before the real robot.

```bash
source /opt/ros/jazzy/setup.bash          # or humble
# once, from the repo root, for the gripper topics: colcon build --packages-select unix_msgs
source install/setup.bash                 # unix_msgs; without it the gripper topics are off
ros2 launch docs/ros2/mock-sim/mock_sim.launch.py
```

One command starts the sim, RViz and rqt. The sim takes the robot's own command topics (see [Controllers](#controllers)), so code written for the robot runs here unchanged.

As on the robot, `dual_arm_forward_position_controller` owns both arms at startup, and the arm trajectory controllers are loaded but inactive. rqt lists only active trajectory controllers, so switch one in first (in a terminal set up as below):

```bash
ros2 control switch_controllers --deactivate dual_arm_forward_position_controller \
  --activate left_arm_joint_trajectory_position_controller
```

In rqt, open the **controller manager ns** dropdown and pick `/controller_manager`. Then open **controller** and pick one; it lists after about 5 s. Press the red power button so it turns green, and drag the sliders. The dropdowns start blank even when they have entries. [`../cartesian/cartesian_follow.py`](../cartesian/README.md) does this switch itself and switches back afterwards.

RViz: left-drag rotates, middle-drag pans, scroll zooms (Interact tool).

## Isolation from the robot

The launch file forces these for its own nodes, whatever your terminal has:

| Variable | Sim value | Why |
| --- | --- | --- |
| `ROS_DOMAIN_ID` | `77` | The sim uses the robot's controller names. On the robot's domain, rqt could command the real arms. |
| `ROS_AUTOMATIC_DISCOVERY_RANGE` | `LOCALHOST` | Stays on this machine. |
| `CYCLONEDDS_URI` | unset | Robot setups pin it to the robot's USB-Ethernet NIC. With the cable out, Cyclone cannot discover anything. |
| `RMW_IMPLEMENTATION` | `rmw_cyclonedds_cpp` | `rqt_joint_trajectory_controller` 4.42 never finds the controller manager under Fast DDS. The robot runs Cyclone too. |

So a terminal set up for the robot is fine for launching. For CLI commands against the sim, use a terminal with:

```bash
source /opt/ros/jazzy/setup.bash
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=77 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 daemon stop    # drop a daemon cached with the robot settings
```

## Controllers

The robot's command topics work here, with the robot's message types:

| Robot interface | Sim | Joints |
| --- | --- | --- |
| `dual_arm_forward_position_controller` (active at start) | same controller | `joint_la0..7`, `joint_ra0..7` |
| `left_arm_joint_trajectory_position_controller` (inactive at start) | same controller | `joint_la0..7` |
| `right_arm_joint_trajectory_position_controller` (inactive at start) | same controller | `joint_ra0..7` |
| `head_forward_position_controller` (active at start) | same controller | `joint_head0`, `joint_head1` |
| `/lifting_controller/commands` (JointState: `position` m, or `velocity` m/s) | `robot_topics.py` → `lifting_joint_trajectory_controller` | `body_joint`, 0 to 0.6 m, at most 0.05 m/s |
| `/left_gripper/gripper_command`, `/right_gripper/gripper_command` (`unix_msgs/GripperCommand`) | `robot_topics.py` → `left_` / `right_gripper_joint_trajectory_controller` | `joint_lf` / `joint_rf`, 0 open to 1 closed |
| `/lifting_controller/joint_states`, `/left_gripper/joint_states`, `/right_gripper/joint_states` | published by `robot_topics.py` | |

The robot's lift and gripper controllers are vendor plugins that only exist on the robot. [`robot_topics.py`](robot_topics.py), started by the launch file, takes their topics and forwards each command as a trajectory to a sim-only controller. Its behaviour is a guess where the vendor does not document it:

- Lift `position`: moves there at 0.05 m/s, or at the command's `velocity` if that is lower.
- Lift `velocity` alone: moves toward the end stop until the next command. Whether the robot also stops when commands stop arriving is unknown.
- Gripper: `mode` is ignored. It moves at `command.velocity` in full strokes per second, or 1 if that is missing. The real gripper speed is not measured.

Not in the sim: `/cmd_vel` (no driving base), `mit_controller`, and the battery, IMU and diagnostics broadcasters. Unlike the robot, `/joint_states` here also carries `body_joint`, `joint_lf` and `joint_rf`, because `robot_state_publisher` needs them.

For rqt, which drives only trajectory controllers: the lift and gripper controllers are already active. For the head, switch it in as for an arm:

```bash
ros2 control switch_controllers --deactivate head_forward_position_controller --activate head_joint_trajectory_controller
```

Examples, as on the robot (from [the interface reference](../UnixAI_Panther_ROS2_Interface_Reference.md)):

```bash
ros2 topic pub --once /lifting_controller/commands sensor_msgs/msg/JointState "{name: [''], position: [0.5]}"
ros2 topic pub --once /left_gripper/gripper_command unix_msgs/msg/GripperCommand "{mode: 'parallel', command: {name: [''], position: [1.0]}}"
ros2 topic pub --once /head_forward_position_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.3, -0.2]}"
```

The arm trajectory controllers reject partial goals, so send all 8 joints:

```bash
ros2 action send_goal /left_arm_joint_trajectory_position_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "{trajectory: {joint_names: [joint_la0, joint_la1, joint_la2, joint_la3, joint_la4, joint_la5, joint_la6, joint_la7],
    points: [{positions: [0, 0.5, 0, 0, 0, 0, 0, 0], time_from_start: {sec: 2}}]}}"
```

## Robot model

`panther.urdf` is the stripped vendor URDF from 4paradigm/phanthymotus-driver PR #521 (`unixai/panther/resource/panther.urdf`, identical in #520). It has no meshes, so the launch file draws stand-in geometry: a box base, wheels, a torso column, rods for links, and orange spheres at the moving joints. Kinematics are the vendor's; the shapes are not.

The real meshes are only on the robot, in the `mh3_description` package; they are not public. To use them, connect to the robot. Then:

```bash
# 1. the full URDF the robot publishes (run in your robot terminal)
python3 -c "
import rclpy; from std_msgs.msg import String; from rclpy.qos import QoSProfile, DurabilityPolicy
rclpy.init(); n = rclpy.create_node('grab')
def cb(m): open('mh3_full.urdf', 'w').write(m.data); raise SystemExit
n.create_subscription(String, '/robot_description', cb, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
rclpy.spin(n)"
# 2. the meshes
scp -r <user>@<robot>:/opt/ros/humble/share/mh3_description ~/mh3_description
# 3. launch with both
ros2 launch docs/ros2/mock-sim/mock_sim.launch.py urdf:=$PWD/mh3_full.urdf meshes:=$HOME/mh3_description
```

The launch file removes the vendor `<ros2_control>` and `<gazebo>` blocks, because they load CAN plugins that only exist on the robot. It also points the `file:///opt/ros/humble/share/mh3_description` mesh paths at `meshes:=`.

The vendor URDF has the lift and grippers fixed, and `ros2_control` rejects fixed joints, so the launch file changes the sim copy:

- `body_joint` becomes prismatic, with the vendor axis (`0 0 -1`) and limit (0 to 0.6 m) and a 0.05 m/s velocity (the SDK limit). With that axis a larger value lowers the torso. Whether the robot's lift goes the same way is not confirmed.
- The vendor `joint_lf` / `joint_rf` stay fixed, renamed `joint_lf_mount` / `joint_rf_mount`, so `link_lf` / `link_rf` and the IK in `cartesian_follow.py` are unchanged. A new `joint_lf` / `joint_rf` (0 to 1) hangs off them on an empty link, and the two finger roots of each gripper (`joint_lf00`, `joint_lf10`) mimic it with 0.7 rad of swing at 1.0. The swing is a stand-in so the gripper visibly closes, not the real finger kinematics.

## Gotchas

- `Controller already loaded, skipping load_controller` and then `no controller with this name exists` mean another sim is already running on domain 77, and your spawners talked to its controller manager. Stop it first: `pkill -f mock_sim.launch.py`.

- The vendor URDF has `velocity="0"` on every arm joint. rqt divides by it and crashes. The launch file sets the sim copy to 0.5 rad/s, our chosen limit, not a vendor spec. The robot's own `/robot_description` has the same zeros, so rqt pointed at the real robot will hit this too.
- From a VS Code **snap** terminal, RViz and rqt crash (`libpthread ... GLIBC_PRIVATE`, `GLib-GIO-ERROR`). Run them from a normal terminal.
