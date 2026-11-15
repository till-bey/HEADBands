# Panther mock sim: rqt_joint_trajectory_controller + RViz2

Arms, head, lift and grippers on `ros2_control` mock hardware (`mock_components/GenericSystem`): commanded position = reported position, no physics, no driving base. Use it to try trajectories before the real robot.

```bash
docker/run.sh --sim                       # from the repo root; domain 77, loopback only
colcon build && source install/setup.bash
ros2 launch panther_sim mock_sim.launch.py
```

One command starts the sim, RViz and rqt. The sim has the robot's own controller names, hardware component names, command topics and CAN enable/disable services (see [Controllers](#controllers)), so the scripts in `panther_examples` (`teach`, `joint_check`, `cartesian_follow`, `commands`) run here unchanged. Open a second `docker/run.sh --sim` terminal for them.

As on the robot, `dual_arm_forward_position_controller` owns both arms at startup, and the arm trajectory controllers are loaded but inactive. rqt lists only active trajectory controllers, so switch one in first (in a terminal set up as below):

```bash
ros2 control switch_controllers --deactivate dual_arm_forward_position_controller \
  --activate left_arm_joint_trajectory_position_controller
```

In rqt, open the **controller manager ns** dropdown and pick `/controller_manager`. Then open **controller** and pick one; it lists after about 5 s. Press the red power button so it turns green, and drag the sliders. The dropdowns start blank even when they have entries. [`cartesian_follow`](../panther_examples/README.md) does this switch itself and switches back afterwards.

RViz: left-drag rotates, middle-drag pans, scroll zooms (Interact tool).

## Isolation from the robot

The launch file forces these for its own nodes, whatever your terminal has:

| Variable | Sim value | Why |
| --- | --- | --- |
| `ROS_DOMAIN_ID` | `77` | The sim uses the robot's controller names. On the robot's domain, rqt could command the real arms. |
| `CYCLONEDDS_URI` | loopback `lo`, multicast forced on | Stays on this machine, and replaces a robot setup's NIC pinning. Humble's `ROS_LOCALHOST_ONLY=1` is not used: with Cyclone it allows only a few participants, and the sim has about ten (`Failed to find a free participant index`). |
| `RMW_IMPLEMENTATION` | `rmw_cyclonedds_cpp` | `rqt_joint_trajectory_controller` can miss the controller manager under Fast DDS. The robot runs Cyclone too. |

`docker/run.sh --sim` sets the same for every command in that container. Outside Docker, use a Humble terminal with:

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
export ROS_DOMAIN_ID=77 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" multicast="true"/></Interfaces></General></Domain></CycloneDDS>'
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
| `lifting_controller`, `/lifting_controller/commands` (JointState: `position` m, or `velocity` m/s) | `robot_topics` → `lifting_controller` (a trajectory controller here) | `body_joint`, 0 (bottom) to 0.8 m (top), at most 0.05 m/s |
| `left_` / `right_unix_gripper3_controller`, `/left_gripper/gripper_command`, `/right_gripper/gripper_command` (`unix_msgs/GripperCommand`) | `robot_topics` → `left_` / `right_unix_gripper3_controller` (trajectory controllers here) | `joint_lf` / `joint_rf`, 0 open to 1 closed |
| `/lifting_controller/joint_states`, `/left_gripper/joint_states`, `/right_gripper/joint_states` | published by `robot_topics` | |
| `/can0/joint_la*/enable\|disable`, `/can1/joint_ra*/enable\|disable` (`std_srvs/Trigger`) | served by `robot_topics`, always succeed | mock hardware has no torque to drop |
| Hardware components `unix_openarm_left`, `unix_openarm_right`, `raw_can0`, `raw_can1`, `mh3_head`, `mh3_movebase` | one mock system each, same joints as on the robot | `set_hardware_component_state` works |

The robot's lift and gripper controllers are vendor plugins that only exist on the robot. Here they are trajectory controllers with the same names, and [`robot_topics`](panther_sim/robot_topics.py), started by the launch file, takes the robot's topics and forwards each command to them as a trajectory. Deactivating a gripper controller (as `teach release` does) stops that gripper, as on the robot. Its behaviour is a guess where the vendor does not document it:

- Lift `position`: moves there at 0.05 m/s, or at the command's `velocity` if that is lower.
- Lift `velocity` alone: moves toward the end stop until the next command. Whether the robot also stops when commands stop arriving is unknown.
- Gripper: `mode` is ignored. It moves at `command.velocity` in full strokes per second, or 1 if that is missing. The real gripper speed is not measured.

Not in the sim: `/cmd_vel` (no driving base), `mit_controller`, and the battery, IMU and diagnostics broadcasters. Unlike the robot, `/joint_states` here also carries `body_joint`, `joint_lf` and `joint_rf`, because `robot_state_publisher` needs them.

For rqt, which drives only trajectory controllers: the lift and gripper controllers are already active. For the head, switch it in as for an arm:

```bash
ros2 control switch_controllers --deactivate head_forward_position_controller --activate head_joint_trajectory_controller
```

Examples, as on the robot (from [the interface reference](../../docs/ros2/UnixAI_Panther_ROS2_Interface_Reference.md)):

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

[`urdf/robot_description.urdf`](urdf/robot_description.urdf) is the robot's own `/robot_description` (xacro output of `mh3_description`). The sim, `cartesian_follow.py`'s IK on the robot, and the `motion/` scripts all use these kinematics. On the robot, `cartesian_follow.py` reads the live `/robot_description` instead.

The meshes are only on the robot, in the `mh3_description` package; they are not public. Without them the launch file draws stand-in geometry: a box base, wheels, a torso column, rods for links, and orange spheres at the moving joints. To use the real meshes, copy them from the robot:

```bash
scp -r <user>@<robot>:/home/unix_ai/unixai_ws/install/mh3_description/share/mh3_description ~/mh3_description
ros2 launch panther_sim mock_sim.launch.py meshes:=$HOME/mh3_description
```

To refresh the URDF after a robot update, save its `/robot_description` over the file (run in your robot terminal):

```bash
python3 -c "
import rclpy; from std_msgs.msg import String; from rclpy.qos import QoSProfile, DurabilityPolicy
rclpy.init(); n = rclpy.create_node('grab')
def cb(m): open('src/panther_sim/urdf/robot_description.urdf', 'w').write(m.data); raise SystemExit
n.create_subscription(String, '/robot_description', cb, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
rclpy.spin(n)"
```

The launch file removes the vendor `<ros2_control>` and `<gazebo>` blocks, because they load CAN plugins that only exist on the robot. It also points the vendor `file://.../share/mh3_description` mesh paths at `meshes:=`, and flips the lift axis so 0 is the bottom.

The vendor URDF has the lift and grippers fixed, and `ros2_control` rejects fixed joints, so the launch file changes the sim copy:

- `body_joint` becomes prismatic, 0 (bottom) to 0.8 m (top), at 0.05 m/s (the SDK limit). The vendor URDF has it backwards (axis `0 0 -1`, limit 0 to 0.6), so the sim flips the axis to `0 0 1` and takes the vendor body height (1.363 m above `base_link`) as the top. At 0 the body is 0.563 m above `base_link`. The sim starts at 0.4 (body 0.963 m above `base_link`).
- The vendor `joint_lf` / `joint_rf` stay fixed, renamed `joint_lf_mount` / `joint_rf_mount`, so `link_lf` / `link_rf` and the IK in `cartesian_follow.py` are unchanged. A new `joint_lf` / `joint_rf` (0 to 1) hangs off them on an empty link, and the two finger roots of each gripper (`joint_lf00`, `joint_lf10`) mimic it with 0.7 rad of swing at 1.0. The swing is a stand-in so the gripper visibly closes, not the real finger kinematics.

## Gotchas

- Run one sim at a time. On domain 77, a second sim, or a `robot_state_publisher` left over from a killed launch, clashes with yours. Symptoms: `Controller already loaded, skipping load_controller` then `no controller with this name exists`, or `command interface 'body_joint/position' is not available` (the controller manager loaded the old URDF). A leftover publisher shows as `Publisher count: 2` in `ros2 topic info /robot_description`. Stop everything, then relaunch: `pkill -f 'mock_sim.launch.py|robot_state_publisher|ros2_control_node'`.

- The vendor URDF has `velocity="0"` on every arm joint. rqt divides by it and crashes. The launch file sets the sim copy to 0.5 rad/s, our chosen limit, not a vendor spec. The robot's own `/robot_description` has the same zeros, so rqt pointed at the real robot will hit this too.
- From a VS Code **snap** terminal, RViz and rqt crash (`libpthread ... GLIBC_PRIVATE`, `GLib-GIO-ERROR`). Run them from a normal terminal.
