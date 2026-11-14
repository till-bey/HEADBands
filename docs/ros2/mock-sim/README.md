# Panther mock sim: rqt_joint_trajectory_controller + RViz2

Arms and head on `ros2_control` mock hardware (`mock_components/GenericSystem`): commanded position = reported position, no physics, no base, no grippers, no lift. Use it to try trajectories before the real robot.

```bash
source /opt/ros/jazzy/setup.bash          # or humble
ros2 launch docs/ros2/mock-sim/mock_sim.launch.py
```

One command starts the sim, RViz and rqt. In rqt, open the **controller manager ns** dropdown and pick `/controller_manager`. Then open **controller** and pick one; it lists after about 5 s. Press the red power button so it turns green, and drag the sliders. The dropdowns start blank even when they have entries.

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

Names as on the robot:

| Controller | Joints |
| --- | --- |
| `left_arm_joint_trajectory_position_controller` | `joint_la0..7` |
| `right_arm_joint_trajectory_position_controller` | `joint_ra0..7` |
| `head_joint_trajectory_controller` (sim only; the robot uses `head_forward_position_controller`) | `joint_head0`, `joint_head1` |

From the CLI, send all 8 arm joints. The controller rejects partial goals:

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

## Gotchas

- The vendor URDF has `velocity="0"` on every arm joint. rqt divides by it and crashes. The launch file sets the sim copy to 1.0 rad/s, which is a guess, not a robot spec. The robot's own `/robot_description` has the same zeros, so rqt pointed at the real robot will hit this too.
- From a VS Code **snap** terminal, RViz and rqt crash (`libpthread ... GLIBC_PRIVATE`, `GLib-GIO-ERROR`). Run them from a normal terminal.
