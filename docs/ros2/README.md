# Panther ROS 2 stack and API

What the UnixAI Panther SDK (English, updated 15 Jun 2026) actually exposes, and what to confirm on the robot before sending motion.

The SDK is a command cheat-sheet for a stack that is already installed on the Jetson. It does not name the ROS 2 distro, and it does not include a URDF, a launch file for the whole robot, or message definitions. Those have to be read off the machine.

Floor rules, charging, and the Quest teleop UI are in [the operator README](../robot-operation/README.md).

## Log in

The controller user from the SDK:

| | |
| --- | --- |
| User | `unix_ai` |
| Password | `unixai` |
| Host | the industrial PC on `192.168.15.0/24` (see the operator README; the SDK says `ssh unix_ai@ip`) |

```bash
ssh unix_ai@192.168.15.XX
```

Two workspaces are named:

| Path | Used for |
| --- | --- |
| `/home/unix_ai/work/controller` | Controller install. Source this before gripper commands, because they use `unix_msgs`. |
| `/home/unix_ai/orbbec_ws` | Orbbec Gemini 2 driver |
| `unix_motor_tool/` | Arm zero calibration. The SDK `cd`s into it with a relative path, so find it with `find /home/unix_ai -type d -name unix_motor_tool`. |

```bash
source /home/unix_ai/work/controller/install/setup.bash
```

## Confirm the distro before you trust a laptop build

The SDK uses `ros2` CLI, `ros2_control`, and `ros2 topic pub`. It never prints `ROS_DISTRO`. The computer is a Jetson AGX Orin 64 GB. Run this once and write the result in the team notes:

```bash
echo "$ROS_DISTRO"
lsb_release -ds
cat /etc/nv_tegra_release 2>/dev/null || true
```

Match the cloud simulator to that distro. A Humble laptop cannot talk to a Jazzy graph, and controller YAML from one distro often will not load on the other.

Also record the RMW:

```bash
echo "$RMW_IMPLEMENTATION"
echo "$ROS_DOMAIN_ID"
```

## The stack that is already running

Boot brings up `ros2_control`. You do not launch a hardware bringup from the SDK. You inspect it, then publish.

Healthy hardware (`ros2 control list_hardware_components`) is six components, all `state: id=3 label=active`:

| # | Name | Plugin | Command interfaces |
| --- | --- | --- | --- |
| 1 | `raw_can1` | `unix_can_hardware_interface/UnixCanHardwareInterface` | `joint_rf` position, velocity, effort |
| 2 | `raw_can0` | `unix_can_hardware_interface/UnixCanHardwareInterface` | `joint_lf` position, velocity, effort |
| 3 | `unix_openarm_right` | `canopen_ros2_control/RobotSystem` | `joint_ra0` … `joint_ra7` position |
| 4 | `unix_openarm_left` | `canopen_ros2_control/RobotSystem` | `joint_la0` … `joint_la7` position |
| 5 | `mh3_head` | `hhs_can_hardware_interface/HhsCanHardwareInterface` | `joint_head0`, `joint_head1`: position (claimed); velocity, increment position, effort, max velocity, max effort (available, unclaimed) |
| 6 | `mh3_movebase` | `hhs_can_hardware_interface/HhsCanHardwareInterface` | `movebase3/work_mode`, `linear_x`, `linear_y`, `angular_z`; `body_joint/lifting_work_mode`, `position`, `velocity` |

`joint_lf` and `joint_rf` are not explained. They are separate from the swerve command interface on `mh3_movebase` and from the gripper controllers. Do not publish them until you have found the YAML that owns them.

Healthy controllers (`ros2 control list_controllers`):

| Controller | Type | Boot state |
| --- | --- | --- |
| `diagnostic_broadcaster` | `diagnostic_broadcaster/DiagnosticBroadcaster` | active |
| `lifting_joint_state_broadcaster` | `joint_state_broadcaster/JointStateBroadcaster` | active |
| `dual_arm_forward_position_controller` | `forward_command_controller/ForwardCommandController` | active |
| `lifting_controller` | `lifting_controller3/LiftingController3` | active |
| `head_forward_position_controller` | `forward_command_controller/ForwardCommandController` | active |
| `movebase3_drive_controller` | `unix_swerve_drive_controller/UnixSwerveDriveController` | active |
| `joint_state_broadcaster` | `joint_state_broadcaster/JointStateBroadcaster` | active |
| `left_arm_joint_trajectory_position_controller` | `joint_trajectory_controller/JointTrajectoryController` | **inactive** |
| `right_arm_joint_trajectory_position_controller` | `joint_trajectory_controller/JointTrajectoryController` | **inactive** |
| `left_unix_gripper3_controller` | `omni_picker_controller/OmniPickerController` | active |
| `right_unix_gripper3_controller` | `omni_picker_controller/OmniPickerController` | active |

The arms wake up under the forward-position controller, not the trajectory controller. VR teleop is running against whatever is active. Switching controllers takes the arms away from that owner. Say so out loud before you switch, and switch back when you are done:

```bash
ros2 control switch_controllers \
  --activate left_arm_joint_trajectory_position_controller \
             right_arm_joint_trajectory_position_controller \
  --deactivate dual_arm_forward_position_controller
```

## Joint names

| Group | Names | Notes from the SDK and the user guide |
| --- | --- | --- |
| Left arm | `joint_la0` … `joint_la7` | 8 position interfaces. CANopen via `unix_openarm_left`. Limits are not in the SDK. |
| Right arm | `joint_ra0` … `joint_ra7` | Same, on `unix_openarm_right`. |
| Head | `joint_head0`, `joint_head1` | User guide: yaw ±90°, pitch 20° up / 86° down. Which index is yaw is not written down. |
| Lift | `body_joint` on the hardware interface | Command range 0.0–0.8 m, velocity −0.05–0.05 m/s. Matches the 800 mm torso travel in the user guide. |
| Base | `movebase3` linear_x, linear_y, angular_z | Holonomic swerve. User guide max speed 2 m/s. |
| Wheels in `/joint_states` | `left_wheel_joint`, `right_wheel_joint` | Present in the sample `JointState`. No command topic is documented for them. |
| Extra CAN joints | `joint_lf`, `joint_rf` | See above. Unmapped. |
| Grippers | separate state topics, not in the sample `/joint_states` name list | Command is 0.0 open, 1.0 closed, mode `parallel`. |

`/joint_states` in the SDK sample lists `joint_la0`–`joint_la7`, `joint_ra0`–`joint_ra7`, `joint_head0`, `joint_head1`, `left_wheel_joint`, `right_wheel_joint`. Order in the message is not joint order. Velocities and efforts in that sample are NaN.

## Command API

Send motion only with the arms free of people and of the floor in front of them. The forward-position topic streams setpoints with no built-in time scaling. A single bad array can command a large step.

### Both arms, streaming position (this is the controller that is active at boot)

| | |
| --- | --- |
| Topic | `/dual_arm_forward_position_controller/commands` |
| Type | `std_msgs/msg/Float64MultiArray` |
| Rate in the example | 50 Hz (`time.sleep(1/50)` inside a loop over a pre-planned trajectory) |

The SDK’s snippet is pseudocode, not a runnable shell line: it indexes a NumPy array and then shows `ros2 topic pub`. The contract is: you plan a smooth trajectory yourself, then stream one `Float64MultiArray` per tick for both arms.

The array length and joint order are **not** specified. Read them before the first publish:

```bash
ros2 control list_controllers -v
grep -R "dual_arm_forward" -n /home/unix_ai/work/controller --include='*.yaml' --include='*.yml'
```

Sixteen numbers (left 8 then right 8, or the reverse) is the layout to expect. Confirm it. Do not publish zeros “to see what happens”; the trajectory examples use an all-zero pose as a real target, held over 6 seconds, and only after the arm is at the marked zero.

### One arm, point to point (controller is inactive until you switch)

Activate the trajectory controllers and deactivate the forward controller first (command above).

| Arm | Topic | Type |
| --- | --- | --- |
| Left | `/left_arm_joint_trajectory_position_controller/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` |
| Right | `/right_arm_joint_trajectory_position_controller/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` |

The SDK’s left-arm examples use `time_from_start` of 6 seconds. First sample is the zero pose. Second sample is a specific pose. Third sample moves only `joint_la5`.

Zero pose (SDK example, 6 s):

```bash
ros2 topic pub --once \
  /left_arm_joint_trajectory_position_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory "{
    joint_names: ['joint_la0','joint_la1','joint_la2','joint_la3','joint_la4','joint_la5','joint_la6','joint_la7'],
    points: [{ positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 6} }]
  }"
```

Documented non-zero left-arm pose, also 6 s:

```text
[0.0, 0.19, 0.37, -0.93, -1.0, 1.779, 0.107, -0.204]
```

on `joint_la0` … `joint_la7`, same topic and message shape.

Single joint (SDK example): `joint_la5` to `0.2` rad over 6 s. Other joints are omitted from `joint_names`, which is legal for `JointTrajectory` only for the joints you name; the controller config decides what the others do. Prefer naming all eight joints once you know a full safe pose.

The right-arm topic takes the same message with `joint_ra0` … `joint_ra7`. The SDK does not give a right-arm sample pose.

### Base

| | |
| --- | --- |
| Topic | `/cmd_vel` |
| Type | `geometry_msgs/msg/Twist` |
| `linear.x` | forward / back |
| `linear.y` | left / right |
| `angular.z` | yaw |

```bash
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.2}}"
```

That sample only yaws, at 10 Hz. Standard ROS units apply (m/s, rad/s); the SDK itself only labels the axes. The user-guide top speed is 2 m/s. Stop a latched publisher with Ctrl-C; a `Twist` left latched at a non-zero rate will keep the base moving. Prefer `--once` while learning, or a publisher you can kill, and stay under a few centimetres per second until ultrasonics are clearly active.

`/cmd_vel` is consumed by `movebase3_drive_controller` (`unix_swerve_drive_controller/UnixSwerveDriveController`). There is no documented nav2, odom, or TF setup.

### Grippers

Source the controller overlay first so `unix_msgs` exists.

| | |
| --- | --- |
| Topics | `/left_gripper/gripper_command`, `/right_gripper/gripper_command` |
| Type | `unix_msgs/msg/GripperCommand` |
| Open | `mode: parallel`, `command.position: [0.0]` |
| Close | `mode: parallel`, `command.position: [1.0]` |

```bash
source /home/unix_ai/work/controller/install/setup.bash

ros2 topic pub --once /left_gripper/gripper_command unix_msgs/msg/GripperCommand "{
  mode: 'parallel',
  command: { name: [''], position: [0.0] }
}"
```

The nested `command` field is filled like a `sensor_msgs/JointState` (name + position). Dump the real definition before you wrap it:

```bash
ros2 interface show unix_msgs/msg/GripperCommand
```

Use `1.0` in `position` to close. The right gripper uses the same message on `/right_gripper/gripper_command`.

### Lift

| | |
| --- | --- |
| Topic | `/lifting_controller/commands` |
| Type | `sensor_msgs/msg/JointState` |
| Position | 0.0–0.8 m. Leave `velocity` empty. |
| Velocity | −0.05–0.05 m/s. Leave `position` empty. The SDK uses `--once` for the velocity example. |

Position example from the SDK (0.75 m):

```bash
ros2 topic pub --once /lifting_controller/commands sensor_msgs/msg/JointState \
  "{name: [''], position: [0.75], velocity: [], effort: []}"
```

Velocity example from the SDK (0.02 m/s):

```bash
ros2 topic pub --once /lifting_controller/commands sensor_msgs/msg/JointState \
  "{name: [''], position: [], velocity: [0.02], effort: []}"
```

A sample hardware-bridge log shows `lifting_controller` warning `Actual position is NaN, cannot update controller` together with CAN send failures. If the lift does not move, read `/tmp/hwb_log.txt` before sending another command.

### Head

| | |
| --- | --- |
| Topic | `/head_forward_position_controller/commands` |
| Type | `std_msgs/msg/Float64MultiArray` |
| Example | `data: [0.1, 0.1]` |

Two values, two joints. The SDK does not say which element is `joint_head0`. User-guide ranges are yaw ±90° (±1.57 rad) and pitch +20° / −86° (+0.35 / −1.50 rad). Stay near the example until you have watched `/joint_states` while someone enables Head tracking in the VR app.

```bash
ros2 topic pub --once /head_forward_position_controller/commands \
  std_msgs/msg/Float64MultiArray "{data: [0.1, 0.1]}"
```

### Ultrasonic enable

| | |
| --- | --- |
| Service | `/movebase3_drive_controller/ultrasonic_switch` |
| Type | `std_srvs/srv/SetBool` |
| `data: true` | sensors on |
| `data: false` | sensors off |

```bash
ros2 service call /movebase3_drive_controller/ultrasonic_switch std_srvs/srv/SetBool "{data: true}"
```

Leave this true while anyone is driving. The user guide lists eight ultrasonics.

## State API

| What | Topic | Type | How |
| --- | --- | --- | --- |
| Body joints | `/joint_states` | `sensor_msgs/msg/JointState` | `ros2 topic echo /joint_states` |
| Right gripper | `/right_gripper/joint_states` | `sensor_msgs/msg/JointState` | `ros2 topic echo /right_gripper/joint_states` |
| Left gripper | `/left_gripper/joint_states` | `sensor_msgs/msg/JointState` | same |
| Front lidar | `/lidar_front/scan` | not stated; treat as `sensor_msgs/msg/LaserScan` until `ros2 topic info` says otherwise | `ros2 topic echo /lidar_front/scan` |
| Rear lidar | `/lidar_rear/scan` | same | `ros2 topic echo /lidar_rear/scan` |
| RGB-D color | `/camera/color/image_raw` | image, viewed in RViz | launch below |
| RGB-D depth | `/camera/depth/image_raw` | image, viewed in RViz | launch below |
| Stereo | not a ROS topic in the SDK | V4L2 device | `ffplay` below |

RGB-D (Orbbec Gemini 2). The SDK says to plug in a monitor first:

```bash
ssh unix_ai@192.168.15.XX
cd /home/unix_ai/orbbec_ws
source install/setup.bash
ros2 launch orbbec_camera gemini2.launch.py
```

Then, on a machine with a display:

```bash
rviz2
```

Add `/camera/color/image_raw` and `/camera/depth/image_raw`.

Stereo camera, also with a monitor:

```bash
v4l2-ctl --list-devices
ffplay -f v4l2 -video_size 2560x720 -framerate 30 -input_format yuyv422 -i /dev/video4
```

`/dev/video4` is the device in the example. Use the number `v4l2-ctl` prints.

A sample `lsusb` pasted in the SDK log includes Orbbec Gemini (`2bc5:0670`), an Orbbec bootloader, a “Pixe, Inc. XYZ 3d” device, C-Media USB audio, a CAN adapter, CH340 serial, and a Bluetooth radio. That same log is a **failed** mic check (no STMicroelectronics or Rockchip USB mic after 31 tries). Treat it as an example of the log format, then read the live log.

## Logs

```bash
cat /tmp/hwb_log.txt          # hardware-bridge warnings and errors
tail -f /tmp/hwb_log.txt      # follow them
cat ~/log/media_pkg/audio_srv_wait_ok.log
```

The sample bridge log is a fault: CAN send failures for ids 335, 336, 337, and a NaN lift position. If you see that on the live robot, stop commanding the lift and the base until CAN is healthy.

## Arm zero calibration

This rewrites the arm zero. Do it only when each joint is on the scale mark in section 1.5 of the SDK PDF. The diagrams did not survive text extraction; use the PDF figures, not memory.

1. Power the robot off so the arms are de-energized. Set every joint to the marked 0.
2. Power on. Wait for the startup voice prompt. Enable the head.
3. Press the physical emergency stop, then reset it. The SDK says this powers the arm and leaves it disabled.
4. Run:

```bash
cd unix_motor_tool
./cia402_calib_zero.sh
```

5. Power off, then power on again. The SDK says the process is finished only after that restart.

## First ten minutes on the robot

```bash
source /home/unix_ai/work/controller/install/setup.bash
echo "$ROS_DISTRO"
ros2 control list_hardware_components
ros2 control list_controllers
ros2 topic echo /joint_states --once
ros2 interface show unix_msgs/msg/GripperCommand
grep -R "dual_arm_forward\|joints:" -n /home/unix_ai/work/controller/share --include='*.yaml' | head
find /home/unix_ai -name '*.urdf' -o -name '*.xacro' -o -name '*.mjcf' -o -name 'scene.xml' | head
```

Copy any URDF, xacro, and controller YAML off the robot. The simulator README needs them. The SDK does not ship a model.

## What the SDK does not contain

- ROS distro, RMW, and domain id.
- A system launch file. Bringup is already installed and starts on boot.
- URDF, frames, and a TF tree. No `robot_state_publisher` is mentioned.
- Odometry, IMU topic, battery, or emergency-stop topic.
- Nav2, MoveIt, or an action API. Arm P2P is a topic, not a `FollowJointTrajectory` action client in the examples (the controller type is `JointTrajectoryController`, so the action may exist and is simply undocumented).
- Joint limits, except the lift range and the head angles in the user guide.
- The element order of the dual-arm `Float64MultiArray` and of the head array.
- A mapping for `joint_lf` / `joint_rf`.
- The `unix_msgs` `.msg` source. `ros2 interface show` is the authority.
- Camera intrinsics, lidar frame, or scan topic type (inferred as `LaserScan` above).

Check a guessed topic with `ros2 topic info <name> -v` before you write a node against it.
