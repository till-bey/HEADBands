# UnixAI Panther — ROS 2 Interface Reference

2026-11-14 · Online version: https://claude.ai/artifact/2me3m7vR4AkWz7zBqdQWRU

## Connecting from a laptop

The robot runs ROS 2 Humble with CycloneDDS on `ROS_DOMAIN_ID=15`; a Jazzy laptop on the Ethernet cable sees all topics once Cyclone is pinned to the cable interface.

| Device | Address | Notes |
| --- | --- | --- |
| Router ("arixo" web UI) | 192.168.100.254 | Web admin on 80/443, SSH open. Hands out 192.168.100.x by DHCP. |
| Jetson A (`tegra-ubuntu`, MAC 4c:bb:47:a4:e4:c3) | 192.168.100.3 | Runs the whole ROS 2 stack. Web file server on port 5021. |
| Jetson B (`tegra-ubuntu`, MAC 4c:bb:47:a8:e4:c3) | 192.168.100.4 | Same disk image and SSH host key as A. No ROS traffic seen. |
| Laptop | DHCP, 192.168.100.x | USB Ethernet `enx144fd7c5500d` |

SSH user is `unix_ai`; the password is on page 1 of the vendor SDK PDF (`UnixAI_Panther_SDK_English.pdf`). The vendor's message definitions live on the robot under `/home/unix_ai/work/controller` (`source /home/unix_ai/work/controller/install/setup.bash`).

Setup script (`~/unixai.sh`), sourced in every robot terminal:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=15
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="enx144fd7c5500d"/></Interfaces></General></Domain></CycloneDDS>'
timeout 5 ros2 daemon stop
```

Pitfalls seen:

- Without `CYCLONEDDS_URI`, Cyclone binds to Wi-Fi and only `/rosout` and `/parameter_events` appear.
- Fast DDS (the Jazzy default) does not discover the robot; install `ros-jazzy-rmw-cyclonedds-cpp`.
- Unplugging the cable can hang the ROS 2 daemon; `pkill -9 -f ros2cli.daemon` clears it.
- Humble/Jazzy mismatch: `ros2 node list` stays empty and `controller_manager_msgs` replies print garbled; topics and plain services work.
- In bridge mode the router drops its IPv4 address; its web UI is still reachable over IPv6 link-local at `/index.html`.

## Controllers

The controller manager (`/controller_manager`, 100 Hz) loads 15 controllers; types and joints below were read from its parameters on 2026-11-14.

| Controller | Type | Joints / interfaces | Command input |
| --- | --- | --- | --- |
| `movebase3_drive_controller` | `unix_swerve_drive_controller/UnixSwerveDriveController` | `movebase3/work_mode`, `linear_x`, `linear_y`, `angular_z` | `/cmd_vel` (Twist). Seen `active`. |
| `movebase_gpio_controller` | `gpio_controllers/GpioCommandController` | base GPIO | `/movebase_gpio_controller/commands` |
| `dual_arm_forward_position_controller` | `forward_command_controller/ForwardCommandController` | `joint_la0`–`la7`, `joint_ra0`–`ra7` (position) | `/dual_arm_forward_position_controller/commands` (Float64MultiArray, 16 values) |
| `left_arm_joint_trajectory_position_controller` | `joint_trajectory_controller/JointTrajectoryController` | `joint_la0`–`la7` (position) | action `follow_joint_trajectory`, topic `joint_trajectory` |
| `right_arm_joint_trajectory_position_controller` | `joint_trajectory_controller/JointTrajectoryController` | `joint_ra0`–`ra7` (position) | action `follow_joint_trajectory`, topic `joint_trajectory` |
| `mit_controller` | `mit_controller/MitController` | all 16 arm joints | `/mit_controller/commands` (`unix_msgs/MitCommandArray`) |
| `head_forward_position_controller` | `forward_command_controller/ForwardCommandController` | `joint_head0`, `joint_head1` (position) | `/head_forward_position_controller/commands` (Float64MultiArray `[head0, head1]`) |
| `left_unix_gripper3_controller` | `omni_picker_controller/OmniPickerController` | `joint_lf` (position) | `/left_gripper/gripper_command` (`unix_msgs/GripperCommand`) |
| `right_unix_gripper3_controller` | `omni_picker_controller/OmniPickerController` | `joint_rf` (position) | `/right_gripper/gripper_command` (`unix_msgs/GripperCommand`) |
| `lifting_controller` | `lifting_controller3/LiftingController3` | `body_joint` | `/lifting_controller/commands` (JointState) |
| `power_management_controller` | `power_management_controller/PowerManagementController` | – | services only |
| `joint_state_broadcaster` | `joint_state_broadcaster/JointStateBroadcaster` | 16 arm + 2 head joints | publishes `/joint_states` |
| `battery_state_broadcaster` | `battery_state_broadcaster/BatteryStateBroadcaster` | current, voltage, percentage, temperature, status, health | publishes `.../battery_state` |
| `imu_sensor_broadcaster` | `imu_sensor_broadcaster/IMUSensorBroadcaster` | IMU | publishes `.../imu` |
| `diagnostic_broadcaster` | `diagnostic_broadcaster/DiagnosticBroadcaster` | – | publishes `/diagnostics` |

Configured but not loaded: `lifting_forward_position_controller` (ForwardCommandController). Topics with no controller behind them: `/gripper_forward_position_controller/commands`, `/joint_command_controller/joint_command`.

The arm controllers share joints, so only one family can be active at a time. To move an arm by trajectory, deactivate `dual_arm_forward_position_controller` and activate that arm's trajectory controller with `/controller_manager/switch_controller`, then switch back (the PR 521 driver does exactly this).

Open question: which controllers are `active` right now. From a Jazzy laptop, `ros2 control list_controllers` prints only the first row because the Humble `ControllerState` message differs; run it on the robot over SSH.

## Topics

The robot exposes 181 topics on domain 15 (captured 2026-11-14 from 192.168.100.3); "cmd" rows are inputs you publish to, "state" rows are outputs to read. Lifecycle `transition_event` topics and the per-joint CANopen topics are summarised at the end.

### Mobile base

| Topic | Type | Use |
| --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | cmd: base velocity (`linear.x`, `linear.y`, `angular.z`) |
| `/cmd_vel_teleop` | `geometry_msgs/msg/Twist` | cmd: vendor teleop input |
| `/cmd_vel_nav` | `geometry_msgs/msg/Twist` | cmd: Nav2 output |
| `/cmd_vel_limit` | `geometry_msgs/msg/TwistStamped` | velocity limit |
| `/preempt_teleop` | `std_msgs/msg/Empty` | cmd: preempt teleop |
| `/odom` | `nav_msgs/msg/Odometry` | state |
| `/movebase_gpio_controller/commands` | `control_msgs/msg/DynamicInterfaceGroupValues` | cmd: base GPIO |
| `/movebase_gpio_controller/gpio_states` | `control_msgs/msg/DynamicInterfaceGroupValues` | state |

### Arms

| Topic | Type | Use |
| --- | --- | --- |
| `/dual_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | cmd: 16 positions, `la0`–`la7` then `ra0`–`ra7` |
| `/left_arm_joint_trajectory_position_controller/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` | cmd |
| `/right_arm_joint_trajectory_position_controller/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` | cmd |
| `/left_arm_joint_trajectory_position_controller/state`, `.../controller_state` | `control_msgs/msg/JointTrajectoryControllerState` | state |
| `/right_arm_joint_trajectory_position_controller/state`, `.../controller_state` | `control_msgs/msg/JointTrajectoryControllerState` | state |
| `/mit_controller/commands` | `unix_msgs/msg/MitCommandArray` | cmd: MIT impedance control (definition unknown) |
| `/joint_command_controller/joint_command` | `unix_msgs/msg/JointCommand` | cmd (definition unknown, no controller loaded) |
| `/left_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | cmd (no controller loaded) |
| `/right_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | cmd (no controller loaded) |
| `/dual_arm_effort_controller/commands` | `std_msgs/msg/Float64MultiArray` | cmd (no controller loaded) |

### Grippers, hand, lift, head

| Topic | Type | Use |
| --- | --- | --- |
| `/left_gripper/gripper_command` | `unix_msgs/msg/GripperCommand` | cmd |
| `/right_gripper/gripper_command` | `unix_msgs/msg/GripperCommand` | cmd |
| `/left_gripper/joint_states` | `sensor_msgs/msg/JointState` | state (`joint_lf`) |
| `/right_gripper/joint_states` | `sensor_msgs/msg/JointState` | state (`joint_rf`) |
| `/gripper_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | cmd (no subscriber) |
| `/dexhand_controller/dexhand_command` | `sensor_msgs/msg/JointState` | cmd: dexterous hand |
| `/dexhand_controller/joint_states` | `sensor_msgs/msg/JointState` | state |
| `/lifting_controller/commands` | `sensor_msgs/msg/JointState` | cmd: lift position/velocity |
| `/lifting_controller/joint_states` | `sensor_msgs/msg/JointState` | state (`body_joint`) |
| `/lifting_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | cmd (controller not loaded) |
| `/head_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | cmd: `[joint_head0, joint_head1]` |

### Robot state and sensors

| Topic | Type | Use |
| --- | --- | --- |
| `/joint_states` | `sensor_msgs/msg/JointState` | state: 16 arm + 2 head joints |
| `/dynamic_joint_states` | `control_msgs/msg/DynamicJointState` | state |
| `/robot_description` | `std_msgs/msg/String` | URDF |
| `/tf`, `/tf_static` | `tf2_msgs/msg/TFMessage` | transforms |
| `/battery_state_broadcaster/battery_state` | `sensor_msgs/msg/BatteryState` | state |
| `/imu_sensor_broadcaster/imu`, `/imu` | `sensor_msgs/msg/Imu` | state |
| `/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | state, includes the e-stop entry |
| `/scan` | `sensor_msgs/msg/LaserScan` | front lidar (frame `link_front_radar`) |
| `/camera/color/image_raw` | `sensor_msgs/msg/Image` | Orbbec colour, only while the camera driver runs |
| `/camera/depth/image_raw` | `sensor_msgs/msg/Image` | Orbbec depth, same |

### Navigation, mapping, docking

| Topic | Type | Use |
| --- | --- | --- |
| `/goal_pose` | `geometry_msgs/msg/PoseStamped` | cmd: Nav2 goal |
| `/initialpose` | `geometry_msgs/msg/PoseWithCovarianceStamped` | cmd: set localisation |
| `/pose` | `geometry_msgs/msg/PoseWithCovarianceStamped` | state: slam_toolbox pose |
| `/map`, `/map_metadata`, `/map_image` | `nav_msgs/msg/OccupancyGrid`, `nav_msgs/msg/MapMetaData`, `sensor_msgs/msg/CompressedImage` | state |
| `/plan`, `/plan_smoothed`, `/unsmoothed_plan`, `/local_plan`, `/received_global_plan`, `/transformed_global_plan` | `nav_msgs/msg/Path` | state |
| `/global_costmap/*`, `/local_costmap/*` | `nav_msgs/msg/OccupancyGrid`, `nav2_msgs/msg/Costmap`, `map_msgs/msg/OccupancyGridUpdate`, footprints, `nav2_msgs/msg/VoxelGrid` | state |
| `/speed_limit` | `nav2_msgs/msg/SpeedLimit` | cmd |
| `/evaluation` | `dwb_msgs/msg/LocalPlanEvaluation` | state |
| `/cost_cloud`, `/clustered_points`, `/laser_pattern` | `sensor_msgs/msg/PointCloud2` | state |
| `/dock_pose`, `/detected_dock_pose`, `/filtered_dock_pose`, `/staging_pose` | `geometry_msgs/msg/PoseStamped` | docking |
| `/docking_trajectory` | `nav_msgs/msg/Path` | docking |
| `/slam_toolbox/*` | `visualization_msgs` markers, `sensor_msgs/msg/LaserScan` | visualisation |
| `/marker` | `visualization_msgs/msg/MarkerArray` | visualisation |
| `/bond` | `bond/msg/Status` | Nav2 lifecycle bonds |

### Voice, apps, system

| Topic | Type | Use |
| --- | --- | --- |
| `/aiui_pkg/audio_topic` | `ros2_comm/msg/AudioTopic` | audio (vendor type) |
| `/dui_node/voice_txt` | `std_msgs/msg/String` | recognised speech |
| `/exercise/state`, `/perform/status` | `std_msgs/msg/String` | app status |
| `/qrcode_wifi/connection_status` | `qrcode_wifi_node/msg/ConnectionStatus` | Wi-Fi setup (vendor type) |
| `/qrcode_wifi/wifi_info` | `qrcode_wifi_node/msg/WiFiInfo` | Wi-Fi setup (vendor type) |
| `/events/write_split` | `rosbag2_interfaces/msg/WriteSplitEvent` | a rosbag recorder is running |

### CANopen joint drivers

Each arm joint has four topics: `/can0/joint_la0`…`joint_la7` (left arm) and `/can1/joint_ra0`…`joint_ra7` (right arm), each with `joint_states` (`sensor_msgs/msg/JointState`), `nmt_state` (`std_msgs/msg/String`), `rpdo` and `tpdo` (`canopen_interfaces/msg/COData`). Every controller also has a `transition_event` topic (`lifecycle_msgs/msg/TransitionEvent`).

## Custom message types (unix_msgs)

Only `GripperCommand` is known: PR [#520](https://github.com/4paradigm/phanthymotus-driver/pull/520) captured it with `ros2 interface show` on the robot, and PR [#521](https://github.com/4paradigm/phanthymotus-driver/pull/521) reuses it unchanged. `JointCommand` and `MitCommandArray` appear in neither PR.

| Type | Used on | Definition |
| --- | --- | --- |
| `unix_msgs/msg/GripperCommand` | `/left_gripper/gripper_command`, `/right_gripper/gripper_command` | `string mode` + `sensor_msgs/JointState command` |
| `unix_msgs/msg/JointCommand` | `/joint_command_controller/joint_command` | unknown: run `ros2 interface show` on the robot |
| `unix_msgs/msg/MitCommandArray` | `/mit_controller/commands` | unknown: run `ros2 interface show` on the robot |

`GripperCommand.msg`:

```
string mode
sensor_msgs/JointState command
```

`mode` takes `parallel`, `2-finger` or `centric`; PR 521 found no difference on this two-finger gripper and always sends `parallel`. `command.position[0]` runs 0 (open) to 1 (closed), matching the SDK's open/close examples; `command.name` is `['']`.

To publish it from the laptop, build the package in a workspace. The package and message names must match the robot's exactly. `CMakeLists.txt`:

```
cmake_minimum_required(VERSION 3.8)
project(unix_msgs)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GripperCommand.msg"
  DEPENDENCIES sensor_msgs
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
```

`package.xml`:

```xml
<?xml version="1.0"?>
<package format="3">
  <name>unix_msgs</name>
  <version>0.0.1</version>
  <description>UniX message types used by the Panther gripper topics.</description>
  <maintainer email="drivers@phanthy.com">Phanthy Motus</maintainer>
  <license>Proprietary</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <build_depend>rosidl_default_generators</build_depend>
  <depend>sensor_msgs</depend>
  <exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>
</package>
```

Build and use:

```bash
mkdir -p ~/unix_ws/src/unix_msgs/msg   # CMakeLists.txt and package.xml in unix_msgs/, GripperCommand.msg in msg/
cd ~/unix_ws && source ~/unixai.sh && colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source ~/unix_ws/install/setup.bash
```

`-DPython3_EXECUTABLE=/usr/bin/python3` avoids the `No module named 'em'` error when another Python is first on `PATH`.

## Services and actions

The robot offers 852 services; below are the ones that do something, without the per-node parameter and lifecycle services. Types with `unix_msgs`, `ros2_comm` or `qrcode_wifi_node` are vendor-only: their definitions are not public.

### Controllers and hardware

| Service | Type | Does |
| --- | --- | --- |
| `/controller_manager/list_controllers` | `controller_manager_msgs/srv/ListControllers` | read controller states |
| `/controller_manager/switch_controller` | `controller_manager_msgs/srv/SwitchController` | activate / deactivate controllers |
| `/controller_manager/load_controller`, `configure_controller`, `unload_controller` | `controller_manager_msgs/srv/*` | controller lifecycle |
| `/controller_manager/list_hardware_components`, `list_hardware_interfaces`, `list_controller_types` | `controller_manager_msgs/srv/*` | read |
| `/controller_manager/set_hardware_component_state`, `reload_controller_libraries` | `controller_manager_msgs/srv/*` | hardware lifecycle |
| `/movebase3_drive_controller/ultrasonic_switch` | `std_srvs/srv/SetBool` | ultrasonic sensors on/off (SDK 1.2.x) |
| `/movebase3_drive_controller/calib` | `std_srvs/srv/Trigger` | base calibration |
| `/power_management_controller/reboot` | `std_srvs/srv/SetBool` | reboot |
| `/left_arm_joint_trajectory_position_controller/query_state`, `/right_arm_...` | `control_msgs/srv/QueryTrajectoryState` | read |

### CANopen arm joints

Each joint (`/can0/joint_la0`…`la7`, `/can1/joint_ra0`…`ra7`) has the same 17 services: `init`, `enable`, `disable`, `halt`, `recover`, `nmt_start_node`, `nmt_reset_node`, mode switches (`position_mode`, `velocity_mode`, `torque_mode`, `interpolated_position_mode`, `cyclic_position_mode`, `cyclic_velocity_mode`, `cyclic_torque_mode`), all `std_srvs/srv/Trigger`; plus `sdo_read` (`canopen_interfaces/srv/CORead`), `sdo_write` (`COWrite`) and `target` (`COTargetDouble`). Bus masters: `/can0/master/sdo_read|sdo_write`, `/can1/master/...` (`COReadID`, `COWriteID`) and `/device_container_can0|can1/init_driver` (`canopen_interfaces/srv/CONode`). These bypass ros2_control; leave them to the vendor unless a joint needs `recover`.

### Mapping, localisation, landmarks

| Service | Type |
| --- | --- |
| `/unix_slam/get_map_list`, `load_map`, `save_map`, `rename_map`, `delete_map`, `enable_map`, `record_initial_pose` | `unix_msgs/srv/GetMapList`, `LoadMap`, `SaveMap`, `RenameMap`, `DeleteMap`, `EnableMap`, `RecordPose` |
| `/unix_slam/set_localization_mode` | `std_srvs/srv/SetBool` |
| `/unix_slam/reset` | `std_srvs/srv/Trigger` |
| `/landmark_manager/list_landmarks`, `record_landmark`, `delete_landmark` | `unix_msgs/srv/ListLandmarks`, `RecordLandmark`, `DeleteLandmark` |
| `/slam_toolbox/save_map`, `serialize_map`, `deserialize_map`, `dynamic_map`, `pause_new_measurements`, `manual_loop_closure`, `clear_changes`, `clear_localization_buffer`, `toggle_interactive_mode` | standard `slam_toolbox` services |
| `/lifecycle_manager_navigation/manage_nodes`, `is_active` | `nav2_msgs/srv/ManageLifecycleNodes`, `std_srvs/srv/Trigger` |
| `/docking_server/reload_database` | `opennav_docking_msgs/srv/ReloadDatabase` |

### Behaviours and apps

| Service | Type |
| --- | --- |
| `/task_trigger` | `unix_msgs/srv/TaskTrigger` |
| `/perform/trigger_ready`, `trigger_zero`, `trigger_calibration`, `trigger_finish`, `stop` | `std_srvs/srv/Trigger` |
| `/perform/love`, `sky`, `sugar_butterfly`, `sugar_gourd`, `sugar_horse`, `sugar_ice_cream`, `sugar_lollipop`, `sugar_rabbit`, `sugar_seal`, `sugar_tree` | `std_srvs/srv/Trigger` (pre-recorded performances) |
| `/exercise/trigger_start`, `trigger_pause`, `trigger_resume`, `trigger_finish` | `std_srvs/srv/Trigger` |
| `/handshake_skill/trigger_handshake`, `/trigger_hug` | `std_srvs/srv/Trigger` |
| `/aiui_pkg/audio_play`, `txt_service`, `web_txt_service` | `ros2_comm/srv/TxtSrv` |
| `/aiui_pkg/audio_service`, `txt_url_service`, `robot_service`, `robot_srv_str_service` | `ros2_comm/srv/AudioSrv`, `TxtUrlSrv`, `RobotSrv`, `RobotSrvStr` |
| `/manager/robot_service`, `/manager/robot_str_service` | `ros2_comm/srv/RobotSrv`, `RobotSrvStr` |
| `/tof_ir_blue/ir_server` | `ros2_comm/srv/SendIRCmd` |

### Actions

| Action | Type |
| --- | --- |
| `/left_arm_joint_trajectory_position_controller/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` |
| `/right_arm_joint_trajectory_position_controller/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` |
| `/play_trajectory` | `unix_msgs/action/TrajectoryPlayback` |
| `/navigate_to_pose`, `/navigate_through_poses`, `/follow_waypoints` | `nav2_msgs/action/*` |
| `/compute_path_to_pose`, `/compute_path_through_poses`, `/follow_path`, `/smooth_path` | `nav2_msgs/action/*` |
| `/spin`, `/backup`, `/drive_on_heading`, `/wait`, `/assisted_teleop` | `nav2_msgs/action/*` |
| `/dock_robot`, `/undock_robot` | `opennav_docking_msgs/action/DockRobot`, `UndockRobot` |
| `/behavior_server` | `btcpp_ros2_interfaces/action/ExecuteTree` |
| `/aiui_pkg/audio_server`, `/aiui_pkg/audio_server_tts` | `ros2_comm/action/AudioServer` |

## Command examples and limits

Every command below moves an 80 kg robot: read the current state first, keep the e-stop in reach, and use `--once` unless you mean to stream.

| Actuator | Range | Source |
| --- | --- | --- |
| Base `linear.x`, `linear.y` | ±0.2 m/s used by the PR 521 driver | driver choice, not a robot limit |
| Base `angular.z` | ±0.3 rad/s used by the PR 521 driver | driver choice |
| Lift position | SDK: 0.0–0.8 m. URDF limit 0.6; robot reads about 0.61; PR 521 caps at 0.65 | conflicting: stay under 0.6 until the vendor confirms |
| Lift velocity | −0.05 to 0.05 m/s | SDK |
| Gripper `position` | 0 = open, 1 = closed | SDK example, PR 521 |
| `joint_head0` | −0.348 to 1.5 rad (pitch) | URDF via PR 521 |
| `joint_head1` | −1.57 to 1.57 rad (yaw) | URDF via PR 521 |
| Arm joints | per joint from `/robot_description`, e.g. `joint_la0` −1.046 to 0.697, `joint_ra7` −1.57 to 1.57 rad | URDF via PR 521 |

The SDK's own lift example sends 0.75, which is above the URDF limit.

**Base, keyboard teleop** (`sudo apt install ros-jazzy-teleop-twist-keyboard` first; press `z` a few times to slow down, `k` to stop):

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

**Base, rotate in place (sent at 10 Hz until Ctrl+C):**

```bash
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0}, angular: {z: 0.2}}"
```

**Gripper** (needs the `unix_msgs` package from the section above):

```bash
ros2 topic pub --once /left_gripper/gripper_command unix_msgs/msg/GripperCommand "{mode: 'parallel', command: {name: [''], position: [0.0]}}"   # open
ros2 topic pub --once /left_gripper/gripper_command unix_msgs/msg/GripperCommand "{mode: 'parallel', command: {name: [''], position: [1.0]}}"   # close
```

**Lift:**

```bash
ros2 topic echo --once /lifting_controller/joint_states
ros2 topic pub --once /lifting_controller/commands sensor_msgs/msg/JointState "{name: [''], position: [0.5]}"
ros2 topic pub --once /lifting_controller/commands sensor_msgs/msg/JointState "{name: [''], velocity: [0.02]}"
```

**Head** (`[joint_head0, joint_head1]`):

```bash
ros2 topic pub --once /head_forward_position_controller/commands std_msgs/msg/Float64MultiArray "data: [0.1, 0.1]"
```

**Arm by trajectory** (switch controllers first, switch back after):

```bash
ros2 control switch_controllers --activate left_arm_joint_trajectory_position_controller right_arm_joint_trajectory_position_controller --deactivate dual_arm_forward_position_controller
ros2 topic pub --once /left_arm_joint_trajectory_position_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{joint_names: ['joint_la5'], points: [{positions: [0.2], time_from_start: {sec: 6}}]}"
ros2 control switch_controllers --activate dual_arm_forward_position_controller --deactivate left_arm_joint_trajectory_position_controller right_arm_joint_trajectory_position_controller
```

`ros2 control switch_controllers` from a Jazzy laptop may fail on the Humble message mismatch; run it on the robot over SSH if it does.

**Ultrasonic sensors:**

```bash
ros2 service call /movebase3_drive_controller/ultrasonic_switch std_srvs/srv/SetBool "{data: true}"
```

## Sources

- Vendor SDK: `UnixAI_Panther_SDK_English.pdf` (16 pages, local copy)
- [phanthymotus-driver PR #520](https://github.com/4paradigm/phanthymotus-driver/pull/520): Panther driver, `GripperCommand.msg`
- [phanthymotus-driver PR #521](https://github.com/4paradigm/phanthymotus-driver/pull/521): Panther driver with ranges, controller switching, sensor notes
- Live robot: `ros2 topic list -t`, `ros2 service list -t`, `ros2 action list -t` and controller-manager parameters, 2026-11-14
