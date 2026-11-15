# Robotics Safety Architecture (Gemini Robotics ER 2 + UniX AI Panther ROS 2)

## Core Principle

**Neither Gemini Flash nor Gemini Robotics ER 2 ever controls motors, joint controllers, or raw ROS 2 topics.**

All robot-related reasoning and on-demand camera captures flow through two independent deterministic safety barriers before reaching the **UniX AI Panther** hardware (`ROS_DOMAIN_ID=15`, `rmw_cyclonedds_cpp`):

```text
Gemini 3 Flash (Voice + Translation + Camera Intent) -> Gemini Robotics ER 2 (global preview endpoint)
        │  Structured JSON proposal only + on-demand camera trigger request
        ▼
[Barrier 1] Cloud ActionSafetyValidator + Human Confirmation Queue (backend/app/robotics/)
        │  Persistent authenticated outbound WebSocket (/ws/robot-bridge)
        ▼
[Barrier 2] Local Panther ROS 2 Safety Supervisor, Camera Manager & 10 Hz Watchdog
        │  (robot_bridge/ros2_ws/src/swiss_robot_bridge/)
        ▼
UniX AI Panther ROS 2 Controllers (CycloneDDS, ROS_DOMAIN_ID=15)
```

---

## Real UniX AI Panther ROS 2 Topic & Service Mapping

| Direction | ROS 2 Topic / Service | Message / Service Type | Purpose & Local Safety Bounds |
|---|---|---|---|
| Publish | `/cmd_vel` | `geometry_msgs/msg/Twist` | Holonomic 4WS base velocity (`linear.x`, `linear.y` clamped to `<= 0.5 m/s`, `angular.z <= 0.4 rad/s`), auto-zeroed when target duration completes or watchdog/LiDAR triggers |
| Publish | `/preempt_teleop` | `std_msgs/msg/Empty` | Immediate teleop preemption emitted alongside zero `Twist()` on `stop` / E-Stop |
| Publish | `/head_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | 2-DOF head pan/tilt `[joint_head0, joint_head1]` clamped to PR #521 safe bounds (`[-0.34, +0.34]` pitch, `[-1.20, +1.20]` yaw) |
| Publish | `/lifting_controller/commands` | `sensor_msgs/msg/JointState` | Torso lift height clamped strictly to URDF `[0.0, 0.60] m` at `<= 0.05 m/s` |
| Publish | `/left_gripper/gripper_command`, `/right_gripper/gripper_command` | `unix_msgs/msg/GripperCommand` | Parallel gripper actuation (`mode="parallel"`, `0.0` open on `place_object`, `1.0` closed on `pick_object`, `velocity=0.1`) |
| Service | `/movebase3_drive_controller/ultrasonic_switch` | `std_srvs/srv/SetBool` | Ensures all 8 ultrasonic obstacle sensors remain active (`data: true`) before base motion |
| Subscribe | `/battery_state_broadcaster/battery_state` | `sensor_msgs/msg/BatteryState` | Live battery telemetry; blocks motion below `10%` |
| Subscribe | `/odom`, `/pose` | `nav_msgs/msg/Odometry`, `PoseWithCovarianceStamped` | Live `(x, y, z)` pose feedback |
| Subscribe | `/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | Hardware / wireless E-Stop detection — immediately latches local E-Stop |
| Subscribe | `/scan` (sim: `/lidar_front/scan`) | `sensor_msgs/msg/LaserScan` | Front radar clearance check (`[-35°, +35°]`); blocks/halts forward `/cmd_vel` if obstacle `< 0.35 m` |
| Subscribe | `/joint_states`, `/lifting_controller/joint_states`, `/left_gripper/joint_states`, `/right_gripper/joint_states` | `sensor_msgs/msg/JointState` | Live torso lift, head, and gripper joint feedback |
| Subscribe (On-Demand) | `/camera/color/image_raw`, `/camera/color/image_raw/compressed`, `/camera/depth/image_raw` | `sensor_msgs/msg/Image`, `CompressedImage` | Lazy on-demand RGB-D capture from Orbbec Gemini 2; auto-launches `ros2 launch orbbec_camera gemini2.launch.py` if no publisher is active and shuts down after 90s idle |

---

## Enforcement of the 10 Mandatory Safety Rules

| # | Safety Rule | Cloud Enforcement (`action_validator.py` / `robot_gateway.py`) | Local Robot Enforcement (`local_safety.py` / `bridge_node.py`) |
|---|---|---|---|
| 1 | Never execute arbitrary LLM code | Regex & schema rejection of shell/Python tokens; no `eval`/`exec` anywhere | Independent schema validation; zero dynamic code evaluation |
| 2 | Never allow arbitrary ROS topic publishing | `ConfigDict(extra="forbid")` + explicit rejection of `/cmd_vel`, `/lifting_controller`, `/left_gripper`, etc. | Publishes exclusively to fixed Panther topics listed above |
| 3 | Allow only explicitly defined actions | `AllowedActionType` enum (`move_to`, `rotate`, `look_at`, `inspect_object`, `pick_object`, `place_object`, `return_home`, `stop`) | `ALLOWED_LOCAL_ACTIONS` allowlist |
| 4 | Validate all action parameters | Finite float checks (`math.isfinite`), `z` bounds, yaw `[-pi, pi]` | Re-validates all numeric parameters locally before computing `PantherActuationPlan` |
| 5 | Enforce speed, position & movement limits | Rejects `speed > ROBOT_MAX_LINEAR_SPEED_MPS` (`0.5 m/s`) or radius `> ROBOT_MAX_WORKSPACE_RADIUS_M` | Clamps & rejects out-of-envelope targets, speeds, lift (`[0, 0.6]m`), head, and gripper bounds |
| 6 | Require human confirmation for dangerous actions | Forces `requires_confirmation=True` on `move_to`, `rotate`, `pick_object`, `place_object`, `return_home` until `POST /api/v1/robotics/confirm/{id}` | Rejects any motion/gripper command where `confirmed_by_human != True` |
| 7 | Implement timeouts and command expiration | Stamps `expires_at = now + ROBOT_COMMAND_TTL_SECONDS` (`30s`) | Rejects any command where `now_utc >= expires_at` and auto-zeroes `/cmd_vel` when target motion duration finishes |
| 8 | Implement a watchdog | Rejects dispatch when bridge heartbeat age exceeds `ROBOT_WATCHDOG_TIMEOUT_SECONDS` | 10 Hz local loop zeroes `/cmd_vel` whenever cloud heartbeat exceeds `2.5s` or front LiDAR detects an obstacle `< 0.35m` |
| 9 | Cloud-independent Emergency Stop | `POST /api/v1/robotics/estop` & WS `emergency_stop` purge queue and latch `ESTOP` | `trigger_local_estop()` + `/diagnostics` E-Stop monitor work completely offline with zero internet |
| 10 | Reject commands in unsafe states | Rejects commands when state is `estop`, `fault`, `low_battery`, `unsafe`, or `disconnected` | Rejects motion when `local_estop_latched`, `local_fault_active`, or `battery < 10%` |

---

## Network Security

- The ROS 2 CycloneDDS network (`ROS_DOMAIN_ID=15`, `192.168.100.0/24`) is **never** exposed to the public internet.
- `robot_bridge` initiates an **outbound** TLS WebSocket connection (`wss://<backend>/ws/robot-bridge`) authenticated with `Authorization: Bearer <ROBOT_BRIDGE_SHARED_TOKEN>` stored in Google Secret Manager.
