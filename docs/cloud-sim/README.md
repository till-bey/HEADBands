# GCP sandbox and a Panther ROS 2 simulator

The hackathon Google Cloud project, what to run on it, and a MuJoCo simulator that speaks the same topics as the robot. Account steps come from Google’s sandbox guide and the credential mail (14 Nov 2026). The simulator design is ours: the UniX AI PDFs have no sim, no URDF, and no MJCF.

## Accounts

Three users share one project. Billing is already enabled. Log in at [console.cloud.google.com](https://console.cloud.google.com) in a fresh Chrome profile or an incognito window, then select the project. There should be only one.

| User | Project |
| --- | --- |
| `devstar6061@gcplab.me` | `robotcarnival26brn-606` |
| `devstar6062@gcplab.me` | `robotcarnival26brn-606` |
| `devstar6063@gcplab.me` | `robotcarnival26brn-606` |

Passwords are only in `docs/Gmail - Hackathon Google cloud sendbox.PDF` on a machine that already has that file. They are intentionally not copied into this README. That PDF is gitignored. Do not paste passwords, AI Studio keys, or service-account JSON into the repo, into screenshots, or into a chat that gets committed.

The same login works for AI Studio and for Antigravity. In Antigravity (2.0 and the `agy` CLI) set the project id to `robotcarnival26brn-606`. If the plan line says “Starter Quota”, log out and sign in again with that project id. The sandbox guide says the organization plan is what carries the Agent Platform quota.

## What you can use

| Need | Use |
| --- | --- |
| Shell in the browser, git, Python, `agy` | Cloud Shell (terminal icon, top right of the console) |
| Editor in the browser | [ide.cloud.google.com](https://ide.cloud.google.com), or Open Editor inside Cloud Shell |
| Agent coding | Antigravity 2.0, the Antigravity IDE, or `agy` in Cloud Shell. Install from [antigravity.google/download](https://antigravity.google/download). |
| Gemini and other models | AI Studio ([ai.studio](https://ai.studio), Get API key, import this project if the key is not already listed) and Agent Platform Model Garden |
| Host a small web app | Cloud Run |
| Kubernetes | GKE Autopilot |
| Postgres | Cloud SQL or AlloyDB |
| A real machine for sim and builds | **Compute Engine.** This is the ROS / MuJoCo host. |
| Files | Cloud Storage |
| Queries / dashboards | BigQuery, Looker |
| App backend | Firebase |
| Notes between the three accounts | Gmail, Docs, Chat, Drive, Notebook on `@gcplab.me` |

Cloud Shell is the admin shell. It is a small ephemeral VM with a few gigabytes of home disk and no GPU. Do not install the simulator there.

## Rules that will get the project shut off

From the sandbox guide:

- The accounts, the project, and the docs are deleted at the end of the event. Push code to a remote you control, and download anything you need, before that.
- Do not abuse quota. One simulator VM, stopped when nobody is in it.
- Do not leak API keys.
- Do not leave ports open. ROS 2 DDS (UDP around 7400–7500, plus discovery) stays off the public internet. SSH goes through IAP. A Foxglove or VNC port binds to localhost and is reached with an SSH tunnel.

## Simulator: what to build

Goal: run our ROS 2 packages against a fake Panther, then move the same package onto the Jetson without renaming topics.

The robot’s boot stack is `ros2_control` plus a handful of vendor controllers (`unix_swerve_drive_controller`, `lifting_controller3`, `omni_picker_controller`, CANopen arms). Those plugins talk to CAN. They will not run in the cloud. The **topics** will.

Two layers, in this order:

1. **Topic shim (build this first).** A MuJoCo model plus one Python node. The node subscribes to the vendor command topics and publishes the vendor state topics. Physics can be rough. Packages can be tested the same day.
2. **`mujoco_ros2_control` (build this once a URDF exists).** The upstream plugin `mujoco_ros2_control/MujocoSystemInterface` loads an MJCF and exposes position, velocity, and effort interfaces to ordinary `ros2_control` controllers. Use it when you want controller YAML, `JointTrajectory`, and sensors (cameras, lidar, IMU, force/torque) instead of a hand-written integrator. Docs: [mujoco_ros2_control](https://control.ros.org/jazzy/doc/mujoco_ros2_control/doc/index.html), source [ros-controls/mujoco_ros2_control](https://github.com/ros-controls/mujoco_ros2_control/). As of November 2026 the apt packages are documented for Jazzy; Humble is supported from source.

Do not point the cloud VM at the live robot. DDS across the public internet is a good way to move the real arms by accident. Develop in sim, `rsync` or git-pull the package onto the Jetson, and run it there on the lab LAN.

```text
laptop  --IAP SSH-->  panther-sim VM
                         ROS 2 (same distro as the Jetson)
                         MuJoCo
                         panther_sim_api     <- same topic names as the SDK
                         your packages

later:  git pull on the Jetson, source the vendor overlay, run against real controllers
```

### Topics the shim must speak

Match [the ROS 2 README](../ros2/README.md). Minimum set:

| Direction | Name | Type |
| --- | --- | --- |
| In | `/dual_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` |
| In | `/left_arm_joint_trajectory_position_controller/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` |
| In | `/right_arm_joint_trajectory_position_controller/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` |
| In | `/cmd_vel` | `geometry_msgs/msg/Twist` |
| In | `/left_gripper/gripper_command`, `/right_gripper/gripper_command` | `unix_msgs/msg/GripperCommand` |
| In | `/lifting_controller/commands` | `sensor_msgs/msg/JointState` |
| In | `/head_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` |
| Out | `/joint_states` | `sensor_msgs/msg/JointState` |
| Out | `/left_gripper/joint_states`, `/right_gripper/joint_states` | `sensor_msgs/msg/JointState` |
| Out | `/lidar_front/scan`, `/lidar_rear/scan` | `sensor_msgs/msg/LaserScan` |
| Out | `/camera/color/image_raw`, `/camera/depth/image_raw` | `sensor_msgs/msg/Image` |

Vendor `unix_msgs` in the sim workspace. On the robot, `ros2 interface show unix_msgs/msg/GripperCommand` is the source of truth. The SDK fills it as `mode: parallel` plus a nested `command` with `name` and `position` (`0.0` open, `1.0` closed).

Publish `/joint_states` with at least: `joint_la0`–`joint_la7`, `joint_ra0`–`joint_ra7`, `joint_head0`, `joint_head1`, `left_wheel_joint`, `right_wheel_joint`. Set `use_sim_time` true on every node, and run a `/clock` publisher.

The dual-arm float array’s joint order is not in the SDK. Pick an order, document it in the shim, and change it when someone pastes the controller YAML from `/home/unix_ai/work/controller`. Until then, test packages that use the trajectory topics (explicit `joint_names`) rather than the float array.

### Body to model, using only numbers we have

Pull a URDF off the Jetson if `find` turns one up. Until then, an MJCF with these joints is enough to stand packages up. Lengths are from the user guide, not from a measured kinematic chain. Arm joint limits are unknown; the SDK’s only fully specified arm pose is the left-arm sample `[0.0, 0.19, 0.37, -0.93, -1.0, 1.779, 0.107, -0.204]`.

| MJCF joint | Motion | Documented range |
| --- | --- | --- |
| base free joint, planar | integrate `/cmd_vel` (`x`, `y`, yaw) | speed cap 2 m/s; track 320 mm; wheelbase 420 mm |
| `body_joint` slide | torso height | 0.0–0.8 m, speed cap 0.05 m/s. Body motion space in the guide is 632–1432 mm. |
| `joint_la0`–`joint_la7` hinge | left arm | limits unknown; reach 982 mm; arm mass about 7 kg; end load 6 kg |
| `joint_ra0`–`joint_ra7` hinge | right arm | same |
| gripper slide, left and right | parallel fingers | command 0 open, 1 closed |
| `joint_head0`, `joint_head1` hinge | yaw and pitch, assignment unconfirmed | yaw ±90°, pitch +20° / −86° |

Model the base as one free joint. Simulating four swerve modules buys nothing until a package is closing wheel-level loops, and the real wheel controller is closed inside `UnixSwerveDriveController`.

`mujoco_ros2_control` can own this MJCF later. Its hardware tag points at the scene XML, and the control node has to be the one shipped in that package (not the stock `controller_manager` node). Headless mode is a parameter (`headless`, default false) plus `headless:=true` on the demos. A cloud VM has no display, so start headless.

## Create the VM

One shared machine named `panther-sim`. MuJoCo for a single robot is a CPU workload. Start with 8 vCPU and 32 GB. Add a GPU only if you are running a vision model or need rendered cameras at high rate.

Pick a zone where the machine type exists. `europe-west4-a` (Eemshaven) is a reasonable default from here. If the API returns a zone error, try `europe-west1-b` or `europe-west3-a`.

From Cloud Shell, or from a laptop already logged in with `gcloud`:

```bash
gcloud config set project robotcarnival26brn-606
gcloud services enable compute.googleapis.com iap.googleapis.com

gcloud compute instances create panther-sim \
  --zone=europe-west4-a \
  --machine-type=e2-standard-8 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=MIGRATE

gcloud compute firewall-rules create allow-ssh-from-iap \
  --direction=INGRESS \
  --action=ALLOW \
  --rules=tcp:22 \
  --source-ranges=35.235.240.0/20 \
  --target-tags=panther-sim

gcloud compute instances add-tags panther-sim --zone=europe-west4-a --tags=panther-sim

gcloud compute ssh panther-sim --zone=europe-west4-a --tunnel-through-iap
```

Do not add a rule for `0.0.0.0/0`. The default VPC already drops unsolicited ingress; leave it that way.

The robot runs Humble, so the VM is Ubuntu 22.04. The OS and the ROS distro have to match the robot.

Optional GPU, only after CPU sim works and only if the project has GPU quota (an L4 is `g2-standard-8`, accelerator `nvidia-l4`, maintenance `TERMINATE`):

```bash
gcloud compute instances create panther-sim-gpu \
  --zone=europe-west4-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=150GB \
  --boot-disk-type=pd-balanced
```

Stop the machine when you leave. Stopped disks still cost a little; running e2-standard-8 costs much more, and the sandbox can be revoked for waste.

```bash
gcloud compute instances stop panther-sim --zone=europe-west4-a
gcloud compute instances start panther-sim --zone=europe-west4-a
```

## Software on the VM

Humble on Ubuntu 22.04:

```bash
sudo apt-get update
sudo apt-get install -y curl gnupg software-properties-common build-essential python3-pip git

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt-get update
sudo apt-get install -y ros-humble-desktop ros-dev-tools \
  ros-humble-rmw-cyclonedds-cpp \
  ros-humble-foxglove-bridge \
  python3-colcon-common-extensions

grep -q ROS_DISTRO ~/.bashrc || cat >> ~/.bashrc <<'EOF'
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
EOF
source ~/.bashrc
```

Use `ROS_DOMAIN_ID=42` on the VM only. Keep the robot on whatever domain it already uses, so a laptop that joins the lab Wi-Fi cannot accidentally share a graph with both.

### MuJoCo via ros2_control

Install from apt if the Humble packages exist, otherwise build the repo (it vendors MuJoCo through `mujoco_vendor`):

```bash
apt-cache search mujoco-ros2-control

mkdir -p ~/mujoco_ws/src && cd ~/mujoco_ws/src
git clone https://github.com/ros-controls/mujoco_ros2_control.git
git clone https://github.com/pal-robotics/mujoco_vendor.git
cd ~/mujoco_ws
rosdep init || true
rosdep update
rosdep install -r --from-paths src --ignore-src --rosdistro "$ROS_DISTRO" -y
colcon build
source install/setup.bash
```

Smoke test, headless, before any Panther model exists:

```bash
ros2 launch mujoco_ros2_control_demos demo.launch.py headless:=true
```

The demo names differ by version (`demo.launch.py` or numbered tutorials such as `01_basic_robot.launch.py`). `ros2 pkg prefix mujoco_ros2_control_demos` and the package’s launch directory are the list.

### Workspace layout to create

```text
~/panther_sim_ws/src/
  unix_msgs/                  GripperCommand.msg copied from `ros2 interface show`
  panther_description/        MJCF scene + a URDF once the Jetson has one
  panther_sim_api/            topic shim
  <your packages>
```

```bash
cd ~/panther_sim_ws
colcon build
source install/setup.bash
```

A package is ready to leave the VM when it depends only on the topics in the table above and on `unix_msgs`, and when it respects `use_sim_time`. On the robot, source `/home/unix_ai/work/controller/install/setup.bash` before launching it, and do not start a second `ros2_control` node. The controllers are already active.

## See the sim

The VM has no monitor. Two ways to look:

- **Foxglove.** Run `ros2 launch foxglove_bridge foxglove_bridge_launch.xml` bound to localhost. From the laptop: `gcloud compute ssh panther-sim --zone=europe-west4-a --tunnel-through-iap -- -L 8765:localhost:8765`, then open Foxglove on `ws://localhost:8765`.
- **MuJoCo window.** Set `headless` false and use a virtual framebuffer (`Xvfb`) plus noVNC, still tunneled over SSH. Headless plus Foxglove is enough for topic-level tests.

## What to copy off the Jetson into this VM

```bash
# on the robot
find /home/unix_ai -name '*.urdf' -o -name '*.xacro' -o -name '*.yaml' \
  | grep -E 'controller|description|urdf|openarm'
tar -C /home/unix_ai -czf /tmp/panther_model.tgz work/controller/share
```

Then `gcloud compute scp` that archive to `panther-sim`. A real URDF replaces the approximate MJCF. Joint limits in that URDF override every guess in this README.

## Save work before the sandbox dies

The project is removed when the event ends.

- Push git history to a remote outside `gcplab.me`.
- Download bags, MJCF, and notes.
- AI Studio keys die with the project. Do not ship a demo that calls them from a laptop you still need next week.
