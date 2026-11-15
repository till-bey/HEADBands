# HEADBands

Teaching the UniX AI Panther to throw a frisbee. Everything ROS is Humble, like the robot.

| Folder | What is in it |
| --- | --- |
| [`docker/`](docker/) | Humble container: `run.sh` for the robot (Ethernet), `run.sh --sim` for the mock sim |
| [`src/panther_examples`](src/panther_examples/README.md) | Robot scripts: `commands`, `teach` (record and play back by hand), `cartesian_follow` (PyKDL IK), `joint_check`; shared names and limits in `robot.py` |
| [`src/panther_sim`](src/panther_sim/README.md) | Mock sim with the robot's controller, hardware and topic names, so the scripts above run on it unchanged |
| `src/unix_msgs` | Vendor `GripperCommand` message |
| `motion/` | Offline tools and data: `throws/` (phone recordings, curate, view), `to_path/` (recording to Cartesian path), `retarget/` (human arm to joint angles), `example_path.csv` |
| [`perception/`](perception/README.md) | Frisbee detection: OpenCV detector, Streamlit site, YOLO training |
| [`interface/`](interface/README.md) | Voice & robotics web UI on GCP (Gemini, ER 2, gestures) and its robot bridge (`docker/bridge.sh`) |
| `docs/` | Guides and vendor PDFs (below) |

## Quick start

GUI apps in the container (RViz, Gazebo) need X access. Run once per login on the host:

```bash
xhost +local:                       # the trailing colon matters
```

Robot, on the Ethernet cable:

```bash
docker/run.sh                       # Humble shell, domain 15, Cyclone pinned to the cable
colcon build && source install/setup.bash
ros2 run panther_examples joint_check          # plan only; add --yes to move each joint a step and back
ros2 launch panther_sim robot_rviz.launch.py   # RViz of the robot's arms and head, stand-in shapes, lift and wheels fixed

# rqt on the arms (with robot_rviz running: the robot's URDF has arm velocity 0, which crashes rqt)
ros2 control switch_controllers --activate left_arm_joint_trajectory_position_controller right_arm_joint_trajectory_position_controller --deactivate dual_arm_forward_position_controller
ros2 run rqt_joint_trajectory_controller rqt_joint_trajectory_controller --ros-args -r /robot_description:=/panther_standin/robot_description
ros2 control switch_controllers --activate dual_arm_forward_position_controller --deactivate left_arm_joint_trajectory_position_controller right_arm_joint_trajectory_position_controller   # when done
```

For the real meshes instead, copy them from the robot once (`docker/run.sh` mounts `~/mh3_description` where the robot's URDF expects it), then use plain RViz:

```bash
scp -r unix_ai@192.168.100.3:/home/unix_ai/unixai_ws/install/mh3_description/share/mh3_description ~/mh3_description
rviz2 -d src/panther_sim/config/panther.rviz -f body_link   # in the robot terminal
```

Mock sim, no robot (domain 77, localhost only, so it can never command the robot):

```bash
docker/run.sh --sim                 # terminal 1
colcon build && source install/setup.bash
ros2 launch panther_sim mock_sim.launch.py

docker/run.sh --sim                 # terminal 2
source install/setup.bash
ros2 run panther_examples joint_check --yes
```

Perception:

```bash
cd perception && uv run streamlit run website/app.py
```

### Connect the container to the cloud interface (GCP)

The voice & robotics web UI runs on Cloud Run ([`interface/`](interface/README.md),
https://swiss-voice-backend-1060412536965.europe-west6.run.app). `docker/bridge.sh` starts its robot bridge in the
Humble container: it dials out over a WebSocket (no port opened), receives only human-confirmed commands, re-checks
them locally, and drives the robot's own controllers. Rebuild the image once (`--build`) for the bridge's packages.

```bash
# on the host, once per terminal (needs gcloud logged in to the sandbox project)
export ROBOT_BRIDGE_SHARED_TOKEN="$(gcloud secrets versions access latest --secret=robot-bridge-shared-token --project=robotcarnival26brn-606)"

docker/run.sh --build docker/bridge.sh    # robot, on the Ethernet cable (domain 15)
docker/run.sh --sim docker/bridge.sh      # or: the mock sim on this machine (domain 77), with mock_sim running
```

- Base, lift and grippers use their usual topics. Throws and gestures switch the arm trajectory controllers in for
  `dual_arm_forward_position_controller` (as `master_gui` does) and leave them in; don't run `master_gui` at the same
  time. The head is streamed to `head_forward_position_controller` at up to 0.6 rad/s.
- The head stays in the conservative range (pitch ±0.34, yaw ±1.2 rad) until `PANTHER_HEAD_FULL_RANGE=1` is exported
  before `docker/run.sh`. Set it only after a small yaw on the robot moved `joint_head1` in `/joint_states`.
- E-stop in the web UI (or a lost connection) stops the base and freezes any arm motion where it is.

## Guides

| Guide | What it covers |
| --- | --- |
| [Operating the robot](docs/robot-operation/README.md) | Charging, power, the rear panel, the lab Wi-Fi, Quest 3s teleop |
| [ROS 2 stack and API](docs/ros2/README.md) | Controllers, topics, services, and what the SDK leaves out |
| [ROS 2 interface reference](docs/ros2/UnixAI_Panther_ROS2_Interface_Reference.md) | Every topic, service and limit read from the live robot |
| [Cloud sandbox and simulator](docs/cloud-sim/README.md) | GCP project `robotcarnival26brn-606`, and a MuJoCo sim that matches the robot’s topics |

Source PDFs are in `docs/`. The mail export that contains sandbox passwords is local only and is gitignored.
