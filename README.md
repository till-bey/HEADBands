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
| `docs/` | Guides and vendor PDFs (below) |

## Quick start

Robot, on the Ethernet cable:

```bash
docker/run.sh                       # Humble shell, domain 15, Cyclone pinned to the cable
colcon build && source install/setup.bash
ros2 run panther_examples joint_check          # plan only; add --yes to move each joint a step and back
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

## Guides

| Guide | What it covers |
| --- | --- |
| [Operating the robot](docs/robot-operation/README.md) | Charging, power, the rear panel, the lab Wi-Fi, Quest 3s teleop |
| [ROS 2 stack and API](docs/ros2/README.md) | Controllers, topics, services, and what the SDK leaves out |
| [ROS 2 interface reference](docs/ros2/UnixAI_Panther_ROS2_Interface_Reference.md) | Every topic, service and limit read from the live robot |
| [Cloud sandbox and simulator](docs/cloud-sim/README.md) | GCP project `robotcarnival26brn-606`, and a MuJoCo sim that matches the robot’s topics |

Source PDFs are in `docs/`. The mail export that contains sandbox passwords is local only and is gitignored.
