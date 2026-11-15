# frisbee_interception

Lightweight (CPU, OpenCV, no neural networks) frisbee detection, tracking and interception prediction for ROS 2 Humble.
Prediction only by default: it never moves the robot.

Nodes: `phone_camera`, `frisbee_detector`, `frisbee_tracker`, `frisbee_interceptor` · messages in `frisbee_msgs`.

```bash
colcon build --packages-select frisbee_msgs frisbee_interception && source install/setup.bash
ros2 launch frisbee_interception frisbee.launch.py source:=<phone stream URL or video file> rviz:=true
```

Architecture, topics, calibration (camera, camera→robot TF, catching plane), safety/arming, controller-integration
proposal, results and CI: **[docs/frisbee_interception.md](../../docs/frisbee_interception.md)**.
