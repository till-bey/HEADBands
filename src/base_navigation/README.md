# Frisbee navigation

Both the CLI and dashboard run `main.py`'s `control_loop()`: read the latest target, reject expired results,
compute P control, and call `robot.set_velocity()`. Responsibilities are split
between:

- `controller.py`: ROS-independent P control.
- `perception/camera/`: shared stream reader, `FrisbeeCamera` API, navigation camera
  setup and worker, and webcam tool (also used by the website).
- `config.py`: shared command-line options and controller configuration.
- `navigation.py`: `BaseController` and the `RobotNavigation` adapter for ROS
  setup, timers, velocity commands, subscriber checks, and shutdown. Its `wait()`
  method processes ROS events between iterations of the main control loop.
- `dashboard.py`: browser dashboard entry point, camera processing, and telemetry.
- `dashboard_server.py` and `dashboard.html`: local HTTP server and browser interface.
- `web_control.py`: browser motion requests and heartbeat handling, using
  `main.control_loop()` for navigation and `navigation.py` for all ROS operations.
  Each loop iteration services pending lift/stop requests and reads the applied
  controller settings before computing motion.

Dependencies: ROS 2 supplies `rclpy`, `geometry_msgs`, and `sensor_msgs`; the camera/detector needs
NumPy and OpenCV. `controller.py` itself uses only Python's standard library.
Streamlit and Pillow are needed by the perception dashboard, not navigation.

From the repository root, create a virtual environment using the system Python
matching your ROS installation. `--system-site-packages` exposes ROS's system
Python dependencies. For ROS 2 Jazzy (installed under `/opt/ros/jazzy` here):

```bash
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 -m venv --system-site-packages src/base_navigation/.venv
source src/base_navigation/.venv/bin/activate
python -m pip install -r src/base_navigation/requirements.txt
python -c 'import rclpy, geometry_msgs.msg, sensor_msgs.msg, cv2, numpy'
```

Use your robot's ROS distribution in the source path if different. `rclpy` and
`geometry_msgs` come from the ROS installation, rather than this pip requirements
file. Use this same activated environment to run navigation:

```bash
python src/base_navigation/main.py 'http://192.168.1.50:8080/stream.mjpg' \
  --robot-z 0.20 --tilt-rad 0.261799
```

FOV defaults are 77.32° horizontal and 61.93° vertical; override them with
`--horizontal-fov` and `--vertical-fov` if needed.
Replace the example URL and camera calibration with your actual values. Use
`--snapshot` for HTTP JPEG snapshots. `--help` lists gains, speed limits,
tolerances, processing resolution, and detection thresholds. Running `main.py` starts
navigation as soon as a camera target and a `cmd_vel` subscriber are available.

The target coordinates are meters, X right and Y forward. Angular error is
`atan2(x, y)`. Above `--angle-tolerance` (default 0.05 rad), forward velocity is
zero and angular velocity is `-angular_kp * error`, limited to 0.4 rad/s. Once
aligned, angular velocity is zero and forward velocity is
`linear_kp * (distance - target_distance)`, limited to 0.15 m/s. If alignment
drifts, the robot stops translating and turns again.

The default goal is zero distance with a 0.05 m tolerance. Use, for example,
`--target-distance 0.5` to stop 0.5 m away. Reaching the goal commands zero speed;
the loop keeps watching and can follow a subsequently moved target.

Projection assumes flat ground and a fixed, forward-facing camera with zero yaw
and roll. Camera height is 0.83 m plus `--robot-z`; tilt is downward in radians.
Landscape-shaped camera frames rotate 90° clockwise into portrait before detection,
projection, and display; already-portrait frames stay unchanged. FOV describes the
upright portrait image axes, with no automatic FOV swapping.
Distance is measured from the point below the camera, so mount position determines
where the robot stops. A target that disappears below the image causes a stop;
the robot cannot continue toward an unseen zero position.

Missing detections, invalid ground projections, camera errors, and expired results
command zero velocity. Results expire after 0.5 s by default, measured from the
start of acquisition; the existing camera API separately rejects frames older
than one second. Camera waits and detection run in a worker so the ROS executor
continues publishing and enforcing each movement's 0.15 s deadline. Ctrl+C stops
the base and closes the camera. This is visual approach control with no obstacle
avoidance.

## Navigation dashboard

`dashboard.py` serves a local browser view with the camera image, a crosshair on the
selected frisbee, ground distance in meters, bearing in radians and degrees,
and the P controller's proposed state and velocities. The camera is interpreted
and displayed in portrait, with telemetry on its right.
It uses the same detector, projection, calibration arguments, and controller as
navigation. It starts with robot motion stopped. Camera preview does not require
ROS; ROS is loaded only when you click **Connect ROS**.

```bash
python src/base_navigation/dashboard.py
```

Open **http://127.0.0.1:8765** in your browser; stop with Ctrl+C in the terminal.
Enter or select a recent camera URL, choose stream or snapshot format, and set
lift, tilt, FOV, processing width, and RGB sum/delta thresholds. Detection uses
contour ellipses.
Settings are grouped into **Camera**, **Robot**, **Detector**, and **Controller**
tabs. Robot contains lift, tilt, and speed limits; Controller contains gains,
tolerances, and target distance. Connect, Save, Reload, and Disconnect remain
available below every tab and apply to the complete configuration.
Detection defaults to **RGB sum ≥160** and **channel delta ≤55**. Sum ranges
from 0–765; delta is max(R,G,B) − min(R,G,B), ranging from 0–255. Raise the sum
to require brighter pixels; raise delta to allow more color variation. You can
also set these with `--rgb-sum-min` and `--channel-difference-max` in navigation
or dashboard mode. Result timeout is in Camera, processing options are in Detector,
and P-control tuning is in Controller.
Click **Connect / apply settings** to start or reconnect with those values.
**Disconnect camera** closes the source and clears the preview. Invalid settings
show an error and leave the current connection unchanged.

All camera, detection, and controller values are stored in
`src/base_navigation/settings.json`. Navigation and dashboard startup load this JSON file;
explicit command-line values override the saved values for that run. Use
`--settings-file /path/to/settings.json` to select a different settings file.
The file also stores the browser port; port changes take effect after restarting
the dashboard server.

**Save settings** writes the current form values to the JSON file without
connecting or changing the active camera. You can save calibration with the
camera URL blank. **Reload saved values** reads the JSON file again and restores
the form, including camera format and controller values; it leaves the running
camera unchanged. Click **Connect / apply settings** to use the reloaded values.
File or validation errors appear below the buttons and leave the current values
in place.

FOV starts at 77.32° horizontal and 61.93° vertical; initial tilt is 15° downward
(0.261799 rad). Match lift and tilt to your setup. Recent camera addresses are
saved in this browser, excluding addresses containing credentials or query tokens.
You can still supply the URL and settings on the command line to connect at startup.

Use `--port` to select another port. The browser view works with headless OpenCV
and does not need GTK, Qt, a desktop display, or additional Python dependencies.
Navigation requirements use `opencv-python-headless`; there is no desktop window
backend.

Use your actual camera calibration. Add `--snapshot` for a JPEG snapshot URL.
The angle is zero forward and positive right; distance is measured from below
the camera. Missing detections, invalid projections, and expired results show
unavailable measurements. While waiting for frames, the previous camera image
remains visible with its detection marker and measurements cleared.

## Run ROS navigation from the browser

Start the website in the same ROS-enabled environment used for navigation:

```bash
source /opt/ros/jazzy/setup.bash
source src/base_navigation/.venv/bin/activate
python src/base_navigation/dashboard.py
```

Use your installed ROS distribution if different from Jazzy. The virtual
environment must expose the ROS system packages as described above.

1. Connect the camera and apply the calibration, detector, and controller values.
2. Click **Connect ROS**. The page reports ROS connection and `cmd_vel` subscriber
   status; connecting ROS leaves motion stopped.
3. With a fresh frisbee ground position available, click **Run controller** to
   send the existing turn-then-approach controller's velocities to the robot.
4. Click **Stop robot** to stop motion. **Disconnect ROS** also stops motion and
   closes the ROS connection. The camera preview remains available.

**Body Up · 1 cm** and **Body Down · 1 cm** adjust the torso lift one centimeter
per click. They require a connected lift controller and fresh measured height;
the measured body lift is displayed above the buttons. Positive movement raises
this robot, as confirmed for the hardware. Body adjustments pause navigation;
click **Run controller** again after the lift reaches its target. Live lift
feedback is used for the camera-height calculation when available.

These buttons send position targets as `sensor_msgs/msg/JointState` on
`/lifting_controller/commands`, using the interface in the [lift documentation](../../docs/ros2/README.md#lift).
Targets stay within 0–0.6 m, following the conservative range in the
[interface reference](../../docs/ros2/UnixAI_Panther_ROS2_Interface_Reference.md).
**Stop robot** stops both base motion and the body lift. Hardware operation still
needs verification on the robot; automated checks use mocked controllers.

The ROS controls stay visible above the preview regardless of the selected
settings tab. They show the commanded forward and turn velocities. Run uses the
currently applied settings, so apply form edits before starting. Missing or stale
camera targets command zero velocity. Changing/disconnecting the camera stops
navigation; click Run again after applying the new camera settings.

The browser sends a heartbeat while navigation or body movement is running. Hiding/leaving the
page requests a stop, and a missing heartbeat stops navigation after three
seconds. Start must be clicked again to resume. Closing the server stops and
cleans up ROS. If ROS packages are unavailable, the page reports the error and
camera preview continues to work.

Offline checks (no ROS or camera needed):

```bash
python -m unittest discover -s src/base_navigation/tests -p 'test_*.py'
```

ROS loop tests use mocked nodes and publish no commands. They run when ROS is
available and otherwise skip; the camera, controller, and dashboard checks still run.
