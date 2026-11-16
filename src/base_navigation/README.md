# Frisbee navigation

Both the CLI and dashboard run `main.py`'s `control_loop()`: read the latest target, reject expired results,
compute PD control, and call `robot.set_velocity()`. Responsibilities are split
between:

- `controller.py`: ROS-independent PD control and observation history.
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

Dependencies: ROS 2 supplies `rclpy`, `geometry_msgs`, `sensor_msgs`, and `std_msgs`; the camera/detector needs
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

The target coordinates are meters, X right and Y forward. Camera steering error
is `atan((pixel_x - image_width/2) / fx)`, where
`fx = image_width / (2*tan(horizontal_fov/2))`. It is zero on the vertical image
center line, positive right, negative left, and independent of camera tilt and
ground distance. Steering and distance control run simultaneously. Angular velocity
is `-(angular_kp * bearing + angular_kd * bearing_rate)`, limited to 0.4 rad/s, and zero inside
`--angle-tolerance` (default 0.05 rad). Distance control uses the signed error
`distance - target_distance` and stops translation only when its absolute value
is within `distance_tolerance`. Outside that band, radial speed is
`linear_kp * error + linear_kd * distance_rate`, limited to the configured speed
and clamped toward the distance goal. Multiplying by the cosine of the geometric
bearing slows translation as the target moves off-center. A too-close target
ahead commands reverse; a too-close target behind commands forward to increase
separation. Damping can reduce either command to zero. Translation toward far
rear targets is paused. At the tracking distance, translation stops while steering can
still center the target; once both errors are within tolerance, both velocities
are zero. The linear speed limit applies equally in forward and reverse.

When distance is below `target_distance - distance_tolerance`, steering is paused
(zero angular velocity) while the robot corrects separation along its current
heading. A target ahead is corrected by backing up straight. Steering resumes
once the distance enters the tolerance band; the angular D history is cleared
during correction so resuming does not create a derivative kick. The debug angle
error remains visible while steering is paused.

The Controller tab exposes **Angular D gain** and **Linear D gain**, also available
as `--angular-kd` and `--linear-kd`. Both default to zero, including for older
settings files. Set nonzero gains and use **Save & apply settings** to enable
damping. Linear D reduces approach speed as distance falls; Angular D damps
turning as the bearing approaches center. Rates use observation timestamps and
a 0.15 s low-pass filter. Repeated frames hold the previous rates. Missing,
invalid or expired targets, long gaps, and gain changes reset derivative history;
the first observation starts with zero derivative. Existing speed limits,
tolerances, and target-loss stopping still apply.

The default goal is 0.3 m from the camera's ground origin, with a 0.05 m tolerance.
Set **Tracking target distance (m)** in the Controller tab and click **Save & apply settings** to persist and use the new value. The CLI also
accepts `--target-distance`, for example `--target-distance 0.5` to stop 0.5 m away.
With the default settings, the accepted distance band is 0.25–0.35 m. Below it,
the robot increases separation; above it, the robot approaches. The loop keeps
watching and can follow a subsequently moved target.

Projection assumes flat ground and a fixed, forward-facing camera with zero yaw
and roll. Camera height is `--camera-base-height` (default 0.83 m) plus the
`--robot-z` mounting offset plus fresh measured ROS body lift. Camera tilt is
the `--tilt-rad` mounting offset plus fresh measured ROS head pitch, in radians
downward from horizontal. Missing or stale feedback contributes zero; configured
mounting values remain in effect. The same calculation is used in the dashboard
and CLI navigation loop.
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
the base and closes the camera. Navigation uses camera-based tracking without
obstacle avoidance.

## Navigation dashboard

`dashboard.py` serves a local browser view with the camera image, a crosshair on the
selected frisbee, ground distance in meters, bearing in radians and degrees,
and a proportional preview of state and velocities. The live robot speed display
includes derivative damping when D gains are enabled.
The image debug panel also shows signed **Position error** (`distance - target_distance`)
in meters and **Angle error** (horizontal image angle) in radians and degrees, taken directly
from the controller calculation before tolerances and speed limits. Positive
angle means right; negative means left. Unavailable errors display dashes.
The same image angle drives steering, derivative damping, and the debug display.
Ground bearing is not displayed and does not drive camera steering. For
callers supplying only ground X/Y, the controller retains the ground-bearing
fallback, wrapped modulo 180° below target distance with derivative history reset
when crossing that distance boundary.
The camera is interpreted and displayed in portrait, with telemetry on its right. The dashboard rotates
the camera image 180° to correct its upside-down mounting before detection,
projection, and display.
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
The Detector tab also has ellipse radius limits, minimum axis ratio, contour area,
white fill fraction, and a side standard deviation threshold (default: 30).
Side 0 is below the threshold; side 1 is at or above it, using the inner 65%
of both ellipse radii. The detection website displays these side labels.
Radius and area refer to the processing resolution.
Both websites use white-mask contour ellipses for detection. Retired edge-detector
fields in older settings files are ignored and removed on the next explicit Save.
Use Save & apply settings to update detection; Save and Reload include these
values. Older settings files load the previous detector defaults automatically.
Click **Save & apply settings** to save all values to JSON and reconnect with them.
**Disconnect camera** closes the source and clears the preview. Invalid settings
show an error and leave the current connection unchanged.

All camera, detection, and controller values are stored in
`src/base_navigation/settings.json`. Navigation and dashboard startup load this JSON file;
explicit command-line values override the saved values for that run. Use
`--settings-file /path/to/settings.json` to select a different settings file.
The file also stores the browser port; port changes take effect after restarting
the dashboard server.

**Save settings** writes the current form values to the JSON file without
connecting or changing the active camera. Both Save actions update the values shown
after a browser refresh; height and tilt also survive restarting the server. You can save calibration with the
camera URL blank. **Reload saved values** reads the JSON file again and restores
the form, including camera format and controller values; it leaves the running
camera unchanged. Click **Save & apply settings** to use the reloaded values.
File or validation errors appear below the buttons and leave the current values
in place. If writing fails, Apply also leaves the active camera unchanged. Atomic
saves preserve the settings file’s existing permissions.
The read-only `/settings/saved` endpoint returns the current validated file
contents for the Streamlit visualization, independently of cached form or
currently applied runtime settings.

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

## From the Web UI (robot bridge)

The robot bridge (`docker/run.sh docker/bridge.sh`) also runs this dashboard, on loopback inside the
bridge, with the same `settings.json`. The Web UI's **Frisbee navigation** page (Full view, or `/navigation`)
shows it through the cloud: `interface/frontend/navigation.html` is a copy of `dashboard.html` whose requests
go to `/api/v1/robotics/navigation/...` and are forwarded to the bridge
(`interface/robot_bridge/.../navigation_mode.py`). Keep the two pages in sync.
There, base velocity goes through the bridge's safety supervisor like a manual jog (speed limits, E-Stop,
cloud watchdog and front LiDAR stop apply); lift and head use `navigation.py` as below,
with its head convention. The standalone dashboard keeps working; do not run both against the same robot.

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
   run simultaneous steering and distance control on the robot.
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

**Look Up · 5°** and **Look Down · 5°** adjust head pitch while preserving measured
yaw. They require a subscriber on `/head_forward_position_controller/commands`
and fresh `joint_head1` (pitch) and `joint_head0` (yaw) feedback on `/joint_states`.
The page displays measured pitch in degrees, positive downward. Commands use
`std_msgs/msg/Float64MultiArray` in `[yaw, pitch]` order, following the observed
robot behavior rather than the reversed labels in the reference. Only the second
command value changes during Up/Down steps. The target is streamed at 0.2 rad/s, within
−0.26 to 1.41 rad (about 15° up to 81° down).

Head adjustments pause navigation and clear previous camera targets. Measured
head tilt is added automatically to the saved camera **Tilt offset (rad)**.
The Robot tab has a configurable **Camera base height (m)**, default 0.83 m,
and **Height offset (m)** for mounting adjustments. Below these inputs it shows
the applied total height and tilt (radians and degrees), with the full sums and
feedback freshness. These totals are the values used for ground projection.
Use a zero height offset when the base height already describes the camera at
zero robot lift. Existing saved height/tilt values become mounting offsets;
remove any robot lift or head pitch already included in those manual values to
avoid counting them twice.
**Stop robot**, disconnect, stale head feedback, and browser heartbeat expiry
cancel head streaming and request a held position. Real head operation still
needs verification on the robot.

The ROS controls stay visible above the preview regardless of the selected
settings tab. They show the commanded forward and turn velocities. Run uses the
currently applied settings, so apply form edits before starting. Missing or stale
camera targets command zero velocity. Changing/disconnecting the camera stops
navigation; click Run again after applying the new camera settings.

The browser sends a heartbeat while navigation, body, or head movement is running. Hiding/leaving the
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
