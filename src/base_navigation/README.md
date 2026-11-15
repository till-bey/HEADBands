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

The target coordinates are meters, X right and Y forward. Angular error is
`atan2(x, y)`. Steering and distance control run simultaneously. Angular velocity
is `-angular_kp * bearing`, limited to 0.4 rad/s, and zero inside
`--angle-tolerance` (default 0.05 rad). Forward velocity is
`min(max_linear_speed, linear_kp * (distance - target_distance)) * cos(bearing)`
when outside the distance tolerance and the target is ahead. This slows approach
as the target moves off-center, while continuing to steer toward it. Targets to
the side or rear cause turning in place. At the tracking distance, forward
motion stops while steering can still center the target; once both errors are
within tolerance, both velocities are zero.

The default goal is 0.3 m from the camera's ground origin, with a 0.05 m tolerance.
Set **Tracking target distance (m)** in the Controller tab and click **Save & apply settings** to persist and use the new value. The CLI also
accepts `--target-distance`, for example `--target-distance 0.5` to stop 0.5 m away.
Reaching the goal commands zero speed (including when already closer than the goal);
the loop keeps watching and can follow a subsequently moved target.

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
the base and closes the camera. CLI approach control has no obstacle avoidance;
the dashboard offers the optional ultrasound potential field described below.

## Navigation dashboard

`dashboard.py` serves a local browser view with the camera image, a crosshair on the
selected frisbee, ground distance in meters, bearing in radians and degrees,
and the P controller's proposed state and velocities. The camera is interpreted
and displayed in portrait, with telemetry on its right. The dashboard rotates
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
and white fill fraction. Radius and area refer to the processing resolution.
Rim edge lower/upper thresholds are saved as `canny_low`/`canny_high` and are
shared with the Streamlit visualization. Older saved files default to 40/120.
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

### Ultrasound proximity display

After **Connect ROS**, `navigation.py` discovers `sensor_msgs/msg/Range` topics
and subscribes using sensor-data QoS. Only messages marked `ULTRASOUND` are shown;
infrared readings are excluded. No enable service or motion command is sent.
The actual robot's distance topic/type is not recorded in the local interface
reference, so this standard-message integration still needs live verification.
If the robot publishes an array or vendor message instead, its topic, units, and
sensor order are needed to add the corresponding adapter.

The four diamond segments show front/right/rear/left proximity: red at ≤0.30 m,
amber at ≤0.60 m, and green beyond 0.60 m. A fresh close reading on **any** sensor
shows **!** in the center, including sensors not yet assigned to a side. Missing,
invalid, or older-than-one-second readings are gray and are never shown as clear.
When multiple sensors cover one side, its nearest reading is shown. A side with
incomplete coverage stays gray unless a known close reading makes it red/amber.

Unambiguous direction names in topics/frames are placed automatically. Use
**Sensor positions** to assign numbered/corner sensors to their physical sides;
select multiple entries for a side if needed. Placements are remembered in this
browser. Standard Range ±infinity readings display an upper/lower distance bound,
without sending non-finite numbers to JSON. With obstacle avoidance disabled,
this display is read-only and does not alter the controller's commands.

### Optional potential-field obstacle avoidance

Click **Enable obstacle avoidance** under Ultrasound proximity, check the sensor
positions, then click **Run controller**. Enabling/disabling avoidance or changing
its sensor map stops navigation and requires another Run click. Avoidance starts
disabled when the server starts. It uses the shared `main.control_loop()` and
keeps all ROS operations in `navigation.py`.

`avoidance.py` combines the tracking attraction with repulsion along the four
sensor directions. Within 0.60 m, repulsive strength is
`0.008 * (1/d - 1/0.60) / d²`, using the nearest reading on each side. The resulting
field steers away from obstacles and reduces forward speed while respecting the
existing speed limits and never requesting reverse motion. Any reading at or
inside 0.20 m (or a Range `too_close` report) holds both base velocities at zero.
Repulsion never starts movement without a fresh camera target or moves the robot
away from an already-reached target position.

All four sides need fresh, valid coverage; every detected ultrasound sensor must
be assigned. The enable action sends the display's sensor map to the server, so
both use the same assignments. Incomplete, invalid, stale or unmapped coverage
stops and disarms navigation. Restored feedback does not restart it automatically;
click Run again. A finite `beyond_range` lower bound must cover at least 0.60 m.
Coverage and close-obstacle checks run again immediately before commands are sent.

This is a reactive field, not a path planner: a centered obstacle can cancel the
attraction and cause a stopped local minimum. Physical collision clearance and
the actual ultrasound topic/type still need verification on the robot. It does
not replace the robot's own safety system or obstacle-aware path planning.

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
