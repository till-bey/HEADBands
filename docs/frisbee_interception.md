# Frisbee detection, tracking and interception (ROS 2 Humble)

Packages: `src/frisbee_msgs` (messages) and `src/frisbee_interception` (nodes, configs, launch, tests, CI).
CPU only: Python + OpenCV + NumPy. No neural networks, no GPU, no training.

**Default state: prediction only. Nothing in these packages moves the robot, and no joint command is ever published.**

---

## 1. What was found in the workspace (Phase 1)

| Question | Finding |
|---|---|
| ROS distro | ROS 2 **Humble** (`docker/Dockerfile`: `osrf/ros:humble-desktop`, CycloneDDS, `ROS_DOMAIN_ID=15`) |
| Phone camera | **No ROS node existed.** `perception/camera/camera_stream.py` (`LatestFrameCamera`) reads a phone MJPEG/RTSP stream in plain Python, and `perception/Dataset/Videos/*.mp4` holds 4 phone recordings (720×1280 portrait, 30 fps). A new `phone_camera` node publishes it as `sensor_msgs/Image` and reuses that reader. |
| Robot camera | Orbbec on the robot (topics such as `/camera/color/image_raw`, see `docs/ros2/`). Can be used with `use_phone_camera:=false image_topic:=...`. |
| Arm control | `ros2_control`. Arms: `dual_arm_forward_position_controller` (default) and `left/right_arm_joint_trajectory_position_controller` (`FollowJointTrajectory`). `panther_examples/cartesian_follow.py` does PyKDL IK on the URDF, in `body_link`, and sends **one** `FollowJointTrajectory` goal, joint speed capped at `MAX_JOINT_VEL = 0.5 rad/s`. |
| How the robot receives targets | **There is no existing "go to this Cartesian point" interface** (topic, service or action). `cartesian_follow` is a command-line tool that takes a whole CSV path. |
| TF / calibration | `robot_description.urdf` has **no camera frame** (`camera_rgb_frame` is commented out). There is **no phone-camera intrinsic calibration** and **no camera-to-robot extrinsic** anywhere in the repo. None was invented. |

### Architecture

```
 phone (MJPEG/RTSP) or video file        or any existing camera topic
            │                                         │
      phone_camera ──► /phone_camera/image_raw  ◄─────┘  (+ /phone_camera/camera_info if calibrated)
                               │
                       frisbee_detector ──► /frisbee/detections   (FrisbeeDetectionArray)
                               │            /frisbee/debug/detections (Image, only when watched)
                       frisbee_tracker  ──► /frisbee/track        (FrisbeeTrack: state + short history)
                               │
                     frisbee_interceptor ─► /frisbee/intercept    (FrisbeeIntercept: valid, TTI, point, σ, confidence)
                               │            /frisbee/catch_point  (PointStamped, every valid prediction, display)
                               │            /frisbee/markers      (MarkerArray for RViz2)
                               │            /frisbee/debug/intercept (Image overlay)
                               └──────────► /frisbee/catch_target (PointStamped)  ONLY when armed AND every check passes
                                                       │
                                     (not connected: see §7 "Controller integration")
```

---

## 2. Algorithms

**Detector** (`core/detection.py`, `frisbee_detector`)
- Runs at `processing_width` (360 px) and reports full-resolution coordinates. ~13 ms/frame at 720×1280 on one laptop core.
- Foreground mask, chosen with `detection.method`:
  - `background` (**default**) — MOG2 background subtraction. Needs a fixed camera (the interception set-up). A
    disc flying straight at the camera barely moves in the image, and only a background model sees it reliably.
  - `motion` — three-frame differencing with phase-correlation shake compensation. For a hand-held camera and
    sideways throws (the 2026-11-14 recordings). It misses head-on throws (see §8).
  - `hsv`, `motion_and_hsv`, `background_and_hsv` — optional color gate (`hsv_lower/upper`). Off by default: no
    color is assumed. The YAML range is the measured white disc of the 2026-11-15 clips, and
    `background_and_hsv` with it removed the false tracks on heads there (§8.1).
- Each blob gets an ellipse fit, then geometric filters: area, axis length, aspect (minor/major ≥ 0.06: an edge-on
  disc is thin), solidity, ellipse fill. Confidence comes from solidity, fill and aspect.
- Output per detection: center, bbox, ellipse (major/minor/angle), confidence. Apparent diameter = **major axis**
  (tilt shrinks only the minor axis).

**Tracker** (`core/tracking.py`, `frisbee_tracker`)
- Up to 8 constant-velocity Kalman tracks on (u, v, d) with image velocity and size rate, using the **message
  timestamps** (not arrival time).
- Gating on distance (gate grows with uncertainty) and size ratio. Associations prefer confirmed tracks.
- The primary track (the one predictions use) is the confirmed track that moves or grows fastest. It switches only
  to a track 1.5× faster, so walking people never hide a disc in flight.
- A track is confirmed after `min_hits` (3) hits **and** if it moves or grows faster than `min_motion_px_s`, which
  rejects isolated false detections and static distractors. It coasts for `max_coast_s` and is then LOST. Time
  going backwards resets it.
- Publishes the current state and the last `history_len` raw measurements.

**Predictor** (`core/prediction.py`, inside `frisbee_interceptor`): a straight-line model with weighted
least-squares fits over the last `max_points` measurements.
- *metric mode* (needs intrinsics + TF + catching plane): depth Z = fx·D/d, X = (u−cx)·Z/fx, Y = (v−cy)·Z/fy. A
  3D line X(t) is fitted, transformed to `target_frame` and intersected with the catching plane. This gives the
  catch point, the intercept time, and σ from the fit covariance.
- *image mode* (no metric calibration): 1/d is linear in time for constant approach speed, so the time to contact is
  the time when d reaches the calibrated `catch_diameter_px`. The crossing pixel comes from the perspective-correct
  fits of (u−cx)/d and (v−cy)/d. An optional homography maps that pixel to catching-plane coordinates in
  `target_frame`.
- **Rejected** (published with `valid=false` and a reason): too few points, too short a time span, not approaching
  (size not growing), size fit not linear, time to intercept outside the window, σ (position/pixel/time) too large.

---

## 3. Topics, messages, parameters

| Topic | Type | Notes |
|---|---|---|
| `/phone_camera/image_raw/compressed` | `sensor_msgs/CompressedImage` (JPEG) | **default** (`image_transport:=compressed`), sensor-data QoS |
| `/phone_camera/image_raw` | `sensor_msgs/Image` (bgr8) | `image_transport:=raw`. Raw 720p frames (2.7 MB) are mostly dropped by best-effort DDS between processes (measured: ~2 of 30 fps arrived), so use raw only for cameras that already publish raw on the same machine with shared memory |
| `/phone_camera/camera_info` | `sensor_msgs/CameraInfo` | only with `calibration_file` |
| `/frisbee/detections` | `frisbee_msgs/FrisbeeDetectionArray` | header = image header |
| `/frisbee/track` | `frisbee_msgs/FrisbeeTrack` | state TENTATIVE/CONFIRMED/COASTING/LOST + history |
| `/frisbee/intercept` | `frisbee_msgs/FrisbeeIntercept` | every track update, plus `valid=false` from the 10 Hz watchdog when data stops |
| `/frisbee/catch_point` | `geometry_msgs/PointStamped` | valid predictions with a point in `target_frame` |
| `/frisbee/catch_target` | `geometry_msgs/PointStamped` | **only when armed and safe** |
| `/frisbee/markers` | `visualization_msgs/MarkerArray` | catch point, 3D line, plane, text |
| `/frisbee/debug/detections`, `/frisbee/debug/intercept` | `sensor_msgs/Image` | only drawn while subscribed |
| `/frisbee/disable` | `std_msgs/Empty` | disarms immediately |
| `/frisbee_interceptor/enable_interception` | `std_srvs/SetBool` | arm / disarm |

`FrisbeeIntercept`: `valid`, `status` (reason if invalid), `mode` (metric/image), `catch_point`, `intercept_time`,
`time_to_intercept`, `position_sigma`, `time_sigma`, `confidence`, `pixel_u/v`, `in_workspace`,
`interception_enabled`.

All parameters are in `src/frisbee_interception/config/frisbee.yaml`, with comments. Calibration values there are
**empty on purpose**.

Dependencies: `rclpy`, `sensor_msgs`, `geometry_msgs`, `std_msgs`, `std_srvs`, `visualization_msgs`, `tf2_ros`,
`rosidl_default_generators` (msgs), `python3-opencv`, `python3-numpy`, `python3-yaml`, `python3-pytest` (tests).
No `cv_bridge` (images are converted in `ros_image.py`).

---

## 4. Build and run

```bash
# inside the ROS container (docker/run.sh), repo mounted at /workspace
cd /workspace && colcon build --packages-select frisbee_msgs frisbee_interception && source install/setup.bash

# phone stream (IP Webcam / DroidCam style URL) — prediction only
ros2 launch frisbee_interception frisbee.launch.py source:=http://PHONE_IP:8080/video rviz:=true

# a recorded video (portrait phone clips need no rotation; add rotate:=cw|ccw|180 otherwise)
ros2 launch frisbee_interception frisbee.launch.py source:=/workspace/perception/Dataset/Videos/VID_20261114_222519.mp4 loop:=true

# the robot's camera instead of the phone (its /compressed topic, or image_transport:=raw)
ros2 launch frisbee_interception frisbee.launch.py use_phone_camera:=false \
    image_topic:=/camera/color/image_raw camera_info_topic:=/camera/color/camera_info

# your own tuned / calibrated config
ros2 launch frisbee_interception frisbee.launch.py source:=... config:=/workspace/my_frisbee.yaml

# watch
ros2 topic echo /frisbee/intercept
ros2 run rqt_image_view rqt_image_view /frisbee/debug/intercept
```

Without ROS (tuning, batch runs): `python -m frisbee_interception.offline VIDEO --config frisbee.yaml --out debug.mp4 --json log.json`

Tests: `colcon test --packages-select frisbee_interception && colcon test-result --verbose`, or `pytest src/frisbee_interception/test`.
- `test_tracking.py`: confirmation, velocity, size rate, isolated false detections, static distractors, loss/coasting, timestamps.
- `test_prediction.py`: line fit, metric and image prediction against ground truth, rejection of sideways/receding/noisy/too-late throws.
- `test_synthetic_throw.py`: detector → tracker → predictor end to end on a rendered throw with exact ground truth.

---

## 5. Calibration

Nothing below is filled in for you. Each step says what to measure and where to put it.

### 5.1 Camera intrinsics (needed for metric mode)

Print a checkerboard (for example 9×6 inner corners, 25 mm squares). Lock the phone's zoom and focus, use the same
resolution and orientation as in use, then:

```bash
ros2 launch frisbee_interception frisbee.launch.py source:=http://PHONE_IP:8080/video   # publishes /phone_camera/image_raw
ros2 run camera_calibration cameracalibrator --size 9x6 --square 0.025 \
    image:=/phone_camera/image_raw camera:=/phone_camera
```

Move the board until the bars are green, press CALIBRATE, then SAVE. Extract `ost.yaml` from
`/tmp/calibrationdata.tar.gz` and launch with `calibration_file:=/path/ost.yaml`. The node then publishes
`camera_info`, and the interceptor uses it. (Or set `camera.fx/fy/cx/cy` in the YAML.)

### 5.2 Camera → robot transform (needed for metric mode)

The URDF has no camera frame. Mount the phone rigidly, measure its optical frame relative to `body_link` (x forward,
z up; optical frame: z out of the lens, x right, y down), and publish it:

```bash
ros2 run tf2_ros static_transform_publisher --x X --y Y --z Z --roll R --pitch P --yaw Yw \
    --frame-id body_link --child-frame-id phone_camera_optical_frame
```

A tape measure gives ±2–3 cm. Verify it: hold the frisbee at a measured point and compare it with
`/frisbee/catch_point`. A better option is to put a checkerboard at a known pose in `body_link` and solve PnP.
The interceptor uses the TF only if it exists, and logs a warning otherwise.

### 5.3 Catching plane (metric mode)

`catch_plane.point` and `catch_plane.normal` in `target_frame`. For example, a vertical plane 0.45 m in front of
`body_link` is `point [0.45, 0, 0]`, `normal [1, 0, 0]`. Choose it where the hand can actually be (check with
`cartesian_follow --dry-run`). Then set `safety.workspace_min/max` to a box the arm can reach safely.

### 5.4 Image-mode calibration (no intrinsics / TF yet)

- `prediction.catch_diameter_px`: hold the frisbee face-on at the catching plane, in front of the hand, and read its
  apparent diameter (`major_px` in `ros2 topic echo /frisbee/detections`, averaged over a few frames). This alone
  gives *when* and *at which pixel* the disc crosses.
- Optional `image_calibration.homography`: mark ≥4 points on the catching plane with known plane coordinates
  (a, b) in metres, read their pixels, and compute `cv2.findHomography(pixels, ab)`. Put the 9 numbers (row-major) in
  the YAML, plus `plane_origin`, `plane_axis_a` and `plane_axis_b` (the plane's (0,0) point and unit axes in
  `target_frame`). Then the predicted pixel becomes a 3D point in `target_frame`.

---

## 6. Safety

- `interception.enabled: false` by default. The only output toward the robot (`/frisbee/catch_target`) is silent
  until armed.
- Arming: `ros2 service call /frisbee_interceptor/enable_interception std_srvs/srv/SetBool "{data: true}"`. **It is
  refused while `safety.workspace_min/max` is unset.**
- Disarm immediately: `ros2 topic pub --once /frisbee/disable std_msgs/msg/Empty {}` (or the service with `false`, or
  set the parameter `interception.enabled false`).
- A catch target is sent only if **all** of these hold: confirmed track, fresh data (`max_data_age_s`), valid
  prediction (every rejection rule in §2), confidence ≥ `safety.min_confidence`, metric σ ≤ `max_position_sigma_m`,
  point inside the workspace box, time to intercept ≥ `min_time_to_intercept_s`, and a jump from the previous target
  ≤ `max_target_step_m`. Each withheld target is logged with its reason.
- Watchdog: if tracks stop for `track_timeout_s`, an explicit `valid=false` is published and the target memory is
  cleared.
- Existing low-level limits (controller limits, `MAX_JOINT_VEL`, e-stop) are untouched. No joint command is
  published.

---

## 7. Controller integration (not done: proposal)

No existing interface accepts a Cartesian target, and the task says not to add a second controller. So
`/frisbee/catch_target` is **not connected** to anything. The smallest change that would connect it reuses what
already exists:

1. Move the IK and timing in `cartesian_follow.py` (`Chain`, `plan`, `timed`) into a small long-running node
   `catch_follow` (same package). It subscribes to `/frisbee/catch_target` (`PointStamped` in `body_link`), uses a
   fixed catching orientation, and solves IK seeded with `/joint_states`.
2. It sends **one** `FollowJointTrajectory` goal to the existing `<arm>_arm_joint_trajectory_position_controller`,
   with the same controller switch `cartesian_follow` already does, and keeps the `MAX_JOINT_VEL` cap.
3. It cancels or replaces the goal when a new target arrives (`max_target_step_m` already bounds jumps), and it
   cancels on `/frisbee/disable`.

That is about 100 lines and needs no new controller, no new message, and no change to `ros2_control`. It was left
out on purpose because it moves the arm, and the timing below makes it unlikely to help yet.

**Feasibility:** at 0.5 rad/s, a 0.3 rad joint move takes ≥0.6 s. A 10 m throw at 10 m/s gives at most ~0.6 s of
valid prediction, and in the recorded clips the disc is visible for only 0.27 s. Real catching needs a pre-positioned
hand, short corrections, and higher joint speeds, and those must be validated on the robot with the vendor.

---

## 8. Results and limitations

| Input | Result |
|---|---|
| Unit tests (20) | pass with OpenCV 4.5.4 (Ubuntu 22.04 / Humble) and 4.14 |
| Synthetic throws, 9 m → camera at 9 m/s, `background`, 3 noise seeds | first valid prediction **0.45–0.65 s** before contact. Crossing-time error stays under **40% of the time still to go** (≈ 0.1–0.2 s half a second out, a few tens of ms near contact). Pixel error ≤ 50 px early and **≤ 20 px in the last 0.3 s** (≤ 4.5 cm at 1 m) |
| Full ROS launch, image mode (Cloud Build CI, looped synthetic throw, 20 s) | 55 valid predictions. Pixel error ≤ 17 px in the last 0.3 s; the first prediction of each throw (~0.5 s out) is 48–64 px off. Arming refused (no workspace), **0 catch targets** |
| Full ROS launch, metric mode (intrinsics + static TF + plane) | catch point error **9–37 mm**. Armed: 16 catch targets; after `/frisbee/disable`: **0**. Far predictions correctly rejected for position uncertainty > 0.25 m |
| Recorded clips (`perception/Dataset/Videos`) | the disc is detected and tracked, but there are **no valid predictions**, correctly: three clips are sideways throws (not approaching), and the one toward-camera clip has ~8 flight frames (0.27 s) on a hand-held, shaking phone. Image mode also needs `catch_diameter_px`. |

The main error source in image mode is a constant ~3 px bias in the measured diameter (blob edges). A constant bias
bends the 1/d-vs-time line while the disc is still small. For that reason a prediction is accepted only after 1/d
has shrunk by ≥ 30% within the fit window (`prediction.min_size_growth: 0.3`). Lowering it gives earlier but less
accurate predictions.

### 8.1 Fixed-camera recordings (2026-11-15, three WhatsApp clips, 30 fps, 480×864 / 848×480)

The phone was static (on the floor or a table) and people threw from about 3–5 m. Six throws were identified:

| Clip / frames | Throw | Disc visible on its own | Result (`background_and_hsv`, white) |
|---|---|---|---|
| WA0000 367–372 | toward the camera | ~5 frames (0.17 s) | tracked; valid from **0.075 s** before arrival (frames 368–371) |
| WA0000 709–714 | toward the camera | ~5 frames | tracked; valid from **0.056 s** before |
| WA0000 1093–1097 | low, past the right edge | ~3 frames | tracked; valid **0.02 s** before |
| WA0000 1463–1474 | down-left, in front of the thrower | ~10 frames, mostly overlapping the body | not separated from the thrower → no prediction |
| WA0001 376–381 | sideways, leaves the frame | ~5 frames, attached to the arm until 381 | no prediction (correct) |
| WA0002 302–307 | toward the camera from ~3 m | 3 frames (0.1 s) | too few points → no prediction |

**Conclusion:** detection and tracking work on these clips. But at 3–5 m and 30 fps the disc is in the air for only
0.1–0.17 s, and the first valid prediction comes 0.02–0.075 s before arrival. That is far too late for any arm, and
no algorithm can predict from 2–3 frames. Errors seen and what was changed:

- **Walking people / heads** grow in the image like an approaching disc (and are about as big). With `background` a
  head produced a valid prediction. Fix: the white color gate (`background_and_hsv`, measured range in the YAML)
  removed it.
- **Warm-lit hands and white shoes** still pass the white gate. A hand reaching for the phone, and a track that jumped
  between white shoes and the disc held in the hand, gave 2 false valid predictions. This is why arming, the
  workspace box and the confidence threshold matter. Real use needs a disc color that is rare in the scene.
- **Primary track:** it was the oldest confirmed track (often a head), so the disc was ignored. Fixed: the primary is
  now the fastest-moving confirmed track (hysteresis 1.5×), and 8 track slots are available.
- **Coasting track:** while the track coasted, the last prediction was re-published with an outdated time to
  intercept. Fixed: predictions are re-timed to the current frame and expire after 0.1 s without new measurements
  (`prediction.max_measurement_age_s`). The ROS interceptor uses the same code.

### 8.2 What a useful recording looks like

- Camera **fixed** (tripod, or the robot's own mount), at the catching position.
- Thrower **8–15 m** away, throwing **at** the camera, with the disc visible from the release on (not in front of the
  body or the background crowd).
- **60 fps or more** (most phones: 1080p60 or 720p120). This doubles to quadruples the frames per throw.
- A disc color that is rare in the scene. Here the floor is orange and the light is warm, so bright green or pink
  would beat white. Then use `background_and_hsv` with that color's range.
- Hold the disc face-on at the catching distance for a few seconds at the start, to read `catch_diameter_px`.

---

## 9. CI on Google Cloud (CPU)

```bash
gcloud builds submit . --config src/frisbee_interception/ci/cloudbuild.yaml \
    --ignore-file src/frisbee_interception/ci/.gcloudignore --region europe-west6
```

`ci/run_ci.sh` runs on `osrf/ros:humble-desktop`: colcon build → colcon test → the full ROS launch on a synthetic
throw. That is done twice: *image mode*, disarmed, which checks that arming is refused and that no target is ever
sent; and *metric mode* with a static TF, which arms, checks the targets against ground truth, sends
`/frisbee/disable`, and checks that no targets follow. It then replays every recorded clip offline. Reports and
annotated videos are uploaded to `gs://<project>_cloudbuild/frisbee-ci/<build id>/ci_out/`. The same script runs
locally in Docker (see its header).
