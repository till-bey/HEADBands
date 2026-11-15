# Frisbee pickup motion

This folder contains a ROS-free stable point-to-point pickup planner.

## Camera floor calibration

For a fixed head-mounted camera looking at the floor, make a CSV with at least
four non-collinear correspondences:

```text
pixel_x,pixel_y,robot_x,robot_y
120,80,0.35,-0.40
540,90,0.35,0.40
100,410,1.10,-0.45
570,420,1.10,0.45
```

Fit and inspect the calibration:

```bash
python3 motion/pickup/camera_calibration.py fit \
  --points camera_points.csv \
  --robot-scale 0.001 \
  --output motion/pickup/models/camera_floor.json
```

Create an annotated image separately:

```bash
python3 motion/pickup/camera_calibration.py overlay \
  --image camera_frame.jpg \
  --points camera_points.csv \
  --robot-scale 0.001 \
  --output /tmp/camera_calibration_overlay.png
```

The example points use millimetres (`480`), so `--robot-scale 0.001` stores
the homography output in metres, matching the pickup and ROS path conventions.
The saved homography maps pixels from the displayed, oriented image to robot-
frame floor `X,Y`. The CSV `robot_x,robot_y` values are used directly: provide
images and annotations where robot `X` increases upward in the image and
robot `Y` increases to the left.

Check a single detection with:

```bash
python3 motion/pickup/camera_calibration.py pixel \
  --calibration motion/pickup/models/camera_floor.json \
  --pixel 320 240
```

Use `overlay` to produce an annotated calibration image. The reported RMS
error is in metres; keep it small compared with the required pickup accuracy.

## Demonstration format

Each pickup demonstration is a CSV with Cartesian gripper positions in `body_link`:

```text
t,x,y,z,roll,pitch,yaw
0.00,0.40,0.00,0.80,0,0,0
0.02,0.40,0.01,0.79,0,0,0
```

`t` may also be named `seconds_elapsed`. The trainer uses `x,y,z`; orientation is kept fixed by the generated path unless `--orientation` is supplied.

## Train and save a model

```bash
python3 motion/pickup/pickup_frisbee.py train \
  --demo-dir path/to/pickup_demos \
  --pattern '*.csv' \
  --n-gaussians 8 \
  --output motion/pickup/models/pickup_seds_g8.npz
```

The model is a shared-attractor stable dynamical system with a diagonal GMM. Increasing `--n-gaussians` gives the learned field more local shape regions; it does not change the pickup target.

## Generate a floor pickup path

`--current` is the current gripper XYZ position. `--target-xy` is the detected disk position in the same `body_link` frame. `--floor-z` is the gripper height at the disk.

```bash
python3 motion/pickup/pickup_frisbee.py generate \
  --model motion/pickup/models/pickup_seds_g8.npz \
  --current 0.40 0.00 0.80 \
  --target-xy 0.65 0.15 \
  --floor-z 0.05 \
  --duration 4.0 \
  --output motion/pickup/pickup_path.csv
```

Or pass a detected camera pixel directly:

```bash
python3 motion/pickup/pickup_frisbee.py generate \
  --model motion/pickup/models/pickup_seds_g8.npz \
  --calibration motion/pickup/models/camera_floor.json \
  --current 0.40 0.00 0.80 \
  --target-pixel 320 240 \
  --floor-z 0.05 \
  --output motion/pickup/pickup_path.csv
```

The output is compatible with the existing Cartesian follower:

```bash
ros2 run panther_examples cartesian_follow \
  --arm right \
  --path motion/pickup/pickup_path.csv \
  --dry-run
```

Always dry-run first. The follower performs URDF-based IK, joint-limit checks, and speed checks before commanding the robot. Closing the gripper is a separate controller command after the end pose is reached.

## Visualize outside ROS/Docker

```bash
MPLBACKEND=TkAgg python3 motion/pickup/visualize_pickup.py \
  --demo-dir path/to/pickup_demos \
  --model motion/pickup/models/pickup_seds_g8.npz \
  --current 0.40 0.00 0.80 \
  --target-xy 0.65 0.15 \
  --floor-z 0.05
```

For a headless preview, add `--save /tmp/pickup.png` and use `MPLBACKEND=Agg`.
