#!/usr/bin/env bash
# Frisbee interception CI, CPU only (Cloud Build or any ROS 2 Humble container), run from the repo root:
#   docker run --rm -v "$PWD":/workspace -w /workspace osrf/ros:humble-desktop bash src/frisbee_interception/ci/run_ci.sh
# Outputs (logs, reports, annotated replays) go to ci_out/.
# (no `set -u`: the ROS setup scripts read unset variables)
OUT="$PWD/ci_out"
mkdir -p "$OUT"
exec > >(tee "$OUT/ci.log") 2>&1
FAIL=0
step() { echo; echo "=================== $*"; }

step "dependencies"
apt-get update -qq && apt-get install -y -qq --no-install-recommends python3-opencv python3-numpy python3-yaml \
    python3-pytest python3-pykdl ros-humble-tf2-ros ros-humble-rmw-cyclonedds-cpp \
    ros-humble-control-msgs ros-humble-controller-manager-msgs >/dev/null || FAIL=1
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=42 PYTHONUNBUFFERED=1
export FRISBEE_TEST_URDF="$PWD/src/panther_sim/urdf/robot_description.urdf"
python3 -c "import cv2, numpy; print('opencv', cv2.__version__, 'numpy', numpy.__version__)"

step "colcon build"
WS=/tmp/frisbee_ws
mkdir -p $WS/src
cp -r src/unix_msgs src/frisbee_msgs src/frisbee_interception $WS/src/
(cd $WS && colcon build --event-handlers console_direct+) || { echo "BUILD FAILED"; exit 1; }
source $WS/install/setup.bash

step "colcon test (tracking, prediction, catch trigger, catching pose, synthetic end-to-end)"
(cd $WS && colcon test --packages-select frisbee_interception --event-handlers console_direct+ \
    --pytest-args -q && colcon test-result --verbose) || FAIL=1
cp -r $WS/build/frisbee_interception/test_results "$OUT/" 2>/dev/null || true

step "interfaces"
ros2 interface show frisbee_msgs/msg/FrisbeeIntercept

step "catching pose from the URDF (frisbee_catch_pose) + FK cross-check against panther_examples' PyKDL Chain"
ros2 run frisbee_interception frisbee_catch_pose --urdf "$FRISBEE_TEST_URDF" --out "$OUT/catch_pose" || FAIL=1
PYTHONPATH="$PWD/src/panther_examples:$PYTHONPATH" python3 src/frisbee_interception/ci/fk_crosscheck.py \
    "$FRISBEE_TEST_URDF" "$OUT/catch_pose/catch_pose.json" || FAIL=1

step "synthetic inputs"
python3 src/frisbee_interception/ci/make_ci_inputs.py "$OUT/synthetic" \
    "$(ros2 pkg prefix frisbee_interception)/share/frisbee_interception/config/frisbee.yaml" \
    "$OUT/catch_pose/catch_pose.json" || FAIL=1

run_launch() {  # mode extra_launch_args...
    local mode=$1; shift
    step "ROS launch smoke test: $mode"
    local tfpid="" mockpid=""
    if [ "$mode" = metric ]; then
        # robot stand-in (robot_description, joint_states, controller_manager, trajectory actions) + camera TF
        python3 src/frisbee_interception/ci/mock_robot.py "$FRISBEE_TEST_URDF" "$OUT/catch_pose/catch_pose.json" \
            > "$OUT/mock_robot.log" 2>&1 &
        mockpid=$!
        ros2 run tf2_ros static_transform_publisher --x -0.55 --y 0.219 --z -0.348 --roll -1.5707963 --pitch 0 \
            --yaw -1.5707963 --frame-id body_link --child-frame-id phone_camera_optical_frame \
            > "$OUT/tf_$mode.log" 2>&1 &
        tfpid=$!
    fi
    ros2 launch frisbee_interception frisbee.launch.py source:="$OUT/synthetic/throw.mp4" loop:=true \
        config:="$OUT/synthetic/cfg_$mode.yaml" "$@" > "$OUT/launch_$mode.log" 2>&1 &
    local pid=$!
    python3 src/frisbee_interception/ci/ros_smoke_check.py --mode "$mode" --truth "$OUT/synthetic/throw_truth.json" \
        --truth-body "$OUT/synthetic/truth_body.json" --urdf "$FRISBEE_TEST_URDF" \
        --duration 25 --report "$OUT/smoke_$mode.json" || FAIL=1
    kill -INT $pid $tfpid $mockpid 2>/dev/null; sleep 3; kill -9 $pid $tfpid $mockpid 2>/dev/null
    pkill -f frisbee_interception/lib 2>/dev/null; pkill -f static_transform_publisher 2>/dev/null
    pkill -f mock_robot.py 2>/dev/null; sleep 1
    tail -n 30 "$OUT/launch_$mode.log"
}
run_launch image
run_launch metric

step "offline replay of the recorded clips (prediction needs calibration; this checks detection/tracking)"
mkdir -p "$OUT/replay"
for v in perception/Dataset/Videos/*.mp4 "$OUT/synthetic/throw.mp4"; do
    [ -f "$v" ] || continue
    name=$(basename "$v" .mp4)
    cfg="$(ros2 pkg prefix frisbee_interception)/share/frisbee_interception/config/frisbee.yaml"
    [ "$name" = throw ] && cfg="$OUT/synthetic/cfg_image.yaml"
    echo "--- $name"
    python3 -m frisbee_interception.offline "$v" --config "$cfg" --out "$OUT/replay/$name.mp4" \
        --json "$OUT/replay/$name.json" || FAIL=1
done

step "summary"
python3 - "$OUT" <<'EOF'
import json, sys, glob, os
out = sys.argv[1]
s = {}
for f in sorted(glob.glob(os.path.join(out, "smoke_*.json"))):
    r = json.load(open(f)); s[os.path.basename(f)] = {"passed": r["passed"], "checks": r["checks"]}
for f in sorted(glob.glob(os.path.join(out, "replay", "*.json"))):
    s[os.path.basename(f)] = json.load(open(f))["stats"]
json.dump(s, open(os.path.join(out, "summary.json"), "w"), indent=1)
print(json.dumps(s, indent=1))
EOF
echo "CI result: $([ $FAIL = 0 ] && echo PASS || echo FAIL)"
echo $FAIL > "$OUT/exit_code"
exit 0  # always upload artifacts; the next build step fails the build on a non-zero exit_code
