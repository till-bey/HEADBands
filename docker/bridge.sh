#!/usr/bin/env bash
# Connect this ROS 2 container to the cloud voice & robotics interface on GCP.
# Run it through docker/run.sh so ROS_DOMAIN_ID / Cyclone DDS are set for the robot (or --sim):
#
#   export ROBOT_BRIDGE_SHARED_TOKEN="$(gcloud secrets versions access latest \
#       --secret=robot-bridge-shared-token --project=robotcarnival26brn-606)"
#   docker/run.sh --build docker/bridge.sh     # robot, on the Ethernet cable (domain 15)
#   docker/run.sh --sim docker/bridge.sh       # mock sim on this machine (domain 77)
#
# The bridge only dials out (WebSocket), so no port is opened. Arm motions (throw, gestures) switch the
# arm trajectory controllers in themselves, as master_gui does; do not run master_gui at the same time.
# Optional: PANTHER_HEAD_FULL_RANGE=1 (after checking the head order), PANTHER_THROW_SPEED (1 = the CSV's timing,
# 2 = twice as fast), ROBOT_GATEWAY_WS_URL, ROBOT_ID.
set -euo pipefail

: "${ROBOT_BRIDGE_SHARED_TOKEN:?Set ROBOT_BRIDGE_SHARED_TOKEN on the host first (see the comment at the top).}"
export ROBOT_GATEWAY_WS_URL="${ROBOT_GATEWAY_WS_URL:-wss://swiss-voice-backend-1060412536965.europe-west6.run.app/ws/robot-bridge}"
export PYTHONPATH="/workspace/interface/robot_bridge/ros2_ws/src/swiss_robot_bridge${PYTHONPATH:+:${PYTHONPATH}}"

echo "Robot bridge -> ${ROBOT_GATEWAY_WS_URL}"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-15}  head full range=${PANTHER_HEAD_FULL_RANGE:-1}  head order=${PANTHER_HEAD_ORDER:-pitch_yaw}"
exec python3 -m swiss_robot_bridge.bridge_node
