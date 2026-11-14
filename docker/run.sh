#!/usr/bin/env bash
# Start the Panther ROS 2 shell on the Ethernet cable.
# Usage: docker/run.sh [--build] [command...]
#   ROS_IFACE=enx... docker/run.sh    pin a specific NIC
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="headbands-ros:jazzy"
ROBOT_IP="192.168.100.3"

if [[ "${1:-}" == "--build" ]]; then
  shift
  docker build -f "${ROOT}/docker/Dockerfile" -t "${IMAGE}" "${ROOT}"
elif ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  docker build -f "${ROOT}/docker/Dockerfile" -t "${IMAGE}" "${ROOT}"
fi

if [[ -n "${ROS_IFACE:-}" ]]; then
  IFACE="${ROS_IFACE}"
else
  IFACE="$(ip -o route get "${ROBOT_IP}" | awk '{print $3; exit}')"
fi

if [[ -z "${IFACE}" ]]; then
  echo "No route to ${ROBOT_IP}. Plug in the robot cable or set ROS_IFACE." >&2
  exit 1
fi

if ! ip -4 -o addr show dev "${IFACE}" | grep -q '192\.168\.100\.'; then
  echo "Interface ${IFACE} is not on 192.168.100.0/24." >&2
  echo "Plug in the robot Ethernet cable, or set ROS_IFACE to that adapter." >&2
  exit 1
fi

CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"${IFACE}\"/></Interfaces></General></Domain></CycloneDDS>"

if [[ $# -eq 0 ]]; then
  set -- bash
fi

tty_args=(-i)
if [[ -t 0 && -t 1 ]]; then
  tty_args=(-it)
fi

echo "Cyclone DDS on ${IFACE}, ROS_DOMAIN_ID=15"
exec docker run --rm \
  "${tty_args[@]}" \
  --network host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e ROS_DOMAIN_ID=15 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e CYCLONEDDS_URI="${CYCLONEDDS_URI}" \
  -v "${ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE}" \
  "$@"
