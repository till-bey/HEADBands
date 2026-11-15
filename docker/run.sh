#!/usr/bin/env bash
# Start the Panther ROS 2 shell, on the robot's Ethernet cable or (--sim) on the local mock sim.
# Usage: docker/run.sh [--build] [--sim] [command...]
#   ROS_IFACE=enx... docker/run.sh    pin a specific NIC
#   docker/run.sh --sim               domain 77, localhost only: matches mock_sim.launch.py, never the robot
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="headbands-ros:humble"
ROBOT_IP="192.168.100.3"

BUILD=0
SIM=0
while [[ "${1:-}" == --build || "${1:-}" == --sim ]]; do
  [[ "$1" == --build ]] && BUILD=1 || SIM=1
  shift
done

if [[ ${BUILD} -eq 1 ]] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  docker build -f "${ROOT}/docker/Dockerfile" -t "${IMAGE}" "${ROOT}"
fi

if [[ ${SIM} -eq 1 ]]; then
  # Same as SIM_CYCLONE in mock_sim.launch.py: loopback only, multicast forced on.
  ros_env=(-e ROS_DOMAIN_ID=77
           -e CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" multicast="true"/></Interfaces></General></Domain></CycloneDDS>')
  echo "Sim mode: ROS_DOMAIN_ID=77, localhost only"
else
  IFACE="${ROS_IFACE:-$(ip -o route get "${ROBOT_IP}" | awk '{print $3; exit}')}"
  if [[ -z "${IFACE}" ]]; then
    echo "No route to ${ROBOT_IP}. Plug in the robot cable or set ROS_IFACE." >&2
    exit 1
  fi
  if ! ip -4 -o addr show dev "${IFACE}" | grep -q '192\.168\.100\.'; then
    echo "Interface ${IFACE} is not on 192.168.100.0/24." >&2
    echo "Plug in the robot Ethernet cable, or set ROS_IFACE to that adapter." >&2
    exit 1
  fi
  ros_env=(-e ROS_DOMAIN_ID=15
           -e CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"${IFACE}\"/></Interfaces></General></Domain></CycloneDDS>")
  echo "Cyclone DDS on ${IFACE}, ROS_DOMAIN_ID=15"
fi

if [[ $# -eq 0 ]]; then
  set -- bash
fi

tty_args=(-i)
if [[ -t 0 && -t 1 ]]; then
  tty_args=(-it)
fi

# The robot's /robot_description loads meshes from its own install path: mount a local copy there for RViz.
mesh_args=()
if [[ -d "${HOME}/mh3_description" ]]; then
  mesh_args=(-v "${HOME}/mh3_description:/home/unix_ai/unixai_ws/install/mh3_description/share/mh3_description:ro")
fi

exec docker run --rm \
  "${tty_args[@]}" \
  --network host \
  "${ros_env[@]}" \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e DISPLAY="${DISPLAY:-:0}" \
  -v "${ROOT}:/workspace" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  "${mesh_args[@]}" \
  -w /workspace \
  "${IMAGE}" \
  "$@"
