# Sourced by the container entrypoint and by interactive shells.
# shellcheck shell=bash

# setup.bash chains /opt/ros/humble. Colcon does not apply ament's
# environment hooks, so overlays are missing from AMENT_PREFIX_PATH
# and `ros2 run` reports the package as not found.
# shellcheck disable=SC1091
source /opt/unix_ws/setup.bash
# Skip a workspace that was built for Jazzy. Rebuild it inside this container.
if [ -f /workspace/install/setup.bash ] && ! grep -q '/opt/ros/jazzy' /workspace/install/setup.bash; then
  # shellcheck disable=SC1091
  source /workspace/install/setup.bash
fi
# shellcheck disable=SC1091
source /workspace/docker/ament_prefix.sh
