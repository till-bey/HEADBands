# Sourced by the container entrypoint and by interactive shells.
# shellcheck shell=bash

# shellcheck disable=SC1091
source /opt/unix_ws/setup.bash
# Skip a workspace that was built for Jazzy. Rebuild it inside this container.
if [ -f /workspace/install/setup.bash ] && ! grep -q '/opt/ros/jazzy' /workspace/install/setup.bash; then
  # shellcheck disable=SC1091
  source /workspace/install/setup.bash
fi
