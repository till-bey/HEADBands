#!/bin/bash
set -e

# setup.bash chains /opt/ros/jazzy. Colcon does not apply ament's
# environment hooks, so overlays are missing from AMENT_PREFIX_PATH
# and `ros2 run` reports the package as not found.
source /opt/unix_ws/setup.bash
if [ -f /workspace/install/setup.bash ]; then
  # shellcheck disable=SC1091
  source /workspace/install/setup.bash
fi

_prepend_ament_prefix() {
  case ":${AMENT_PREFIX_PATH:-}:" in
    *":$1:"*) ;;
    *) AMENT_PREFIX_PATH="$1${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}" ;;
  esac
}

_prepend_ament_prefix /opt/unix_ws
IFS=':'
for _prefix in ${COLCON_PREFIX_PATH:-}; do
  [ -n "$_prefix" ] || continue
  if [ -d "$_prefix/share/ament_index" ]; then
    _prepend_ament_prefix "$_prefix"
  fi
  for _pkg in "$_prefix"/*; do
    if [ -d "$_pkg/share/ament_index" ]; then
      _prepend_ament_prefix "$_pkg"
    fi
  done
done
unset IFS
export AMENT_PREFIX_PATH

exec "$@"
