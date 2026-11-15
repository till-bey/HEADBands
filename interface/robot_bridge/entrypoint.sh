#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash
if [ -f /ros2_ws/install/setup.bash ]; then
  # shellcheck disable=SC1091
  source /ros2_ws/install/setup.bash
fi

_prepend_ament_prefix() {
  case ":${AMENT_PREFIX_PATH:-}:" in
    *":$1:"*) ;;
    *) AMENT_PREFIX_PATH="$1${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}" ;;
  esac
}

if [ -d /ros2_ws/install ]; then
  _prepend_ament_prefix /ros2_ws/install
fi
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
