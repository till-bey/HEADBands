# Colcon's setup.bash does not put these installs on AMENT_PREFIX_PATH.
# Source this after install/setup.bash. Safe to source more than once.

_prepend_ament_prefix() {
  case ":${AMENT_PREFIX_PATH:-}:" in
    *":$1:"*) ;;
    *) AMENT_PREFIX_PATH="$1${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}" ;;
  esac
}

_register_prefix_tree() {
  [ -d "$1" ] || return 0
  if [ -d "$1/share/ament_index" ]; then
    _prepend_ament_prefix "$1"
  fi
  for _pkg in "$1"/*; do
    if [ -d "$_pkg/share/ament_index" ]; then
      _prepend_ament_prefix "$_pkg"
    fi
  done
}

_register_prefix_tree /opt/unix_ws
_register_prefix_tree /workspace/install
IFS=':'
for _prefix in ${COLCON_PREFIX_PATH:-}; do
  [ -n "$_prefix" ] || continue
  _register_prefix_tree "$_prefix"
done
unset IFS
export AMENT_PREFIX_PATH

unset -f _prepend_ament_prefix _register_prefix_tree
