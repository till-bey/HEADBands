#!/bin/bash
set -e

# shellcheck disable=SC1091
source /workspace/docker/shell.sh

exec "$@"
