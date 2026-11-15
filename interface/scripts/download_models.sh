#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/download_models.py" --dialects vs zh be bs sg --acknowledge-cc-by-nc-4.0 "$@"
