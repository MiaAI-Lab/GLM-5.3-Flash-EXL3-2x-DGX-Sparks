#!/usr/bin/env bash
# Serve the pre-edited EXL3 checkpoint with the regular TP=2 launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GLM53_MODEL_PRESET=abliterated
exec "$SCRIPT_DIR/start.sh" "$@"
