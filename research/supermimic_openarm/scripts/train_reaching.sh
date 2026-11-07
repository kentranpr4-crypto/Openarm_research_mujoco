#!/usr/bin/env bash
# Starts a reaching run from configs/reaching_4090.yaml.  Extra arguments are passed through (e.g. --override imagination.horizon=32, --resume RUN_DIR).
source "$(dirname "$0")/_common.sh"
[ -x "$VENV/bin/python" ] || { echo "ERROR: run scripts/setup_ubuntu.sh first" >&2; exit 2; }
exec "$VENV/bin/python" -m research.supermimic_openarm.train --config research/supermimic_openarm/configs/reaching_4090.yaml "$@"
