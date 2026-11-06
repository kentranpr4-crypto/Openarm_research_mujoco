#!/usr/bin/env bash
# Start OpenArm Research Studio (read-only).  Usage:  app/run_studio.sh [--data-root DIR] [--port N] [--no-render] [--bundle-only]
# PYTHON overrides the interpreter; by default the virtual environment .venv-studio in the repository root is used when it exists.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x .venv-studio/bin/python ] && PY=.venv-studio/bin/python || PY=python3; }
exec "$PY" -m app.backend.server "$@"
