#!/usr/bin/env bash
# Starts a pushing run from configs/pushing_4090.yaml.  Extra arguments are passed through (e.g. --override imagination.horizon=32, --resume RUN_DIR).
source "$(dirname "$0")/_common.sh"
if ! ls outputs/supermimic_openarm/reaching_4090/seed*/COMPLETED >/dev/null 2>&1 && [ "${ALLOW_SKIP_GATE:-0}" != 1 ]; then echo "GATE: no COMPLETED reaching_4090 run found. Finish and evaluate the reaching pilot first (README, phase 8) or set ALLOW_SKIP_GATE=1 knowingly." >&2; exit 3; fi
exec "$VENV/bin/python" -m research.supermimic_openarm.train --config research/supermimic_openarm/configs/pushing_4090.yaml "$@"
