#!/usr/bin/env bash
# Usage: evaluate.sh CHECKPOINT TASK [extra evaluate args]   e.g. evaluate.sh outputs/supermimic_openarm/reaching_4090/seed1000/checkpoints/best.pt reaching --episodes 64
source "$(dirname "$0")/_common.sh"
[ $# -ge 2 ] || { echo "usage: $0 CHECKPOINT TASK [args]" >&2; exit 2; }
CK="$1"; TASK="$2"; shift 2; exec "$VENV/bin/python" -m research.supermimic_openarm.evaluate --checkpoint "$CK" --task "$TASK" "$@"
