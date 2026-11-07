# sourced by the other scripts: resolves the repository root and the Python interpreter. No sudo, no system-wide changes.
set -euo pipefail
SM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; REPO_ROOT="$(cd "$SM_DIR/../.." && pwd)"; cd "$REPO_ROOT"
VENV="${SM_VENV:-$REPO_ROOT/.venv-supermimic}"
py() { if [ -x "$VENV/bin/python" ]; then "$VENV/bin/python" "$@"; else echo "ERROR: virtual environment $VENV not found. Run research/supermimic_openarm/scripts/setup_ubuntu.sh first." >&2; return 2; fi; }
