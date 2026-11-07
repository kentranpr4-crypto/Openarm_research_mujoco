#!/usr/bin/env bash
# Idempotent, non-sudo setup for Ubuntu 22.04+ : private virtual environment + pinned dependencies + pinned upstream OpenArm model.
# It NEVER installs NVIDIA drivers, never touches system Python, never uses sudo. Re-running it is safe.
source "$(dirname "$0")/_common.sh"
echo "== repository: $REPO_ROOT"
command -v git >/dev/null || { echo "ERROR: git missing. Install it yourself:  sudo apt install git" >&2; exit 2; }
PYBIN="${PYTHON:-python3}"; command -v "$PYBIN" >/dev/null || { echo "ERROR: $PYBIN missing" >&2; exit 2; }
"$PYBIN" - <<'PY' || exit 2
import sys; v = sys.version_info
if not ((3, 10) <= v[:2] < (3, 14)): sys.exit(f"ERROR: Python {v.major}.{v.minor} is outside the tested range 3.10-3.13")
print("python", sys.version.split()[0])
PY
"$PYBIN" -c "import venv, ensurepip" 2>/dev/null || { echo "ERROR: python venv module missing. Install it yourself:  sudo apt install python3-venv" >&2; exit 2; }
if [ ! -x "$VENV/bin/python" ]; then echo "== creating $VENV"; "$PYBIN" -m venv "$VENV"; else echo "== reusing $VENV"; fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip
LOCK="$SM_DIR/requirements.lock.txt"; [ -f "$LOCK" ] || { echo "ERROR: $LOCK missing" >&2; exit 2; }
echo "== installing pinned dependencies (torch from PyPI = CUDA 12.8 wheels on Linux x86_64; large download, several GB)"
"$VENV/bin/python" -m pip install --quiet -r "$LOCK"
UP="$REPO_ROOT/sim/openarm_mujoco"; SHA="a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c"     # same pin as sim/setup.sh; upstream is never modified
if [ ! -d "$UP/.git" ]; then echo "== cloning upstream openarm_mujoco"; git clone --quiet https://github.com/enactic/openarm_mujoco.git "$UP"; fi
CUR="$(git -C "$UP" rev-parse HEAD)"
if [ "$CUR" != "$SHA" ]; then
  [ -z "$(git -C "$UP" status --porcelain)" ] || { echo "ERROR: $UP has local modifications; refusing to touch it" >&2; exit 2; }
  git -C "$UP" checkout --quiet "$SHA"
fi
echo "== upstream model at $(git -C "$UP" rev-parse --short HEAD)"
echo "== versions"; "$VENV/bin/python" - <<'PY'
import torch, numpy, mujoco, yaml
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda, "| cuda available:", torch.cuda.is_available()); print("numpy", numpy.__version__, "| mujoco", mujoco.__version__)
PY
"$SM_DIR/scripts/check_gpu.sh" || echo "(GPU check failed: CPU debugging still works; see the message above)"
echo "== done. Next: research/supermimic_openarm/scripts/smoke_test.sh"
