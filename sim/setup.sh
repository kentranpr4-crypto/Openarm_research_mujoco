#!/usr/bin/env bash
# Dựng môi trường mô phỏng: clone openarm_mujoco (Apache-2.0, commit đã kiểm) + venv MuJoCo.
set -euo pipefail
cd "$(dirname "$0")"
[ -d openarm_mujoco ] || git clone https://github.com/enactic/openarm_mujoco.git
git -C openarm_mujoco checkout -q a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c
uv venv -q .venv 2>/dev/null || python3 -m venv .venv
(uv pip install -q -p .venv/bin/python "mujoco>=3.6.0" numpy imageio imageio-ffmpeg) || .venv/bin/pip install -q "mujoco>=3.6.0" numpy imageio imageio-ffmpeg
echo "OK. Chạy: .venv/bin/python run_sim.py ; .venv/bin/python realsense_cam.py"
