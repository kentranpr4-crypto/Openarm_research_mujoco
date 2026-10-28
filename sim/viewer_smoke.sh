#!/usr/bin/env bash
# Real GUI smoke test of the exported lab scene: Xvfb (virtual display) + software GL + the MuJoCo viewer module,
# then a screenshot of the X root window. Proves the viewer opens and renders the exported MJCF; it is NOT a human
# usability test (no mouse/keyboard interaction). Usage: bash viewer_smoke.sh [out.png] [xml]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="${1:-$HERE/../outputs/lab_scene/viewer_xvfb.png}"; XML="${2:-$HERE/../outputs/lab_scene/lab_scene.xml}"
D=:98; PAT="mujoco[.]viewer"      # bracket trick: the pattern does not match this script's own command line
pkill -f "Xvfb $D" 2>/dev/null; rm -f /tmp/.X11-unix/X98 /tmp/.X98-lock
(Xvfb $D -screen 0 1600x1000x24 +extension GLX >/dev/null 2>&1 &)
n=0; until [ -S /tmp/.X11-unix/X98 ] || [ $n -gt 20 ]; do sleep 0.5; n=$((n+1)); done
(DISPLAY=$D LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=glfw timeout 150 "$HERE/.venv/bin/python" -m mujoco.viewer --mjcf="$XML" >/tmp/viewer_smoke.log 2>&1 &)
n=0; until DISPLAY=$D xdotool search --name "MuJoCo" >/dev/null 2>&1 || [ $n -gt 60 ]; do sleep 1; n=$((n+1)); done
[ $n -gt 60 ] && { echo "FAIL: no MuJoCo window"; cat /tmp/viewer_smoke.log; exit 1; }
c=0
for i in $(seq 1 45); do      # wait until the 3D viewport shows real content (>500 distinct colours), up to 90 s
  sleep 2; DISPLAY=$D timeout 20 import -window root "$OUT" 2>/dev/null
  c=$(python3 -c "from PIL import Image;import numpy as np;a=np.asarray(Image.open('$OUT').convert('RGB'))[0:660,300:1060];print(len(np.unique(a.reshape(-1,3),axis=0)))")
  [ "$c" -gt 500 ] && break
done
sleep 4; DISPLAY=$D timeout 20 import -window root "$OUT" 2>/dev/null
pkill -f "$PAT"; pkill -f "Xvfb $D"
echo "window opened; viewport distinct colours=$c; screenshot=$OUT"; [ "$c" -gt 500 ]
