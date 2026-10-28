#!/usr/bin/env python3
"""Self-contained checks (plain asserts, no pytest) of the action/state contract. Run: sim/.venv/bin/python control/test_action_adapter.py"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, numpy as np, mujoco
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path: sys.path.insert(0, p)
import lab_scene
from control.action_adapter import ActionAdapter

m, d = lab_scene.compile_lab(); ad = ActionAdapter(m, 20); ok = []
def check(name, cond, extra=""): ok.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name, extra)

# 1. index mapping: state index i is the joint driven by actuator i; it is NOT qpos[i] from the right arm onward
naive_equal = [bool(ad.qadr[i] == i) for i in range(16)]
check("ctrl index != qpos index for the right arm (audit claim)", not all(naive_equal) and all(naive_equal[:8]), f"qadr={ad.qadr.tolist()}")
# 2. semantics: ctrl is an absolute joint-position target for ITS joint (apply +0.1 rad to right_joint1 -> that joint moves)
for name in ("left_joint1_ctrl", "right_joint1_ctrl", "right_joint4_ctrl"):
    i = ad.names.index(name); lab_scene.reset_home(m, d); s0 = ad.state(d).copy(); c = d.ctrl.copy(); c[i] = np.clip(c[i] + 0.1, *m.actuator_ctrlrange[i])
    d.ctrl[:] = c
    for _ in range(1500): mujoco.mj_step(m, d)
    moved = np.abs(ad.state(d) - s0)
    check(f"{name}: state[{i}] follows its own ctrl (moved {moved[i]:.3f} rad, largest mover idx {int(np.argmax(moved))})", int(np.argmax(moved)) == i and moved[i] > 0.05)
# 3. gripper unit is metres (v1) and finger2 is excluded from state
check("gripper ctrlrange is metres-scale (v1)", all(abs(m.actuator_ctrlrange[i]).max() < 0.1 for i in ad.gripper_idx), f"{[m.actuator_ctrlrange[i].tolist() for i in ad.gripper_idx]}")
check("state has 16 entries (finger_joint2 dropped)", ad.state(d).shape == (16,) and m.nq - 21 == 18)
# 4. safety: shape/NaN rejected, range clip, slew limit
prev = d.ctrl.copy(); bad = 0
for a in (np.zeros(15), np.full(16, np.nan), np.full(16, np.inf)):
    try: ad.to_ctrl(a, prev)
    except ValueError: bad += 1
check("bad shape / NaN / Inf rejected", bad == 3)
big = prev + 10.0; c, nr, ns = ad.to_ctrl(big, prev)
check("clipped to ctrlrange and slew-limited", np.all(c <= m.actuator_ctrlrange[:, 1] + 1e-12) and np.all(np.abs(c - prev) <= ad.limits.max_speed * ad.dt + 1e-12), f"n_range={nr} n_slew={ns}")
c2, _, _ = ad.to_ctrl(prev, prev); check("identity within limits", np.allclose(c2, prev))
print("ALL PASS" if all(ok) else "SOME FAILED"); sys.exit(0 if all(ok) else 1)
