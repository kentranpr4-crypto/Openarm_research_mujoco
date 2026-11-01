#!/usr/bin/env python3
"""Stage 0.1 analysis (pure numpy on outputs/closed_loop_design/stage0_1_chunks.npz).  Run: python3 eval/act_clip_analysis.py"""
import json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; D = ROOT / "outputs/closed_loop_design"
Z = np.load(D / "stage0_1_chunks.npz"); M = json.load(open(D / "stage0_1_chunks_meta.json")); raw, prev0, state, lo, hi, vmax = Z["raw"].astype(float), Z["prev_exec"].astype(float), Z["state"], Z["lo"], Z["hi"], Z["max_speed"]
meta, ck, act = M["meta"], M["ckpts"], M["actuators"]; NP, NC, T, NA = raw.shape; DT = 0.05
clipped = np.zeros(raw.shape, bool); slewed = np.zeros(raw.shape, bool); exe = np.zeros(raw.shape); side = np.zeros(raw.shape, int)       # side: -1 lower, +1 upper
for p in range(NP):
    for c in range(NC):
        prev = prev0[p].copy()
        for t in range(T):
            r = raw[p, c, t]; cl = np.clip(r, lo, hi); clipped[p, c, t] = cl != r; side[p, c, t] = np.where(r < lo, -1, np.where(r > hi, 1, 0))
            step = np.clip(cl - prev, -vmax * DT, vmax * DT); slewed[p, c, t] = step != cl - prev; prev = prev + step; exe[p, c, t] = prev
out = dict(n_points=NP, n_ckpt=NC, steps=T, actuators=act, total_values=int(raw.size))
out["ctrlrange_clips_total"] = int(clipped.sum()); out["slew_events_total"] = int(slewed.sum())
out["clips_by_actuator"] = {act[a]: dict(n=int(clipped[..., a].sum()), lower=int((side[..., a] == -1).sum()), upper=int((side[..., a] == 1).sum()), max_exceed=float(np.max(np.where(raw[..., a] < lo[a], lo[a] - raw[..., a], np.where(raw[..., a] > hi[a], raw[..., a] - hi[a], 0))))) for a in range(NA) if clipped[..., a].any()}
out["ctrlrange"] = {act[a]: [float(lo[a]), float(hi[a])] for a in range(NA)}
out["clips_by_ckpt"] = {k: int(clipped[:, i].sum()) for i, k in enumerate(ck)}; out["clips_by_phase"] = {g: int(clipped[[i for i, m in enumerate(meta) if m["phase_group"] == g]].sum()) for g in ("approach", "grasp", "lift", "place")}
out["clips_left_arm_dims0_7"] = int(clipped[..., :8].sum()); out["clips_right_arm_dims8_15"] = int(clipped[..., 8:].sum())
out["slews_by_actuator"] = {act[a]: int(slewed[..., a].sum()) for a in range(NA) if slewed[..., a].any()}
out["slews_by_phase"] = {g: int(slewed[[i for i, m in enumerate(meta) if m["phase_group"] == g]].sum()) for g in ("approach", "grasp", "lift", "place")}; out["slews_by_ckpt"] = {k: int(slewed[:, i].sum()) for i, k in enumerate(ck)}
out["slew_step0_events"] = int(slewed[:, :, 0].sum()); out["slew_after_step0_events"] = int(slewed[:, :, 1:].sum())
dv = np.abs(exe - raw); out["raw_vs_exec_abs_by_actuator"] = {act[a]: dict(mean=float(dv[..., a].mean()), max=float(dv[..., a].max()), unit="m" if "finger" in act[a] else "rad") for a in range(NA)}
# persistent saturation: per (point, ckpt, actuator) fraction of steps clipped
frac = clipped.mean(2); pers = [(meta[p]["phase_group"], meta[p]["pose"][0], ck[c], act[a], float(frac[p, c, a])) for p in range(NP) for c in range(NC) for a in range(NA) if frac[p, c, a] >= 0.5]
out["persistent_saturation_ge50pct_of_chunk"] = dict(n=len(pers), by_actuator={a: sum(1 for q in pers if q[3] == a) for a in sorted({q[3] for q in pers})}, all_20_of_20=int((frac == 1.0).sum()), examples=pers[:8])
# diversity before / after: left arm (dims 0..6 arm, 7 gripper), clip-only (range) and executed
rclip = np.clip(raw, lo, hi)
def stats(X, dims, idx):
    v = [np.abs(X[p, i][:, dims] - X[p, j][:, dims]).max() for p in range(NP) for ii, i in enumerate(idx) for j in idx[ii + 1:]]; return dict(median=float(np.median(v)), mean=float(np.mean(v)), p10=float(np.percentile(v, 10)))
groups = dict(c0_only=[0, 1, 2], s1_only=[3, 4, 5], mixed=list(range(6)))
out["diversity_linf"] = {g: {"left_arm_raw": stats(raw, list(range(7)), i), "left_arm_clipped_only": stats(rclip, list(range(7)), i), "left_arm_executed": stats(exe, list(range(7)), i), "gripper_raw": stats(raw, [7], i), "gripper_executed": stats(exe, [7], i),
                          "right_arm_raw": stats(raw, list(range(8, 15)), i), "right_arm_executed": stats(exe, list(range(8, 15)), i)} for g, i in groups.items()}
THR_ARM, THR_GRIP = 0.02, 0.005
def ndist(X, idx, p):
    reps = []
    for n in idx:
        if not any(np.abs(X[p, n][:, :7] - X[p, r][:, :7]).max() < THR_ARM and np.abs(X[p, n][:, 7] - X[p, r][:, 7]).max() < THR_GRIP for r in reps): reps.append(n)
    return len(reps)
out["n_distinct"] = {g: dict(raw_mean=float(np.mean([ndist(raw, i, p) for p in range(NP)])), executed_mean=float(np.mean([ndist(exe, i, p) for p in range(NP)])), executed_min=int(min(ndist(exe, i, p) for p in range(NP)))) for g, i in groups.items()}
col = 0; pairs = 0
for p in range(NP):
    for i in range(NC):
        for j in range(i + 1, NC):
            pairs += 1
            if (np.abs(raw[p, i][:, :7] - raw[p, j][:, :7]).max() >= THR_ARM) and (np.abs(exe[p, i][:, :7] - exe[p, j][:, :7]).max() < THR_ARM): col += 1
out["pairs_collapsed_after_clip_slew"] = dict(n=col, of=pairs)
# idempotence: re-applying the chain to executed gives itself (executed actions are legal, so WM sees exactly what the servo would receive)
ok = True
for p in range(NP):
    for c in range(NC):
        prev = prev0[p].copy()
        for t in range(T):
            cl = np.clip(exe[p, c, t], lo, hi); st = np.clip(cl - prev, -vmax * DT, vmax * DT); ok &= bool(np.allclose(prev + st, exe[p, c, t], atol=1e-9)); prev = exe[p, c, t]
out["executed_chain_idempotent"] = ok
out["max_exec_rate_over_limit"] = float(max(np.abs(np.diff(np.concatenate([prev0[p][None], exe[p, c]], 0), axis=0)).max(0).__truediv__(DT * vmax).max() for p in range(NP) for c in range(NC)))
# initial conditions: measured joint state vs previous command; first raw target vs previous command
dq = np.abs(state - prev0); out["state_minus_prevcmd_abs"] = {act[a]: float(dq[:, a].max()) for a in range(NA)}
out["first_raw_target_minus_prevcmd_abs_max"] = {act[a]: float(np.abs(raw[:, :, 0, a] - prev0[:, None, a]).max()) for a in range(NA)}
out["first_step_slew_limit"] = {act[a]: float(vmax[a] * DT) for a in range(NA)}
(D / "stage0_1_clip_audit.json").write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k not in ("raw_vs_exec_abs_by_actuator", "state_minus_prevcmd_abs", "first_raw_target_minus_prevcmd_abs_max", "first_step_slew_limit", "ctrlrange")}, indent=1))
print("RAW-EXEC", json.dumps(out["raw_vs_exec_abs_by_actuator"])); print("RANGE", json.dumps(out["ctrlrange"])); print("DQ", json.dumps(out["state_minus_prevcmd_abs"])); print("FIRST", json.dumps(out["first_raw_target_minus_prevcmd_abs_max"]))
