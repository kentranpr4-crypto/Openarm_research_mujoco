#!/usr/bin/env python3
"""ACT V1 research audit: reads RAW logs only (rollout .npz, episodes.jsonl, dataset npz, checkpoint tensors) and writes
outputs/audit_v1/{audit_results.json, *.png}. Ground truth (box/TCP pose) is used for DIAGNOSTICS, never as policy input.
Run (ACT venv, numpy+scipy+matplotlib+safetensors): python eval/audit_v1.py [--rollouts outputs/eval/audit_v1]
Sections: A1 metrics re-computation | A2 failure timelines + taxonomy | A3 chunk boundaries & jerk | A4 leakage |
          A5 pose distribution | A6 localisation (grasp point vs box) | A7 image-space probe | A8 vision ablation + inference pilots."""
import sys, json, argparse, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as st
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT),):
    if p not in sys.path: sys.path.insert(0, p)
from eval.metrics import task_labels, PICK_LIFT_M, HOLD_DIST_M, PLACE_XY_TOL, PLACE_Z_TOL, REST_SPEED, smoothness

HZ = 20; TRAY = np.array([0.44, 0.30]); TRAY_BASE_TOP = 0.36 + 0.004; OPEN, CLOSE_THR = 0.044, 0.028
TRAIN_BOX = dict(x=(0.38, 0.45), y=(0.07, 0.13), yaw=(-0.20, 0.20)); SCALE = np.array([0.07, 0.06, 0.40])   # normalisation = train-box widths
ARM = [i for i in range(16) if i not in (7, 15)]
OUT = ROOT / "outputs/audit_v1"; OUT.mkdir(parents=True, exist_ok=True); R = {}

def jl(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def load_run(root, agent, seed, cond):
    d = Path(root) / f"seed{seed}" / agent / cond
    if not (d / "episodes.jsonl").exists(): return []
    out = []
    for e in jl(d / "episodes.jsonl"):
        f = d / "rollouts" / f"ep{e['episode_index']:04d}.npz"
        if f.exists():
            with np.load(f) as z: e = dict(e, **{k: z[k] for k in z.files})
        out.append(dict(e, seed=seed, cond=cond, agent=agent))
    return out

def events(e):
    """Per-episode diagnostics from raw arrays (GT only for diagnostics)."""
    a, B, T = e["action"], e["box_pos"], e["tcp_pos"]; yaw = e["box_yaw"]; f = np.array([np.cos(yaw), np.sin(yaw)]); n = np.array([-np.sin(yaw), np.cos(yaw)])
    g = a[:, 7]; closing = np.nonzero(g < CLOSE_THR)[0]; t_close = int(closing[0]) if len(closing) else None
    rise = B[:, 2] - B[0, 2]; dist = np.linalg.norm(B[:len(T)] - T, axis=1); dxy = np.linalg.norm(B[:len(T), :2] - T[:, :2], axis=1)
    held = (dist < HOLD_DIST_M) & (rise[:len(T)] > 0.03); n_close = int(np.sum((g[1:] < CLOSE_THR) & (g[:-1] >= CLOSE_THR)))
    d = dict(t_close=t_close, n_close_events=n_close, max_rise=float(rise.max()), held_steps=int(held.sum()), min_xy_dist=float(dxy.min()), t_min_xy=int(dxy.argmin()),
             tcp_min_xy_to_tray=float(np.linalg.norm(T[:, :2] - TRAY, axis=1).min()), final_box_to_tray=float(np.linalg.norm(B[-1, :2] - TRAY)),
             box_xy_shift_before_close=float(np.linalg.norm(B[t_close, :2] - B[0, :2])) if t_close is not None else None)
    if t_close is not None:
        off = T[t_close, :2] - B[t_close, :2]; d.update(lateral=float(off @ n), longitudinal=float(off @ f), dz=float(T[t_close, 2] - B[t_close, 2]), grasp_tcp=T[t_close].tolist(),
                                                           track_err_at_close=float(np.abs(a[t_close, ARM] - e["state"][t_close, ARM]).mean()))
        d["finger_box_vertical_overlap_cm"] = float(100 * (0.020 + 0.031 - d["dz"]))     # finger pad half-height 3.1 cm, box half-height 2.0 cm; expert: 2.6 cm
        d["flags"] = dict(pushed=bool(d["box_xy_shift_before_close"] > 0.03), lateral_miss=bool(abs(d["lateral"]) > 0.022), too_high=bool(d["dz"] > 0.034), long_miss=bool(abs(d["longitudinal"] - 0.01) > 0.045))
    return d

def classify(e, d):
    """Rule-based taxonomy (thresholds = finger geometry: inner gap half-width 0.043 m vs box half-width 0.02 m)."""
    if e["task_success"]: return "success"
    if d["t_close"] is None: return "approach: gripper never closed"
    if e["pick_success"]:                                    # box was lifted: failure happened later
        return "transport: lost box before tray" if d["tcp_min_xy_to_tray"] > 0.05 else "release: reached tray, box not resting in it"
    fl = d["flags"]
    if fl["pushed"]: return "approach: box pushed before closing"
    if fl["lateral_miss"] or fl["long_miss"]: return "approach/localization: lateral or longitudinal miss at closing"
    if fl["too_high"]: return "approach/localization: closed too high (finger-box overlap < 1.7 cm)"
    if d["max_rise"] < PICK_LIFT_M: return "grasp/contact: closed at box, no lift"
    if d["tcp_min_xy_to_tray"] > 0.05: return "transport: lost box before tray"
    return "release: reached tray, box not resting in it"

# ------------------------------------------------------------------ A1 metric definitions & re-computation
def a1(runs):
    res = dict(definitions=dict(pick=f"box rise >= {PICK_LIFT_M} m while |box-tcp| < {HOLD_DIST_M} m at the same step (any step)",
               place=f"final box centre within {PLACE_XY_TOL} m (xy) of tray centre {TRAY.tolist()}, |z - (tray_base_top+0.02)| < {PLACE_Z_TOL} m, box speed < {REST_SPEED} m/s",
               task="pick AND place", horizon="200 control steps at 20 Hz = 10 s", n_per_cell=10, pairing="expert and ACT see identical (seed, idx) box poses",
               ci="Wilson 95% on k/n (n=10 per seed; pooled over 3 seeds n=30)"))
    mism, rows = 0, 0
    for e in runs:
        bs = e["box_speed"] if "box_speed" in e else np.r_[np.zeros(len(e["box_pos"]) - 1), np.linalg.norm(e["box_pos"][-1] - e["box_pos"][-2]) / (1 / HZ)]   # proxy: last-step displacement / dt
        lab = task_labels(e["box_pos"], e["tcp_pos"], bs, TRAY, TRAY_BASE_TOP); rows += 1
        for k in ("pick_success", "place_success", "task_success"): mism += int(lab[k] != e[k])
    res["recomputed_rollouts"] = rows; res["box_speed_source"] = "rollout box_speed if present, else last-step displacement/dt (these rollouts predate box_speed logging)"; res["label_mismatches_vs_episodes_jsonl"] = mism
    # determinism: compare audit re-run to the original reports in outputs/eval/full_v1*
    old = {}
    for d, s in (("full_v1", 7), ("full_v1_seed8", 8), ("full_v1_seed9", 9)):
        for c in ("nominal", "unseen_pose"):
            p = ROOT / "outputs/eval" / d / "act" / c / "episodes.jsonl"
            if p.exists(): old[(s, c)] = {r["episode_index"]: r for r in jl(p)}
    same = diff = 0; maxd = 0.0
    for e in runs:
        if e["agent"] != "act" or (e["seed"], e["cond"]) not in old: continue
        o = old[(e["seed"], e["cond"])][e["episode_index"]]; same += int(o["outcome"] == e["outcome"]); diff += int(o["outcome"] != e["outcome"])
        maxd = max(maxd, abs(o["final_box_to_tray_xy_m"] - e["final_box_to_tray_xy_m"]))
    res["determinism_vs_original_eval"] = dict(same_outcome=same, different_outcome=diff, max_abs_final_box_to_tray_diff_m=maxd)
    tab = {}
    for ag in ("expert", "act"):
        for c in ("nominal", "unseen_pose"):
            es = [e for e in runs if e["agent"] == ag and e["cond"] == c]
            tab[f"{ag}/{c}"] = dict(n=len(es), pick=sum(e["pick_success"] for e in es), place=sum(e["place_success"] for e in es), task=sum(e["task_success"] for e in es),
                                    per_seed={s: sum(e["task_success"] for e in es if e["seed"] == s) for s in sorted({e["seed"] for e in es})})
    res["recomputed_table"] = tab; R["A1_metrics"] = res

# ------------------------------------------------------------------ A2 failure timelines + taxonomy
def a2(runs):
    act = [e for e in runs if e["agent"] == "act" and "action" in e]; rows = []
    for e in act:
        d = events(e); d["class"] = classify(e, d); rows.append(dict(seed=e["seed"], cond=e["cond"], idx=e["episode_index"], box_xy=e["box_xy"], yaw=e["box_yaw"], outcome=e["outcome"],
                                                                      **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k != "grasp_tcp"}))
        rows[-1]["success"] = bool(e["task_success"])
    R["A2_episodes"] = rows; fails = [r for r in rows if r["class"] != "success"]
    R["A2_taxonomy"] = {k: sum(r["class"] == k for r in fails) for k in sorted({r["class"] for r in fails})}
    R["A2_taxonomy_by_cond"] = {c: {k: sum(r["class"] == k for r in fails if r["cond"] == c) for k in sorted({r["class"] for r in fails})} for c in ("nominal", "unseen_pose")}
    R["A2_n_fail"] = len(fails); R["A2_n_total"] = len(rows)
    R["A2_flags_multilabel_among_failures"] = {k: sum(bool(r.get("flags", {}).get(k)) for r in fails) for k in ("pushed", "lateral_miss", "too_high", "long_miss")}
    R["A2_recovery"] = dict(failed_with_second_close=sum(r["n_close_events"] > 1 for r in fails), failed_never_closed=sum(r["t_close"] is None for r in fails),
                            failed_total=len(fails), successes_with_regrasp=sum(r["n_close_events"] > 1 for r in rows if r["class"] == "success"))
    # timeline plot of up to 8 failures (+2 reference successes)
    pick = [e for e in act if not e["task_success"]][:8]; refs = [e for e in act if e["task_success"] and e["cond"] == "unseen_pose"][:2]; sel = pick + refs
    if sel:
        fig, ax = plt.subplots(len(sel), 3, figsize=(14, 2.1 * len(sel)), sharex=True)
        ax = np.atleast_2d(ax)
        for r, e in enumerate(sel):
            nn_ = len(e["action"]); t = np.arange(nn_) / HZ; B, T = e["box_pos"][:nn_], e["tcp_pos"][:nn_]; d = events(e)
            ax[r, 0].plot(t, np.linalg.norm(B[:, :2] - T[:, :2], axis=1) * 100, label="|tcp-box| xy"); ax[r, 0].plot(t, (T[:, 2] - B[:, 2]) * 100, label="tcp z - box z"); ax[r, 0].axhline(0, c="k", lw=.3)
            ax[r, 1].plot(t, e["action"][:, 7] * 1000, c="C2"); ax[r, 1].axhline(CLOSE_THR * 1000, c="k", ls=":", lw=.6); ax[r, 1].set_ylim(-6, 50)
            ax[r, 2].plot(t, (B[:, 2] - B[0, 2]) * 100, c="C3"); ax[r, 2].plot(t, np.linalg.norm(B[:, :2] - TRAY, axis=1) * 100, c="C4")
            ax[r, 0].set_ylabel(f"s{e['seed']} ep{e['episode_index']}\n{e['outcome'][:7]}", fontsize=7)
            if d["t_close"] is not None:
                for k in range(3): ax[r, k].axvline(d["t_close"] / HZ, c="C1", lw=.6)
        for k, ttl in enumerate(["tcp-box offset [cm]", "left gripper cmd [mm]", "box rise [cm] (red) / box-tray xy [cm] (purple)"]): ax[0, k].set_title(ttl, fontsize=8)
        ax[0, 0].legend(fontsize=6); ax[-1, 0].set_xlabel("time [s]"); fig.tight_layout(); fig.savefig(OUT / "A2_failure_timelines.png", dpi=110); plt.close(fig)

# ------------------------------------------------------------------ A3 chunk boundaries, jerk
def jerk_stats(a, mask_boundary=None, H=20):
    x = a[:, ARM]; j = np.abs(np.diff(x, 3, axis=0)) / (1 / HZ) ** 3   # j[t] uses x[t..t+3]
    inside = np.ones(len(j), bool)
    if H:                                                                # exclude windows that straddle a chunk boundary (new chunk starts at t = k*H)
        for t in range(len(j)):
            if any(((t + k) % H == 0) and (t + k) > 0 for k in range(1, 4)): inside[t] = False
    return dict(all=float(j.mean()), interior=float(j[inside].mean()), boundary=float(j[~inside].mean()) if (~inside).any() else None, boundary_share_of_total=float(j[~inside].sum() / j.sum())), j, inside

def a3(runs):
    res = {}
    act = [e for e in runs if e["agent"] == "act" and "action" in e]; exp = [e for e in runs if e["agent"] == "expert" and "action" in e]
    def agg(es, H):
        s = [jerk_stats(e["action"], H=H)[0] for e in es]; return {k: float(np.mean([x[k] for x in s if x[k] is not None])) for k in s[0]}
    res["act_exec20"] = agg(act, 20); res["expert"] = agg(exp, 20)
    res["expert_metric_check_vs_report"] = float(np.mean([smoothness(e["action"], HZ)["mean_abs_jerk"] for e in exp])); res["act_metric_check_vs_report"] = float(np.mean([smoothness(e["action"], HZ)["mean_abs_jerk"] for e in act]))
    # step size by position inside the chunk (0 = first action of a new chunk)
    pos = {k: [] for k in range(20)}
    for e in act:
        d = np.abs(np.diff(e["action"][:, ARM], axis=0)).mean(1)       # d[t] = |a[t+1]-a[t]|
        for t, v in enumerate(d): pos[(t + 1) % 20].append(v)
    res["mean_step_by_pos_in_chunk_rad"] = [float(np.mean(pos[k])) for k in range(20)]
    res["step_at_chunk_start_vs_interior_median"] = float(np.mean(pos[0]) / np.median([np.mean(pos[k]) for k in range(1, 20)]))
    # raw (pre-safety) policy output: is the boundary jump in the model output itself? and chunk-internal noise
    raws = [e for e in act if "raw_action" in e]
    if raws:
        jr = [jerk_stats(np.asarray(e["raw_action"]), H=20)[0] for e in raws]; res["act_raw_output_jerk"] = {k: float(np.mean([x[k] for x in jr if x[k] is not None])) for k in jr[0]}
        res["max_abs_raw_minus_ctrl_arm"] = float(max(np.abs(np.asarray(e["raw_action"])[:, ARM] - e["action"][:, ARM]).max() for e in raws))
    # tracking: how far the measured joints are from the command (all arm joints)
    res["tracking_mean_abs_rad"] = dict(act=float(np.mean([np.abs(e["action"][:, ARM] - e["state"][:, ARM]).mean() for e in act])), expert=float(np.mean([np.abs(e["action"][:, ARM] - e["state"][:, ARM]).mean() for e in exp])))
    # per-joint jerk share
    jj = np.mean([np.abs(np.diff(e["action"][:, ARM], 3, axis=0)).mean(0) for e in act], 0) * HZ ** 3; je = np.mean([np.abs(np.diff(e["action"][:, ARM], 3, axis=0)).mean(0) for e in exp], 0) * HZ ** 3
    res["per_joint_jerk_act"] = jj.round(2).tolist(); res["per_joint_jerk_expert"] = je.round(2).tolist()
    # pilots (inference-only): jerk vs execute horizon / temporal ensembling
    pil = {}
    for name, H in (("pilot_exec10", 10), ("pilot_exec5", 5), ("pilot_te001", 0)):
        es = []
        d = Path(args.rollouts) / name / "act" / "unseen_pose"
        if (d / "episodes.jsonl").exists():
            es = []
            for e in jl(d / "episodes.jsonl"):
                f = d / "rollouts" / f"ep{e['episode_index']:04d}.npz"
                if f.exists():
                    with np.load(f) as z: es.append(dict(e, action=z["action"]))
            if es: pil[name] = dict(n=len(es), task=sum(e["task_success"] for e in es), jerk=float(np.mean([smoothness(e["action"], HZ)["mean_abs_jerk"] for e in es])),
                                    interior=float(np.mean([jerk_stats(e["action"], H=H)[0]["interior"] for e in es])))
    # same configs under exec20 (seed 7 unseen) for a paired reference
    ref = [e for e in act if e["seed"] == 7 and e["cond"] == "unseen_pose"]
    if ref: pil["exec20_seed7_unseen"] = dict(n=len(ref), task=sum(e["task_success"] for e in ref), jerk=float(np.mean([smoothness(e["action"], HZ)["mean_abs_jerk"] for e in ref])),
                                              interior=float(np.mean([jerk_stats(e["action"], H=20)[0]["interior"] for e in ref])))
    res["pilots_inference_only"] = pil; R["A3_chunk_jerk"] = res
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    ax[0].bar(range(20), np.array(res["mean_step_by_pos_in_chunk_rad"]) * 1000); ax[0].set_xlabel("position inside chunk (0 = first action of a new chunk)"); ax[0].set_ylabel("mean |a[t]-a[t-1]| arm [mrad]"); ax[0].set_title("A3 action step vs chunk position (ACT, exec 20)")
    lab = ["expert", "ACT exec20"] + [k for k in pil if k.startswith("pilot")]; val = [res["expert"]["all"], res["act_exec20"]["all"]] + [pil[k]["jerk"] for k in pil if k.startswith("pilot")]
    ax[1].bar(range(len(lab)), val, color=["gray", "C0"] + ["C1"] * (len(lab) - 2)); ax[1].set_xticks(range(len(lab))); ax[1].set_xticklabels(lab, rotation=25, ha="right", fontsize=7); ax[1].set_ylabel("mean |jerk| arm [rad/s^3]"); ax[1].set_title("jerk by execution scheme")
    cats = ["interior", "boundary-straddling"]; ax[2].bar([0, 1], [res["act_exec20"]["interior"], res["act_exec20"]["boundary"] or 0], color=["C0", "C3"]); ax[2].axhline(res["expert"]["all"], c="gray", ls="--", label="expert (all)")
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(cats); ax[2].legend(); ax[2].set_title("ACT exec20: jerk interior vs chunk boundary"); fig.tight_layout(); fig.savefig(OUT / "A3_chunk_jerk.png", dpi=110); plt.close(fig)

# ------------------------------------------------------------------ A4 leakage / independence
def a4():
    res = {}; tr = [e for e in jl(ROOT / "outputs/datasets/demo_train_clipped/episodes.jsonl")]
    sp = json.loads((ROOT / "outputs/lerobot/pickplace_v1/openarm_splits.json").read_text()); info = json.loads((ROOT / "outputs/checkpoints/act_full_v1/train_info.json").read_text())
    res["train_demo_rng_seeds"] = [min(e["seed"] for e in tr), max(e["seed"] for e in tr)]; ev_seeds = sorted({s * 100000 + i for s in (7, 8, 9) for i in range(10)})
    res["eval_rng_seeds"] = [ev_seeds[0], ev_seeds[-1]]; res["rng_seed_overlap"] = sorted(set(e["seed"] for e in tr) & set(ev_seeds))
    res["split_train_val_overlap"] = sorted(set(sp["train"]) & set(sp["val"])); res["split_sizes"] = [len(sp["train"]), len(sp["val"])]
    res["train_info_train_eps_equal_split"] = info["train_episodes"] == sp["train"]
    # pose-level independence: nearest train pose for every eval pose
    P = np.array([[*e["box_xy"], e["box_yaw"]] for e in tr]); evp = []
    for s in (7, 8, 9):
        for c in ("nominal", "unseen_pose"):
            for e in jl(ROOT / "outputs/eval/audit_v1" / f"seed{s}" / "act" / c / "episodes.jsonl") if (ROOT / "outputs/eval/audit_v1" / f"seed{s}" / "act" / c / "episodes.jsonl").exists() else []:
                evp.append((s, c, e["episode_index"], [*e["box_xy"], e["box_yaw"]], e["outcome"]))
    D = np.array([np.linalg.norm((np.array(p[3]) - P) / SCALE, axis=1).min() for p in evp]) if evp else np.array([]); res["min_nn_distance_any_eval_pose_to_train_pose_norm"] = float(D.min()) if len(D) else None
    res["exact_duplicate_poses"] = int((D < 1e-9).sum()) if len(D) else None
    # normaliser statistics provenance
    try:
        from safetensors.numpy import load_file
        t = load_file(str(ROOT / "outputs/checkpoints/act_full_v1/last/policy_preprocessor_step_3_normalizer_processor.safetensors"))
        L = {int(k): tr[int(v)]["length"] for k, v in sp["lerobot_to_raw_episode"].items()}; ntr = sum(L[i] for i in sp["train"]); nva = sum(L[i] for i in sp["val"])
        res["normalizer"] = dict(action_count_in_checkpoint=int(t["action.count"].item()), frames_train=ntr, frames_val=nva, includes_val_episodes=bool(int(t["action.count"].item()) == ntr + nva))
        A_all, A_tr = [], []
        for i, k in enumerate(sorted(map(int, sp["lerobot_to_raw_episode"]))):
            with np.load(ROOT / "outputs/datasets/demo_train_clipped" / tr[sp["lerobot_to_raw_episode"][str(k)]]["file"]) as z: (A_tr if k in sp["train"] else A_all).append(z["action"])
        A_tr = np.concatenate(A_tr); A_both = np.concatenate([A_tr] + A_all)
        live = np.nonzero(A_both.std(0) > 1e-3)[0]          # dims with real variance (left arm + left gripper)
        res["normalizer"]["live_action_dims"] = live.tolist()
        res["normalizer"]["rel_diff_std_train_only_vs_all_max_live_dims"] = float(np.max(np.abs(A_tr.std(0)[live] - A_both.std(0)[live]) / A_both.std(0)[live]))
        res["normalizer"]["max_mean_shift_in_std_units_live_dims"] = float(np.max(np.abs(A_tr.mean(0)[live] - A_both.mean(0)[live]) / A_both.std(0)[live]))
        res["normalizer"]["near_constant_action_dims_std_lt_1e-3"] = {int(i): float(A_both.std(0)[i]) for i in range(16) if A_both.std(0)[i] <= 1e-3}
        S_all = []
        for k in sorted(map(int, sp["lerobot_to_raw_episode"])):
            with np.load(ROOT / "outputs/datasets/demo_train_clipped" / tr[sp["lerobot_to_raw_episode"][str(k)]]["file"]) as z: S_all.append(z["observation.state"])
        S_all = np.concatenate(S_all); res["normalizer"]["state_std_by_dim"] = {i: float(f"{v:.3g}") for i, v in enumerate(S_all.std(0))}
        res["normalizer"]["state_dims_with_std_lt_1e-2_that_are_normalised_by_that_std"] = [int(i) for i in range(16) if S_all.std(0)[i] < 1e-2 and i != 7]
        res["normalizer"]["checkpoint_matches_all_episodes_stats"] = bool(np.allclose(t["action.mean"], A_both.mean(0), atol=1e-4))
    except Exception as ex: res["normalizer"] = dict(error=f"{type(ex).__name__}: {ex}")
    res["checkpoint_selection"] = "'last' checkpoint chosen a priori (no early stopping, no selection on val or test)"
    res["light_randomisation"] = "demos: light_scale ~ U(0.85,1.15), nominal eval = 1.0 (inside); eval conditions light_dim 0.55 / light_bright 1.6 / light_shift / cam_shift never seen in training"
    res["paired_design_note"] = "conditions share the same (seed, idx) box poses, so conditions are NOT independent samples; only seeds are independent replicates"
    R["A4_leakage"] = res; return evp

# ------------------------------------------------------------------ A5 pose distributions
def a5(evp, runs):
    tr = jl(ROOT / "outputs/datasets/demo_train_clipped/episodes.jsonl"); P = np.array([[*e["box_xy"], e["box_yaw"]] for e in tr]); res = {}
    res["train_demo_pose_range"] = dict(x=[float(P[:, 0].min()), float(P[:, 0].max())], y=[float(P[:, 1].min()), float(P[:, 1].max())], yaw=[float(P[:, 2].min()), float(P[:, 2].max())], n=len(P), mean=P.mean(0).round(4).tolist(), std=P.std(0).round(4).tolist())
    ex = lambda v, lo, hi: max(lo - v, v - hi, 0)
    rows = []
    for s, c, i, pose, outc in evp:
        nn = float(np.linalg.norm((np.array(pose) - P) / SCALE, axis=1).min()); exc = [ex(pose[0], *TRAIN_BOX["x"]), ex(pose[1], *TRAIN_BOX["y"]), ex(pose[2], *TRAIN_BOX["yaw"])]
        cen = float(np.linalg.norm((np.array(pose) - P.mean(0)) / SCALE)); rows.append(dict(seed=s, cond=c, idx=i, pose=pose, outcome=outc, success=outc == "success", nn_dist=nn, excess=exc, excess_norm=float(np.linalg.norm(np.array(exc) / SCALE)), centroid_dist=cen))
    R["A5_rows"] = rows
    for c in ("nominal", "unseen_pose"):
        r = [x for x in rows if x["cond"] == c]
        if r: res[c] = dict(n=len(r), nn_dist_median=float(np.median([x["nn_dist"] for x in r])), nn_dist_mean=float(np.mean([x["nn_dist"] for x in r])), frac_outside_train_box=float(np.mean([x["excess_norm"] > 0 for x in r])),
                            pose_range=dict(x=[min(x["pose"][0] for x in r), max(x["pose"][0] for x in r)], y=[min(x["pose"][1] for x in r), max(x["pose"][1] for x in r)], yaw=[min(x["pose"][2] for x in r), max(x["pose"][2] for x in r)]),
                            success=sum(x["success"] for x in r))
    nom = [x["nn_dist"] for x in rows if x["cond"] == "nominal"]; un = [x["nn_dist"] for x in rows if x["cond"] == "unseen_pose"]
    if nom and un: res["mannwhitney_nn_dist_unseen_gt_nominal_p"] = float(st.mannwhitneyu(un, nom, alternative="greater").pvalue)
    u = [x for x in rows if x["cond"] == "unseen_pose"]
    if len(u) > 5:
        y = np.array([x["success"] for x in u], float)
        for name in ("nn_dist", "excess_norm", "centroid_dist"):
            v = np.array([x[name] for x in u]); rho, p = st.spearmanr(v, y); auc = float(st.mannwhitneyu(v[y == 0], v[y == 1], alternative="greater").statistic / (max((y == 0).sum(), 1) * max((y == 1).sum(), 1))) if (y == 0).any() and (y == 1).any() else None
            res[f"unseen_success_vs_{name}"] = dict(spearman_rho=float(rho), p=float(p), auc_failure_has_larger_value=auc)
        for k, nm in enumerate(("x", "y", "yaw")):
            v = np.array([x["pose"][k] for x in u]); rho, p = st.spearmanr(v, y); res[f"unseen_success_vs_{nm}"] = dict(spearman_rho=float(rho), p=float(p))
        res["unseen_success_vs_abs_yaw"] = dict(zip(("spearman_rho", "p"), map(float, st.spearmanr(np.abs([x["pose"][2] for x in u]), y))))
        grp = {}
        for x in u:
            xy_out = bool(x["excess"][0] > 0 or x["excess"][1] > 0); yaw_out = bool(x["excess"][2] > 0); key = ("xy_out" if xy_out else "xy_in") + "/" + ("yaw_out" if yaw_out else "yaw_in")
            g = grp.setdefault(key, [0, 0]); g[0] += int(x["success"]); g[1] += 1
        res["unseen_groups_success_over_n"] = {k: f"{a_}/{b_}" for k, (a_, b_) in sorted(grp.items())}
        a_ = sum(g[0] for k, g in grp.items() if k.startswith("xy_out")); n_a = sum(g[1] for k, g in grp.items() if k.startswith("xy_out")); b_ = sum(g[0] for k, g in grp.items() if k.startswith("xy_in")); n_b = sum(g[1] for k, g in grp.items() if k.startswith("xy_in"))
        res["fisher_exact_success_xy_out_vs_xy_in"] = dict(xy_out=f"{a_}/{n_a}", xy_in=f"{b_}/{n_b}", p_two_sided=float(st.fisher_exact([[a_, n_a - a_], [b_, n_b - b_]])[1]))
        exy = np.array([max(x["excess"][0], x["excess"][1]) for x in u]); res["unseen_success_vs_xy_excess_m"] = dict(zip(("spearman_rho", "p"), map(float, st.spearmanr(exy, y)))); res["xy_excess_m_fail_mean_vs_success_mean"] = [float(exy[y == 0].mean()), float(exy[y == 1].mean())]
        ex_ = np.array([x["pose"][0] for x in u]); res["unseen_x_lt_0p38_success"] = f"{int(y[ex_ < 0.38].sum())}/{int((ex_ < 0.38).sum())}"; res["unseen_x_ge_0p38_success"] = f"{int(y[ex_ >= 0.38].sum())}/{int((ex_ >= 0.38).sum())}"
    R["A5_distribution"] = res
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for a_, (i, j, xl, yl) in zip(ax[:2], ((0, 1, "box x [m]", "box y [m]"), (0, 2, "box x [m]", "box yaw [rad]"))):
        a_.scatter(P[:, i], P[:, j], s=14, c="0.6", label="train demos (60)")
        for c, mk in (("nominal", "o"), ("unseen_pose", "^")):
            for ok, col in ((True, "C2"), (False, "C3")):
                r = [x for x in rows if x["cond"] == c and x["success"] == ok]
                if r: a_.scatter([x["pose"][i] for x in r], [x["pose"][j] for x in r], marker=mk, c=col, s=34, edgecolors="k", linewidths=.4, label=f"ACT {c} {'ok' if ok else 'FAIL'}")
        bx = TRAIN_BOX; xs = (bx["x"], bx["y"], bx["yaw"]); a_.add_patch(plt.Rectangle((xs[i][0], xs[j][0]), xs[i][1] - xs[i][0], xs[j][1] - xs[j][0], fill=False, ec="b", ls="--")); a_.set_xlabel(xl); a_.set_ylabel(yl)
    ax[0].legend(fontsize=6); ax[0].set_title("A5 box pose: train vs evaluation (blue box = train region)")
    for c, col in (("nominal", "C0"), ("unseen_pose", "C1")): ax[2].hist([x["nn_dist"] for x in rows if x["cond"] == c], bins=12, alpha=.6, color=col, label=c)
    ax[2].set_xlabel("distance to nearest train pose (z-scored by train-box widths)"); ax[2].legend(); fig.tight_layout(); fig.savefig(OUT / "A5_pose_distribution.png", dpi=110); plt.close(fig)

# ------------------------------------------------------------------ A6 localisation: where does ACT close the gripper?
def a6(runs):
    res = {}; fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for ag, col in (("expert", "gray"), ("act", "C0")):
        es = [e for e in runs if e["agent"] == ag and "action" in e]; rows = [(e, events(e)) for e in es]; rows = [(e, d) for e, d in rows if d["t_close"] is not None]
        bx = np.array([e["box_xy"] for e, _ in rows]); gp = np.array([d["grasp_tcp"][:2] for _, d in rows]); lat = np.array([d["lateral"] for _, d in rows]); lon = np.array([d["longitudinal"] for _, d in rows]); ok = np.array([e["task_success"] for e, _ in rows])
        fit = lambda x, y: st.linregress(x, y)
        r = dict(n=len(rows), slope_grasp_x_vs_box_x=fit(bx[:, 0], gp[:, 0]).slope, slope_grasp_y_vs_box_y=fit(bx[:, 1], gp[:, 1]).slope, r2_x=fit(bx[:, 0], gp[:, 0]).rvalue ** 2, r2_y=fit(bx[:, 1], gp[:, 1]).rvalue ** 2,
                 lateral_offset_abs_mean_m=float(np.abs(lat).mean()), lateral_offset_abs_p95_m=float(np.percentile(np.abs(lat), 95)), longitudinal_mean_m=float(lon.mean()), longitudinal_std_m=float(lon.std()),
                 lateral_abs_mean_success=float(np.abs(lat[ok]).mean()) if ok.any() else None, lateral_abs_mean_fail=float(np.abs(lat[~ok]).mean()) if (~ok).any() else None)
        res[ag] = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
        ax[0].scatter(bx[:, 1], gp[:, 1], c=col, s=18, label=f"{ag} (slope {r['slope_grasp_y_vs_box_y']:.2f})"); ax[1].scatter(bx[:, 0], gp[:, 0], c=col, s=18, label=f"{ag} (slope {r['slope_grasp_x_vs_box_x']:.2f})")
        if ag == "act": ax[2].scatter(np.abs(lat) * 100, lon * 100, c=np.where(ok, "C2", "C3"), s=22, edgecolors="k", linewidths=.3)
    ax[0].plot([0.04, 0.16], [0.04, 0.16], "k:", lw=.6); ax[1].plot([0.34, 0.5], [0.34, 0.5], "k:", lw=.6); ax[0].set_xlabel("box y [m]"); ax[0].set_ylabel("TCP y when gripper starts closing [m]"); ax[1].set_xlabel("box x [m]"); ax[1].set_ylabel("TCP x at closing [m]")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7); ax[2].axvline(2.2, c="k", ls="--", lw=.6); ax[2].set_xlabel("|lateral offset| at closing [cm]  (green=success, red=fail)"); ax[2].set_ylabel("longitudinal offset [cm]"); ax[0].set_title("A6 localisation: grasp point vs true box position")
    fig.tight_layout(); fig.savefig(OUT / "A6_localisation.png", dpi=110); plt.close(fig); R["A6_localisation"] = res

def a6b(runs):
    act = [(e, events(e)) for e in runs if e["agent"] == "act" and "action" in e]; act = [(e, d) for e, d in act if d["t_close"] is not None]
    exp = [(e, events(e)) for e in runs if e["agent"] == "expert" and "action" in e]; exp = [(e, d) for e, d in exp if d["t_close"] is not None]
    X = np.array([[*e["box_xy"], e["box_yaw"]] for e, _ in act]); dz = np.array([d["dz"] for _, d in act]) * 100; ok = np.array([e["task_success"] for e, _ in act]); tc = np.array([d["t_close"] for _, d in act])
    A = np.c_[X, np.ones(len(X))]; coef, *_ = np.linalg.lstsq(A, dz, rcond=None); pred = A @ coef; r2 = 1 - ((dz - pred) ** 2).sum() / ((dz - dz.mean()) ** 2).sum()
    res = dict(n=len(act), dz_cm_success=dict(mean=float(dz[ok].mean()), std=float(dz[ok].std()), min=float(dz[ok].min()), max=float(dz[ok].max())), dz_cm_fail=dict(mean=float(dz[~ok].mean()), std=float(dz[~ok].std()), min=float(dz[~ok].min()), max=float(dz[~ok].max())) if (~ok).any() else None,
               expert_dz_cm=dict(mean=float(np.mean([d["dz"] for _, d in exp]) * 100), std=float(np.std([d["dz"] for _, d in exp]) * 100)),
               dz_linear_fit_on_box_x_y_yaw=dict(coef_cm_per_m_x=float(coef[0]), coef_cm_per_m_y=float(coef[1]), coef_cm_per_rad_yaw=float(coef[2]), r2=float(r2)),
               spearman_dz_vs_box_x=dict(zip(("rho", "p"), map(float, st.spearmanr(X[:, 0], dz)))), spearman_dz_vs_box_y=dict(zip(("rho", "p"), map(float, st.spearmanr(X[:, 1], dz)))),
               t_close_steps=dict(act_success_mean=float(tc[ok].mean()), act_fail_mean=float(tc[~ok].mean()) if (~ok).any() else None, act_min=int(tc.min()), act_max=int(tc.max()), expert_min=int(min(d["t_close"] for _, d in exp)), expert_max=int(max(d["t_close"] for _, d in exp))),
               track_err_at_close_rad=dict(success=float(np.mean([d["track_err_at_close"] for (_, d), o in zip(act, ok) if o])), fail=float(np.mean([d["track_err_at_close"] for (_, d), o in zip(act, ok) if not o])) if (~ok).any() else None),
               unseen_only_dz_cm=dict(mean=float(np.mean([v for v, (e, _) in zip(dz, act) if e["cond"] == "unseen_pose"])), nominal_mean=float(np.mean([v for v, (e, _) in zip(dz, act) if e["cond"] == "nominal"]))))
    R["A6b_closing_height"] = res
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for k, (v, nm) in enumerate(((X[:, 0], "box x [m]"), (X[:, 1], "box y [m]"), (np.abs(X[:, 2]), "|box yaw| [rad]"))):
        ax[k].scatter(v[ok], dz[ok], c="C2", s=20, edgecolors="k", linewidths=.3, label="success"); ax[k].scatter(v[~ok], dz[~ok], c="C3", s=26, marker="x", label="fail"); ax[k].axhline(res["expert_dz_cm"]["mean"], c="gray", ls="--", label="expert"); ax[k].axhline(3.4, c="k", ls=":", label="overlap<1.7 cm")
        ax[k].set_xlabel(nm); ax[k].set_ylabel("TCP z - box z when gripper starts closing [cm]")
    ax[0].legend(fontsize=7); ax[0].set_title("A6b closing height (higher = fingers miss the box)"); fig.tight_layout(); fig.savefig(OUT / "A6b_closing_height.png", dpi=110); plt.close(fig)

# ------------------------------------------------------------------ A7 image-space probe
def a7():
    p = ROOT / "outputs/eval/audit_v1/pose_visibility.json"
    if not p.exists(): return
    rows = json.loads(p.read_text()); res = {}; fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, cam in enumerate(("front", "wrist_left")):
        tr = [r for r in rows if r["set"] == "train" and r[cam]["u"] is not None]; un = [r for r in rows if r["set"] == "unseen_pose" and r[cam]["u"] is not None]; nm = [r for r in rows if r["set"] == "nominal" and r[cam]["u"] is not None]
        uv = lambda rs: np.array([[r[cam]["u"], r[cam]["v"]] for r in rs]); T = uv(tr)
        res[cam] = dict(box_visible_frac_train=float(np.mean([r[cam]["area"] > 0 for r in rows if r["set"] == "train"])), box_visible_frac_unseen=float(np.mean([r[cam]["area"] > 0 for r in rows if r["set"] == "unseen_pose"])),
                        area_px_train_min=int(min(r[cam]["area"] for r in tr)), area_px_unseen_min=int(min(r[cam]["area"] for r in un)) if un else None,
                        centroid_train_range_uv=[T.min(0).round(1).tolist(), T.max(0).round(1).tolist()], centroid_unseen_range_uv=[uv(un).min(0).round(1).tolist(), uv(un).max(0).round(1).tolist()] if un else None,
                        px_per_cm_front_approx=None)
        ax[k].scatter(*T.T, c="0.6", s=12, label="train"); ax[k].scatter(*uv(nm).T, c="C0", s=16, label="eval nominal"); ax[k].scatter(*uv(un).T, c="C1", marker="^", s=22, label="eval unseen"); ax[k].invert_yaxis(); ax[k].set_title(f"A7 box centroid in {cam} image at t=0 [px, 160x120]"); ax[k].legend(fontsize=7)
        # pixel spread of train vs size of a 1 cm displacement: slope of centroid vs box position
        X = np.array([r["box_xy"] for r in tr]); A = np.c_[X, np.ones(len(X))]; coef = np.linalg.lstsq(A, T, rcond=None)[0]; res[cam]["pixels_per_cm_x_y"] = [float(np.linalg.norm(coef[0]) / 100), float(np.linalg.norm(coef[1]) / 100)]
    R["A7_image_space"] = res; fig.tight_layout(); fig.savefig(OUT / "A7_image_space.png", dpi=110); plt.close(fig)

# ------------------------------------------------------------------ A8 vision ablation + pilots
def a8():
    p = Path(args.rollouts) / "vision_ablation_seed7.json"; res = {}
    if p.exists():
        j = json.loads(p.read_text())["results"]; res["vision_ablation_seed7"] = {k: dict(task=f"{v['task']}/{v['n']}", pick=f"{v['pick']}/{v['n']}") for k, v in j.items()}
    R["A8_vision_ablation"] = res
    pil = {}
    for nm in ("pilot_exec10", "pilot_exec5", "pilot_te001", "diag_right_mean", "diag_right_jitter"):
        f = Path(args.rollouts) / nm / "summary_act.json"
        if f.exists():
            for cond, v in json.loads(f.read_text())["conditions"].items():
                pil[f"{nm}/{cond}"] = dict(task=f"{round(v['task_rate'] * v['n'])}/{v['n']}", pick=f"{round(v['pick_rate'] * v['n'])}/{v['n']}", jerk=round(v["mean_abs_jerk"], 2), tracking=round(v["tracking_mean_arm_rad"], 4),
                                           query_ms=v.get("latency_ms_per_query"), outcomes=v["outcomes"])
    base = {r["episode_index"]: r for r in jl(Path(args.rollouts) / "seed7" / "act" / "unseen_pose" / "episodes.jsonl")}; paired = {}
    for nm in ("pilot_exec10", "pilot_exec5", "pilot_te001", "diag_right_mean", "diag_right_jitter"):
        f = Path(args.rollouts) / nm / "act" / "unseen_pose" / "episodes.jsonl"
        if f.exists():
            cur = {r["episode_index"]: r for r in jl(f)}; b = sum((not base[i]["task_success"]) and cur[i]["task_success"] for i in cur); c = sum(base[i]["task_success"] and (not cur[i]["task_success"]) for i in cur)
            paired[nm] = dict(fixed_vs_exec20=int(b), broken_vs_exec20=int(c), mcnemar_exact_p=float(st.binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0)
    R["A8_paired_vs_exec20_unseen_seed7"] = paired
    R["A8_inference_pilots_and_state_diagnostics"] = pil
    if res:
        fig, ax = plt.subplots(figsize=(6, 3.2)); ks = list(j); ax.bar(range(len(ks)), [j[k]["task"] for k in ks], color="C3"); ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, rotation=25, ha="right", fontsize=7); ax.set_ylabel("task success /10"); ax.set_title("A8 images replaced at inference (real images: nominal 10/10, unseen 3/10 seed 7)")
        fig.tight_layout(); fig.savefig(OUT / "A8_vision_ablation.png", dpi=110); plt.close(fig)

def a9():
    p = Path(args.rollouts) / "grasp_pose_error.json"
    if p.exists(): R["A9_grasp_pose_vs_expert"] = json.loads(p.read_text())["summary"]

def main():
    global args
    ap = argparse.ArgumentParser(); ap.add_argument("--rollouts", default=str(ROOT / "outputs/eval/audit_v1")); args = ap.parse_args()
    runs = []
    for s in (7, 8, 9):
        for ag in ("expert", "act"):
            for c in ("nominal", "unseen_pose"): runs += load_run(args.rollouts, ag, s, c)
    R["n_runs_loaded"] = len(runs)
    a1(runs); a2(runs); a3(runs); evp = a4(); a5(evp, runs); a6(runs); a6b(runs); a7(); a8(); a9()
    (OUT / "audit_results.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: R[k] for k in ("A1_metrics", "A2_taxonomy", "A2_recovery")}, indent=1, default=str)[:3000])

if __name__ == "__main__":
    main()
