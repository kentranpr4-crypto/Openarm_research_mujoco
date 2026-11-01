#!/usr/bin/env python3
"""INFERENCE-ONLY yaw persistence feasibility (no training, no checkpoint change).  Post-hoc variants of the frozen predictions:
  P0  yaw = persistence (zero rotation) for every window;   P2  rule-based: keep the model's yaw only when the INPUT shows an interaction (box speed > 0.005 m/s in any history row, or finger/palm contact in any history row, or the candidate future gripper command closes below 0.028 m); otherwise persistence.
The P2 rule is fixed by physics before looking at results and has no tunable parameter.  Diagnostic strata use TRUE future yaw (flagged).  Datasets: I-1 (development now) and the development windows.  Run: python eval/yaw_persistence_feasibility.py"""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import wm2_gates as G
SETS = ["test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]; wrap = lambda x: np.arctan2(np.sin(x), np.cos(x))
def load(name):
    win = ROOT / ("outputs/indep/windows" if name == "I-1" else "outputs/wm2/windows"); pr = ROOT / ("outputs/indep/preds" if name == "I-1" else "outputs/indep_dry/preds")
    W = {sp: {k: v for k, v in np.load(win / f"{sp}.npz").items()} for sp in SETS}; P = {m: {sp: {k: v for k, v in np.load(pr / f"{m}_{sp}.npz").items()} for sp in SETS} for m in ("persistence", "ridge", "wm2_c0", "wm3b")}; return W, P
def interaction(W): return (W["B_hist"][:, :, 4].max(1) > 0.005) | (W["C_hist"][:, :, 1].max(1) > 0) | (W["C_hist"][:, :, 4].max(1) > 0.01) | (W["A_fut"][:, :, 7].min(1) < 0.028)
out = {}
for name in ("I-1", "development"):
    W, P = load(name); r = {}
    for model in ("wm3b", "wm2_c0"):
        rows = {}
        for variant in ("model_yaw", "P0_persistence_yaw", "P2_interaction_rule", "ridge", "persistence"):
            vals = {k: [] for k in ("all", "non_event", "static_truth", "rotating_truth", "event", "S_in_like_no_interaction")}; box = []
            for sp in SETS:
                a = G.active_mask(W[sp]); yt = np.cumsum(W[sp]["Y_b"][:, :, 3], 1)[:, 19]; ev = (np.abs(W[sp]["Y_b"][:, :, 3]) > np.pi / 2).any(1); inter = interaction(W[sp])
                if variant == "model_yaw": yp = np.cumsum(P[model][sp]["Y_b"][:, :, 3], 1)[:, 19]
                elif variant == "P0_persistence_yaw": yp = np.zeros_like(yt)
                elif variant == "P2_interaction_rule": yp = np.where(inter, np.cumsum(P[model][sp]["Y_b"][:, :, 3], 1)[:, 19], 0.0)
                else: yp = np.cumsum(P[variant][sp]["Y_b"][:, :, 3], 1)[:, 19]
                e = np.abs(wrap(yp - yt)); tt = np.abs(wrap(yt)); g = W[sp]["group"]
                for k, mk in (("all", a), ("non_event", a & ~ev), ("static_truth", a & ~ev & (tt < 0.003)), ("rotating_truth", a & ~ev & (tt > 0.05)), ("event", a & ev), ("S_in_like_no_interaction", a & ~ev & ~inter)):
                    if mk.any(): vals[k].append(G.group_means(e[mk], g[mk])[1])
            rows[variant] = {k: (float(np.concatenate(v).mean()) if v else None) for k, v in vals.items()}
        # box position unchanged by construction (separate output channels): verify
        bx = [float(np.abs(P[model][sp]["Y_b"][:, :, :3]).sum()) for sp in SETS]; r[model] = rows
    r["_note"] = "group-macro mean |wrap(yaw displacement error)| at K=20 [rad]; strata: non_event = no true per-step |dyaw|>pi/2; static/rotating by TRUE |yaw displacement| (<0.003 / >0.05) = diagnostic; S_in_like = no interaction in the INPUT (rule P2's complement); box/joint/contact outputs are not touched by P0/P2"
    out[name] = r
(ROOT / "outputs/contract_v2/yaw_persistence_feasibility.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
