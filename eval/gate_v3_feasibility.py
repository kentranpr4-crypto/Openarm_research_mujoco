#!/usr/bin/env python3
"""Gate v3 FEASIBILITY / SUPPORT probe.  Reports ONLY data and baseline properties (reference outcome-only AUROC and branch, hard-subset support, event / static-subset support, persistence & ridge angular errors).
No candidate (WM3-B, WM2-C0, hybrid) number is computed or printed; no threshold is touched.  Datasets: I-1 (now development) and the development windows.  Run: python eval/gate_v3_feasibility.py"""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import wm3_gates as V, wm4_gates as G
SETS = ["test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]; C = G.CONFIG
def load(name):
    win = ROOT / ("outputs/indep/windows" if name == "I-1" else "outputs/wm2/windows"); pr = ROOT / ("outputs/indep/preds" if name == "I-1" else "outputs/indep_dry/preds"); W = {sp: {k: v for k, v in np.load(win / f"{sp}.npz").items()} for sp in SETS}
    P = {m: {sp: {k: v for k, v in np.load(pr / f"{m}_{sp}.npz").items()} for sp in SETS} for m in ("outcome_only", "persistence", "ridge")}; return W, P
out = {}
for name in ("I-1", "development"):
    W, P = load(name); T = {sp: {k: W[sp][k] for k in ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut", "Y_s", "Y_b", "Y_c", "Y_e")} for sp in SETS}; grp = {sp: W[sp]["group"] for sp in SETS}; act = {sp: V.active_mask(W[sp]) for sp in SETS}; masks = {(l, sp): V.at_risk_mask(W[sp], l) for l in V.LAB for sp in SETS}; r = {}
    for tgt in C["G3"]["primary"]:
        y, p, po, g = V._gather(P, P["outcome_only"], T, grp, masks, tgt["label"], tgt["sets"]) if False else V._gather(P["outcome_only"], P["outcome_only"], T, grp, masks, tgt["label"], tgt["sets"]); ar = V.auroc(y, po); suff = V._suff(C, y, g); lo, hi = C["G3"]["hard_subset"]["ref_prob_range"]; hm = (po >= lo) & (po <= hi)
        r[f"{tgt['label']}/{'+'.join(tgt['sets'])}"] = dict(n_windows=int(len(y)), prevalence=float(y.mean()), auroc_ref=ar, branch="margin" if ar <= C["G3"]["branch_threshold_auroc_ref"] else "ceiling", max_possible_v2_margin=float(1 - ar), sufficiency_ok=suff[0], sufficiency=suff[1],
                                                           hard_subset=dict(n_windows=int(hm.sum()), frac=float(hm.mean()), sufficiency_ok=V._suff(C, y[hm], g[hm])[0] if hm.any() else False, sufficiency=V._suff(C, y[hm], g[hm])[1] if hm.any() else None), ref_brier_macro=float(G.brier_macro(y, po, g).mean()))
    ev = {sp: G.event_mask(T[sp]) for sp in SETS}; gated = {sp: act[sp] & ~ev[sp] for sp in SETS}; sin = {sp: G.static_input_mask(T[sp]) for sp in SETS}; st = {sp: sin[sp] & ~ev[sp] for sp in SETS}
    g7 = dict(active_windows=int(sum(act[sp].sum() for sp in SETS)), event_windows=int(sum((act[sp] & ev[sp]).sum() for sp in SETS)), event_groups=len({int(x) for sp in SETS for x in np.unique(grp[sp][act[sp] & ev[sp]])}), gated_windows=int(sum(gated[sp].sum() for sp in SETS)),
              gated_groups=len({int(x) for sp in SETS for x in np.unique(grp[sp][gated[sp]])}), S_in_windows=int(sum(st[sp].sum() for sp in SETS)), S_in_groups=len({int(x) for sp in SETS for x in np.unique(grp[sp][st[sp]])}),
              S_in_frac_of_gated=float(sum(st[sp].sum() for sp in SETS) / sum(gated[sp].sum() for sp in SETS)), S_in_frac_truth_static_lt_0p003=float(np.mean(np.concatenate([np.abs(G.theta(T[sp]["Y_b"], 20))[st[sp]] < 0.003 for sp in SETS]))),
              S_in_events_excluded_windows=int(sum((sin[sp] & ev[sp]).sum() for sp in SETS)))
    for m in ("persistence", "ridge"):
        for tag, mk in (("gated", gated), ("S_in", st)):
            vals = []
            for sp in SETS:
                if mk[sp].any(): e = G.ang_err(V.sub(P[m][sp], mk[sp]), V.sub(T[sp], mk[sp]), 20); vals.append(V.group_means(e, grp[sp][mk[sp]])[1])
            g7[f"E_{m}_{tag}_K20_rad"] = float(np.concatenate(vals).mean())
    g7["error_floor_rad"] = C["G7"]["error_floor_rad"]; r["G7_support"] = g7; out[name] = r
(ROOT / "outputs/gates_v3/feasibility.json").write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out, indent=1, default=str))
