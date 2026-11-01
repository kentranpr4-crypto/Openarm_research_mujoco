#!/usr/bin/env python3
"""Gate v3 FEASIBILITY / SUPPORT probe (final draft).  Data and baseline properties ONLY: no candidate (WM3-B, WM2-C0, hybrid) number is computed; no threshold is touched.
G3: reference outcome-only AUROC / branch, prevalence / rare-label status, hard-subset support, reference operating point.  G7: v1 windows carry no quaternions, so G7 cannot be evaluated on I-1 / development; the regime mix and S_in support are
estimated from the 394 replayed episodes (outputs/contract_v2/replay_sample.npz; event episodes OVER-sampled) and from input-only S_in on the v1 windows.   Run: python eval/gate_v3_feasibility.py"""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import wm3_gates as V, wm4_gates as G, geometry_v2 as Gm
SETS = ["test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]; C = G.CONFIG
def load(name):
    win = ROOT / ("outputs/indep/windows" if name == "I-1" else "outputs/wm2/windows"); pr = ROOT / ("outputs/indep/preds" if name == "I-1" else "outputs/indep_dry/preds"); W = {sp: {k: v for k, v in np.load(win / f"{sp}.npz").items()} for sp in SETS}
    P = {m: {sp: {k: v for k, v in np.load(pr / f"{m}_{sp}.npz").items()} for sp in SETS} for m in ("outcome_only",)}; return W, P
out = {}
for name in ("I-1", "development"):
    W, P = load(name); T = {sp: {k: W[sp][k] for k in ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut", "Y_s", "Y_b", "Y_c", "Y_e")} for sp in SETS}; grp = {sp: W[sp]["group"] for sp in SETS}; masks = {(l, sp): V.at_risk_mask(W[sp], l) for l in V.LAB for sp in SETS}; r = {}
    for tgt in C["G3"]["primary"]:
        y, po, _, g = V._gather(P["outcome_only"], P["outcome_only"], T, grp, masks, tgt["label"], tgt["sets"]); ar = V.auroc(y, po); prev = float(y.mean()); suff = V._suff(C, y, g); lo, hi = C["G3"]["hard_subset"]["ref_prob_range"]; hm = (po >= lo) & (po <= hi); rare = prev < C["G3"]["rare_label"]["prevalence_below"]
        tpr, prec, thr = G.tpr_at_fpr(y, po, C["G3"]["rare_label"]["operating_point"]["fpr"])
        r[f"{tgt['label']}/{'+'.join(tgt['sets'])}"] = dict(n_windows=int(len(y)), prevalence=prev, rare_label=rare, auroc_ref=ar, branch="margin" if ar <= C["G3"]["branch_threshold_auroc_ref"] else "ceiling", max_possible_v2_margin=float(1 - ar), sufficiency_ok=suff[0], sufficiency=suff[1],
                                                           hard_subset=("NOT_APPLICABLE (rare label)" if rare else dict(n_windows=int(hm.sum()), sufficiency_ok=V._suff(C, y[hm], g[hm])[0] if hm.any() else False, sufficiency=V._suff(C, y[hm], g[hm])[1] if hm.any() else None)),
                                                           reference_tpr_at_fpr5pct=tpr, reference_brier_macro=float(G.brier_macro(y, po, g).mean()))
    sin = {sp: G.static_input_mask(T[sp]) for sp in SETS}; act = {sp: V.active_mask(W[sp]) for sp in SETS}; ev_proxy = {sp: (np.abs(W[sp]["Y_b"][:, :, 3]) > np.pi / 2).any(1) for sp in SETS}
    r["G7_input_only_support_on_v1_windows"] = dict(S_in_windows=int(sum((sin[sp] & ~ev_proxy[sp]).sum() for sp in SETS)), S_in_groups=len({int(x) for sp in SETS for x in np.unique(grp[sp][sin[sp] & ~ev_proxy[sp]])}), active_windows=int(sum(act[sp].sum() for sp in SETS)), legacy_yaw_event_windows_proxy=int(sum((act[sp] & ev_proxy[sp]).sum() for sp in SETS)),
                                                    note="v1 windows have no quaternions: G7 regimes and components are NOT_EVALUABLE on I-1 / development by construction")
    out[name] = r
z = np.load(ROOT / "outputs/contract_v2/replay_sample.npz"); eids = sorted({k.split("/")[0] for k in z.files}); ev = json.load(open(ROOT / "outputs/yaw_audit/yaw_events.json")); evset = {x["eid"] for x in ev}; cnt = {k: 0 for k in ("singular", "tumbling", "rotating", "stable")}; cnt_ne = dict(cnt)
for e in eids:
    q = z[f"{e}/q"]
    for t in range(4, len(q) - 21, 5):
        Qw = q[t:t + 21]; rg = G.regimes(Qw[:1], Gm.step_rotvecs(Qw)[None], C)[0]; k = [n for n, v in rg.items() if v[0]][0]; cnt[k] += 1
        if e not in evset: cnt_ne[k] += 1
tot = sum(cnt.values()); tne = sum(cnt_ne.values()); out["regime_mix_from_replay_sample"] = dict(windows=tot, mix={k: v / tot for k, v in cnt.items()}, counts=cnt, non_event_episodes_only=dict(windows=tne, mix={k: v / tne for k, v in cnt_ne.items()}, counts=cnt_ne),
                                                                                   note="394 episodes: 34 event episodes (8.6 %, vs 0.7 % of the dataset) + 360 random others; the non-event row is the unbiased-for-events view of the sample")
(ROOT / "outputs/gates_v3/feasibility.json").write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out, indent=1, default=str)[:6000])
