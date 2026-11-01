#!/usr/bin/env python3
"""Gate v3 SUPPORT AUDIT on the contract-v2 windows (quaternions available).  Counts only (windows / initial-state groups per regime and split), regime-threshold sensitivity, precedence consistency, mask coverage and the effect of D2 symmetry.
No model prediction is used; nothing is tuned.  Run: python eval/gate_v3_support_audit.py"""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import wm4_gates as G, geometry_v2 as Gm, copy
SPL = ["train", "val", "test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]; DEV = SPL[2:]; W = {sp: {k: v for k, v in np.load(ROOT / "outputs/wm2_v2/windows" / f"{sp}.npz").items()} for sp in SPL}
def regs(z, cfg=G.CONFIG): rg, Q = G.regimes(z["Q_t"], z["Y_rv"], cfg); return rg, Q
out = {"regime_support_by_split": {}, "S_in": {}}
for sp in SPL:
    z = W[sp]; rg, Q = regs(z); act = z["active"]; g = z["group"]
    out["regime_support_by_split"][sp] = {r: dict(windows=int((act & m).sum()), groups=int(len(np.unique(g[act & m])))) for r, m in rg.items()} | dict(active_windows=int(act.sum()), groups=int(len(np.unique(g[act]))))
    sin = G.static_input_mask(z) & (rg["stable"] | rg["rotating"]); out["S_in"][sp] = dict(windows=int(sin.sum()), groups=int(len(np.unique(g[sin]))), note="mirrors gate_G7: S_in is input-defined and heading-valid, NOT restricted to active windows")
# pooled development sets (the four wm2 test splits) and the validation split
def pooled(sps):
    res = {}
    for r in ("stable", "rotating", "tumbling", "singular"):
        gg = set(); n = 0
        for sp in sps: rg, _ = regs(W[sp]); m = W[sp]["active"] & rg[r]; n += int(m.sum()); gg |= set(W[sp]["group"][m].tolist())
        res[r] = dict(windows=n, groups=len(gg), evaluable=bool(n >= 200 and len(gg) >= 10))
    st_n = 0; st_g = set()
    for sp in sps: rg, _ = regs(W[sp]); m = G.static_input_mask(W[sp]) & (rg["stable"] | rg["rotating"]); st_n += int(m.sum()); st_g |= set(W[sp]["group"][m].tolist())
    res["S_in_heading_valid"] = dict(windows=st_n, groups=len(st_g), evaluable=bool(st_n >= 200 and len(st_g) >= 10)); ot = res["tumbling"]["windows"] + res["singular"]["windows"]; og = set()
    for sp in sps: rg, _ = regs(W[sp]); m = W[sp]["active"] & (rg["tumbling"] | rg["singular"]); og |= set(W[sp]["group"][m].tolist())
    res["orientation_singular_plus_tumbling"] = dict(windows=ot, groups=len(og), evaluable=bool(ot >= 200 and len(og) >= 10)); return res
out["pooled_development_splits"] = pooled(DEV); out["pooled_validation"] = pooled(["val"]); out["pooled_train"] = pooled(["train"])
# threshold sensitivity (counts only) on the pooled development splits
sens = {}
for name, key, vals in (("singular_hnorm", "tau_hnorm_singular", [0.1, 0.2, 0.3]), ("tumble_step_geodesic", "tumble_step_geodesic_rad", [0.2, 0.3, 0.5]), ("rotating_total_geodesic", "rotating_total_geodesic_min_rad", [0.01, 0.02, 0.05])):
    sens[name] = {}
    for v in vals:
        cfg = copy.deepcopy(G.CONFIG); cfg["G7"]["regimes"][key] = v; cnt = {r: 0 for r in ("stable", "rotating", "tumbling", "singular")}
        for sp in DEV: rg, _ = regs(W[sp], cfg); [cnt.__setitem__(r, cnt[r] + int((W[sp]["active"] & m).sum())) for r, m in rg.items()]
        sens[name][str(v)] = cnt
out["threshold_sensitivity_windows_pooled_dev"] = sens
# precedence consistency: exactly one regime per window
ok = True
for sp in SPL: rg, _ = regs(W[sp]); ok &= bool((sum(m.astype(int) for m in rg.values()) == 1).all())
out["precedence_exactly_one_regime_per_window"] = ok
# D2 effect: how often does the symmetric geodesic differ from the full geodesic over the horizon, and the yaw-flip artefact
d = {}
for sp in DEV:
    z = W[sp]; rg, Q = regs(z); full = Gm.geodesic(Q[:, 0], Q[:, -1]); sym = Gm.geodesic_sym(Q[:, 0], Q[:, -1]); m = z["active"]; d[sp] = dict(frac_windows_full_minus_sym_gt_0p1=float(((full - sym) > 0.1)[m].mean()), frac_in_singular=float(rg["singular"][m].mean()), frac_full_minus_sym_gt_0p1_among_singular=float(((full - sym) > 0.1)[m & rg["singular"]].mean()) if (m & rg["singular"]).any() else None, max_sym=float(sym.max()), max_full=float(full.max()))
out["d2_effect"] = d
# mask coverage
out["yaw_mask_coverage"] = {sp: dict(frac_steps_masked=float(1 - W[sp]["Y_yaw_valid"].mean()), windows_with_any_masked_step=int((~W[sp]["Y_yaw_valid"]).any(1).sum()), windows=int(len(W[sp]["t"]))) for sp in SPL}
(ROOT / "outputs/gates_v3/support_audit.json").write_text(json.dumps(out, indent=1)); print(json.dumps({k: out[k] for k in ("pooled_development_splits", "pooled_validation", "precedence_exactly_one_regime_per_window", "threshold_sensitivity_windows_pooled_dev")}, indent=1))
