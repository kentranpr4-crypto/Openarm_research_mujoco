#!/usr/bin/env python3
"""Analysis of the offline branching replay (numpy only; pre-registered in outputs/branching_replay/protocol_freeze.json).  DEVELOPMENT experiment: short-horizon (one/two chunk) candidate ranking against EXECUTED branch outcomes; not closed-loop success, not independent, not Sim2Real.
Run: python3 eval/branching_replay_analysis.py"""
import sys, json, glob, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); import branching_metrics as BM
D = ROOT / "outputs/branching_replay"; W = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(D / "results/worker*.json")))]
S = [s for w in W for s in w["states"]]; PH = ("approach", "grasp", "carry", "place"); out = dict(label="DEVELOPMENT offline branching replay (executed branch outcomes as labels; short-horizon; privileged observer)")
out["status"] = dict(workers=[dict(worker=w["worker"], phase1=w["phase1"], phase2=w["phase2"], cpu_s=w.get("cpu_s"), stop_reason=w.get("stop_reason")) for w in W])
out["counts"] = dict(n_poses=len({s["pose_idx"] for s in S}), n_states=len(S), n_branches=sum(len(s["branches"]) for s in S), expected=dict(poses=24, states=216, branches=648))
bad = [b["branch_id"] for s in S for b in s["branches"] if not (b["legal"] and b["finite"] and b["applied_ctrl_maxdiff"] == 0.0)]; out["safety_and_semantics"] = dict(violations=len(bad), ids=bad[:5], clipped_total=int(sum(b["clipped"] for s in S for b in s["branches"])), slewed_total=int(sum(b["slewed"] for s in S for b in s["branches"])))
def metrics_for(states, key_p, key_t):
    M = []
    for s in states:
        Jp = s["pred"][key_p]; Jt = [b[key_t] for b in s["branches"]]
        if key_t == "J2_real" and any(v is None for v in Jt): continue
        m = BM.state_metrics(Jp, Jt, s["pose_idx"]); m.update(pose=s["pose_idx"], phase=s["phase"], t=s["t"]); M.append(m)
    return M
def summarize(M):
    if not M: return dict(n_states=0, verdict="NOT_EVALUABLE")
    byp = lambda f: {}.copy()
    def per_pose(f):
        d = {}
        for m in M:
            v = f(m)
            if v is not None: d.setdefault(m["pose"], []).append(v)
        return d
    num = {}; den = {}
    for m in M: num[m["pose"]] = num.get(m["pose"], 0.0) + m["pair"][0]; den[m["pose"]] = den.get(m["pose"], 0.0) + m["pair"][1]
    acc = BM.boot_ratio(num, den); r = {k: BM.boot_mean_by_pose(per_pose(lambda m, k=k: m[k])) for k in ("headroom", "regret_wm", "regret_default", "regret_random", "regret_gated")}
    pd = lambda a, b: BM.boot_paired_by_pose(per_pose(lambda m: m[a]), per_pose(lambda m: m[b]))
    hd = [m["headroom"] for m in M]; hs = sum(m["headroom"] for m in M); captured = (hs - sum(m["regret_wm"] for m in M)) / hs if hs > 0 else None
    nrm = {k: [m[k] for m in M if m[k] is not None] for k in ("norm_regret_wm", "norm_regret_default", "norm_regret_random")}
    return dict(n_states=len(M), n_poses=len({m["pose"] for m in M}), n_meaningful_pairs=int(sum(m["pair"][1] for m in M)), pairwise_accuracy_pooled_ratio_ci=acc, mean_by_pose_ci=r,
                paired_diff_wm_minus_default=pd("regret_wm", "regret_default"), paired_diff_wm_minus_random=pd("regret_wm", "regret_random"), paired_diff_gated_minus_default=pd("regret_gated", "regret_default"),
                headroom_states_ge_0p01=int(sum(h >= BM.THR_PAIR for h in hd)), headroom_frac_ge_0p01=float(np.mean([h >= BM.THR_PAIR for h in hd])), headroom_quantiles={q: float(np.percentile(hd, q)) for q in (10, 25, 50, 75, 90, 100)}, fraction_of_headroom_captured_by_wm=captured,
                normalized_regret_mean={k: (float(np.mean(v)) if v else None) for k, v in nrm.items()}, normalized_regret_n={k: len(v) for k, v in nrm.items()}, wm_top1_is_oracle_best_frac=float(np.mean([m["regret_wm"] < 1e-12 for m in M])), wm_equals_default_frac=float(np.mean([m["selected"] == 0 for m in M])))
M20 = metrics_for(S, "J20", "J20_real"); out["J20_overall"] = summarize(M20); out["J20_by_phase"] = {ph: summarize([m for m in M20 if m["phase"] == ph]) for ph in PH}
out["oracle_stop_rule"] = dict(rule="oracle headroom < 0.01 in > 80% of decision states -> report and stop further ranking research under this protocol", frac_states_without_headroom=float(1 - out["J20_overall"].get("headroom_frac_ge_0p01", 0)) if M20 else None, triggered=bool(M20 and (1 - out["J20_overall"]["headroom_frac_ge_0p01"]) > 0.8))
# margin / P3 diagnostics
mb = {"<1e-3": (0, 1e-3), "1e-3..5e-3": (1e-3, 5e-3), "5e-3..1e-2": (5e-3, 1e-2), ">=1e-2": (1e-2, 9e9)}; pairs = []
for s in S:
    Jp = np.array(s["pred"]["J20"]); Jt = np.array([b["J20_real"] for b in s["branches"]])
    for i in range(3):
        for j in range(i + 1, 3):
            dt = Jt[i] - Jt[j]
            if abs(dt) < BM.THR_PAIR: continue
            dp = Jp[i] - Jp[j]; pairs.append((abs(dp), 1.0 if dp * dt > 0 else (0.5 if abs(dp) < 1e-9 else 0.0), s["pose_idx"]))
out["margin_vs_correctness_meaningful_pairs"] = {k: dict(n=int(sum(lo <= a < hi for a, _, _ in pairs)), accuracy=(float(np.mean([c for a, c, _ in pairs if lo <= a < hi])) if any(lo <= a < hi for a, _, _ in pairs) else None)) for k, (lo, hi) in mb.items()}
cov = [m["margin"] >= BM.TAU for m in M20]; out["P3_margin_gating_tau_0p01_fallback_c0_s0"] = dict(coverage_states_with_margin_ge_tau=int(sum(cov)), of=len(cov), coverage_frac=float(np.mean(cov)) if cov else None, regret_gated_mean=out["J20_overall"].get("mean_by_poses_ci") if False else out["J20_overall"].get("mean_by_pose_ci", {}).get("regret_gated"),
                                                                    regret_always_rank=out["J20_overall"].get("mean_by_pose_ci", {}).get("regret_wm"), regret_default=out["J20_overall"].get("mean_by_pose_ci", {}).get("regret_default"), paired_gated_minus_default=out["J20_overall"].get("paired_diff_gated_minus_default"))
# predicted vs realised palm/dxy on all branches
conf = dict(tp=0, fp=0, fn=0, tn=0); dx = []
for s in S:
    for k, b in enumerate(s["branches"]):
        pp = s["pred"]["terms"][k]["palm"]; rp = b["terms_real"]["palm"]; conf["tp" if pp and rp else "fp" if pp and not rp else "fn" if rp and not pp else "tn"] += 1; dx.append(abs(s["pred"]["terms"][k]["dxy"] - b["terms_real"]["dxy"]))
out["palm_pred_vs_real"] = dict(confusion=conf, predicted_rate=(conf["tp"] + conf["fp"]) / max(sum(conf.values()), 1), realised_rate=(conf["tp"] + conf["fn"]) / max(sum(conf.values()), 1)); out["dxy_pred_vs_real_mae_m"] = float(np.mean(dx)) if dx else None
vs = [e for s in S for e in s["exe_linf_vs_default_left_arm"][1:]]; out["candidate_diversity_executed_linf_vs_default_left_arm_rad"] = dict(median=float(np.median(vs)), p10=float(np.percentile(vs, 10)), p90=float(np.percentile(vs, 90))) if vs else None
# J2
S2 = [s for s in S if s.get("j2_states") and all("J2_real" in b for b in s["branches"])]; M2 = metrics_for(S2, "J2", "J2_real"); M20_same = [m for m in M20 if (m["pose"], m["t"]) in {(s["pose_idx"], s["t"]) for s in S2}]
out["J2_hold_continuation"] = dict(definition="prediction: WM 2-block rollout [candidate, hold last executed target 20 steps]; label: realised J over the same 40 steps; states t<=160 only", n_states_evaluable=len(S2), of=sum(1 for s in S if s.get("j2_states")), J2=summarize(M2), J20_same_states=summarize(M20_same), verdict="NOT_EVALUABLE" if len(S2) == 0 else ("PARTIAL" if len(S2) < sum(1 for s in S if s.get("j2_states")) else "EVALUABLE"))
(D / "results/branching_analysis.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps(out, indent=1, default=float)[:12000])
