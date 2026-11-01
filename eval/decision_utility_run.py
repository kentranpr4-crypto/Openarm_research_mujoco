#!/usr/bin/env python3
"""OFFLINE DECISION-UTILITY EVALUATION (DEVELOPMENT only).  Frozen WM2-C0, WM3-B, post-hoc hybrid, ridge, MLP, action-blind MLP / ridge rank the 25 paired action variants of each initial state by an autoregressive K=20-block rollout scored with the FIXED function of eval/decision_utility_lib.py.
Datasets (both already analysed => DEVELOPMENT): the 31 held-out poses of the wm2 test splits and the 50 poses of I-1.  Inputs are privileged simulator states (disclosed; simulation research only).  No training; the CPU guard stops the run (PARTIAL) at 0.1 process CPU-hour.
Modes (ACT venv):  python eval/decision_utility_run.py check | freeze | run"""
import os, sys, json, time, hashlib, resource, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "data"))
MODE = sys.argv[1]; sys.argv = [sys.argv[0], "none"]
import torch; torch.set_num_threads(1)
import wm2_run as R, wm2_models as M, wm3b_model as B, decision_utility_lib as L
M.logistic_fit = lambda X, y, l2=0.01, iters=200: (np.zeros((X.shape[1], y.shape[1]), np.float32), np.zeros(y.shape[1], np.float32))          # event heads of the ridge are not used in the ranking (closed-form Y_s/Y_b/Y_c unchanged)
OUT = ROOT / "outputs/decision_utility"; OUT.mkdir(exist_ok=True); PRIOR = float(os.environ.get("PRIOR_CPU", 0)); CAP = 0.1 * 3600 - PRIOR      # CPU already spent by the sanity-check run counts against the same 0.1 CPU-h cap
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cpu(): a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN); return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime
T0, C0 = time.time(), cpu()
class Stop(Exception): pass
def guard(tag):
    if cpu() - C0 > CAP: raise Stop(tag)
FAMS = ["depth", "timing", "noise", "speed", "release_height"]
FILES = ["eval/decision_utility_lib.py", "eval/decision_utility_run.py", "eval/test_decision_utility.py", "docs/ai-native/tasks/2026-11-01-decision-utility-protocol.md", "outputs/wm2_1/models/gru_c0.pt", "outputs/wm2_1/models/mlp.pt", "outputs/wm2_1/models/mlp_blind.pt", "outputs/wm3/models/wm3b.pt", "outputs/gates_v3/freeze.json"]
CONST = dict(yaw_variants=["persistence"], auroc_bootstrap_resamples=500, J="dxy + |z-0.384| + 0.05*palm_any(>0.01N) + 0.05*(max rise<0.05m)", t0=4, K=20, tie="sha1(pose|vid|tie)", meaningful_pair=0.01, bootstrap="poses, 10000, seed 0, 95%", cap_cpu_s=CAP)
if MODE == "freeze":
    assert json.loads((OUT / "unit_tests.json").read_text())["all_pass"], "library tests not passing"; assert (OUT / "check.json").exists() and json.loads((OUT / "check.json").read_text())["passed"], "run `check` first"
    if (OUT / "protocol_freeze.json").exists(): sys.exit("already frozen")
    F = dict(frozen_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), constants=CONST, files_sha256={f: sha(ROOT / f) for f in FILES}, status="DEVELOPMENT evaluation; scoring function and protocol frozen BEFORE any ranking result was computed", ranking_results_exist=(OUT / "results.json").exists()); assert not F["ranking_results_exist"]
    (OUT / "protocol_freeze.json").write_text(json.dumps(F, indent=1)); print("FROZEN", sha(OUT / "protocol_freeze.json")); sys.exit(0)
if MODE == "run":
    F = json.loads((OUT / "protocol_freeze.json").read_text()); bad = [f for f, h in F["files_sha256"].items() if sha(ROOT / f) != h]; assert not bad, f"FROZEN FILES CHANGED {bad}"
# ------------------------------------------------------------------------------------------------ models (frozen)
tr = R.load("train"); nrm = M.Norm(tr); ridge = M.Ridge("cond").fit(nrm, tr); ridge_blind = M.Ridge("blind").fit(nrm, tr)
def load_nn(kind, path):
    n = M.NN(kind); n.nrm = nrm; n.model = M.GRUModel() if kind == "gru" else M.MLP(280 if kind == "mlp_cond" else 88); n.model.load_state_dict(torch.load(path)); n.model.eval(); return n
c0 = load_nn("gru", ROOT / "outputs/wm2_1/models/gru_c0.pt"); mlp = load_nn("mlp_cond", ROOT / "outputs/wm2_1/models/mlp.pt"); mlpb = load_nn("mlp_blind", ROOT / "outputs/wm2_1/models/mlp_blind.pt"); w3 = B.WM3B(ridge, nrm); w3.model = B.GRUOC(); w3.model.load_state_dict(torch.load(ROOT / "outputs/wm3/models/wm3b.pt")); w3.model.eval(); guard("models")
def hyb(W): rp = ridge.predict(W); cp = c0.predict(W); return dict(Y_s=rp["Y_s"], Y_b=cp["Y_b"], Y_c=cp["Y_c"], Y_e=cp["Y_e"])
def zero_box(W): n = len(W["t"]); return dict(Y_s=np.zeros((n, 20, 8)), Y_b=np.zeros((n, 20, 7)), Y_c=np.repeat(W["C_hist"][:, -1:], 20, 1), Y_e=np.full((n, 4), 0.5, np.float32))
PRED = {"wm3b": w3.predict, "wm2_c0": c0.predict, "hybrid_posthoc": hyb, "ridge": ridge.predict, "mlp": mlp.predict, "mlp_blind(action-blind)": mlpb.predict, "ridge_blind(action-blind)": ridge_blind.predict, "persistence(box stays)": zero_box}
YAW_VARIANTS = {"wm3b": w3.predict, "wm2_c0": c0.predict}; ridge_yaw = lambda W: ridge.predict(W)["Y_b"][:, :, 3]
# ------------------------------------------------------------------------------------------------ data
def load_dataset(name):
    if name == "wm2_test": D = ROOT / "outputs/wm2/dataset"; idx = [e for e in json.loads((D / "index.json").read_text()) if e["kind"] == "interventional" and e["split"].startswith("test") and e["replay_ok"]]
    else: D = ROOT / "outputs/indep/dataset"; idx = [e for e in json.loads((D / "index.json").read_text()) if e["replay_ok"]]
    poses = {}
    for e in idx: poses.setdefault(e["pid"], []).append(e)
    out = []
    for pid, es in sorted(poses.items()):
        if len(es) < 10: continue
        cands = []
        for e in sorted(es, key=lambda x: x["idx"]):
            z = np.load(D / "episodes" / f"{e['eid']}.npz"); cands.append(dict(eid=e["eid"], vid=e["idx"], family=e["family"], level=e["level"], s=z["s"], a=z["a"], b=z["b"], c=z["c"], T=len(z["a"]), success=bool(e["task_success"]), pose=[e["box_xy"][0], e["box_xy"][1], e["box_yaw"]], split=e["split"]))
        out.append(dict(pid=pid, split=es[0]["split"], cands=cands))
    return out
def stack(poses):
    cl = [(pi, c) for pi, p in enumerate(poses) for c in p["cands"]]; N = len(cl); Tm = max(c["T"] for _, c in cl); A = np.zeros((N, Tm, 8), np.float32)
    for i, (_, c) in enumerate(cl): A[i, :c["T"]] = c["a"]
    hist = dict(S=np.stack([c["s"][1:5] for _, c in cl]), B=np.stack([c["b"][1:5] for _, c in cl]), C=np.stack([c["c"][1:5] for _, c in cl])); return cl, A, np.array([c["T"] for _, c in cl]), hist
if MODE == "check":
    res = {}; ok = True
    for name in ("wm2_test", "I-1"):
        poses = load_dataset(name); cl, A, Tl, hist = stack(poses); N = len(cl); W = dict(S_hist=hist["S"], B_hist=hist["B"], C_hist=hist["C"], A_hist=A[:, 0:4], A_fut=A[:, 4:24], t=np.full(N, 4), Y_e=np.zeros((N, 4), np.float32), pose=np.array([c["pose"] for _, c in cl], np.float32))
        stored = {}; wins = ROOT / ("outputs/wm2/windows" if name == "wm2_test" else "outputs/indep/windows"); prd = ROOT / ("outputs/indep_dry/preds" if name == "wm2_test" else "outputs/indep/preds"); mx_in = 0.0; mx_pred = 0.0; n_cmp = 0
        for sp in ("test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"):
            z = np.load(wins / f"{sp}.npz"); p3 = np.load(prd / f"wm3b_{sp}.npz"); key = {(str(a), int(t)): i for i, (a, t) in enumerate(zip(z["eid"], z["t"]))}
            for i, (_, c) in enumerate(cl):
                k = (c["eid"], 4)
                if k in key:
                    j = key[k]; n_cmp += 1; mx_in = max(mx_in, *[float(np.abs(W[a][i] - z[a][j]).max()) for a in ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut")])
                    Wi = {kk: v[i:i + 1] for kk, v in W.items()}; mx_pred = max(mx_pred, float(np.abs(w3.predict(Wi)["Y_b"][0] - p3["Y_b"][j]).max()))
        res[name] = dict(n_poses=len(poses), n_candidates=N, n_windows_compared=n_cmp, max_abs_diff_constructed_vs_stored_window=mx_in, max_abs_diff_wm3b_prediction_vs_stored=mx_pred); ok &= n_cmp > 100 and mx_in == 0.0 and mx_pred < 1e-4; guard("check " + name)
    res["passed"] = bool(ok); res["cpu_s"] = cpu() - C0; (OUT / "check.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)); sys.exit(0 if ok else 1)
# ------------------------------------------------------------------------------------------------ run
res = dict(status="DEVELOPMENT (privileged-state WM inputs; simulation; datasets previously analysed); NOT independent evidence", protocol=json.loads((OUT / "protocol_freeze.json").read_text()), partial=[]); wrap_ = np.arctan2
def per_pose_metrics(poses, cl, rolls, Jp_by_model, extra):
    pass
def evaluate(name, poses):
    cl, A, Tl, hist = stack(poses); N = len(cl); pose_of = np.array([pi for pi, _ in cl]); T_terms = []; tend = []
    # ground-truth terms over rows t0..t_end of the same horizon (t_end depends on the rolled length)
    nb = ((Tl - 1 - 4) // 20); tend = 4 + 20 * nb
    for i, (_, c) in enumerate(cl): T_terms.append(L.terms(c["b"][4:tend[i] + 1], c["c"][4:tend[i] + 1]))
    Jt = np.array([L.score(t) for t in T_terms]); placed_t = np.array([t["placed"] for t in T_terms]); succ = np.array([c["success"] for _, c in cl]); vids = [c["vid"] for _, c in cl]; fam = np.array([c["family"] or "control" for _, c in cl])
    settled = np.array([float(np.linalg.norm(c["b"][tend[i], :2] - c["b"][-1, :2])) < 1e-3 for i, (_, c) in enumerate(cl)])
    out = dict(n_poses=len(poses), n_candidates=N, candidates_per_pose=[len(p["cands"]) for p in poses], success_rate=float(succ.mean()), poses_with_outcome_variation=int(sum(1 for pi in range(len(poses)) if 0 < succ[pose_of == pi].sum() < (pose_of == pi).sum())),
               true_horizon_end_settled_fraction=float(settled.mean()), true_placed_fraction_at_horizon_end=float(placed_t.mean()), true_J_spread_median=float(np.median([np.ptp(Jt[pose_of == pi]) for pi in range(len(poses))])), rollout_blocks=dict(min=int(nb.min()), max=int(nb.max())))
    runs = {}
    def run_model(label, predict, yaw_mode="model", yaw_predict=None):
        Rr = L.rollout(predict, hist, A, Tl, [0, 0, 0], yaw_mode=yaw_mode, yaw_predict=yaw_predict); Jp = np.array([L.score(L.terms(r["b"], r["c"])) for r in Rr]); placed_p = np.array([L.terms(r["b"], r["c"])["placed"] for r in Rr]); pp = np.array([r["p_placed"] for r in Rr]); runs[label] = dict(Jp=Jp, placed_p=placed_p, p=pp); guard("rollout " + label)
    for lab, fn in PRED.items(): run_model(lab, fn)
    for lab, fn in YAW_VARIANTS.items(): run_model(lab + " | yaw=persistence", fn, "persist")
    rng = np.random.default_rng(0); runs["random(null)"] = dict(Jp=np.array([L.tie_key("null", f"{i}") % 100000 / 1e5 for i in range(N)]), placed_p=np.zeros(N, bool), p=np.full(N, np.nan)); runs["oracle(true J)"] = dict(Jp=Jt.copy(), placed_p=placed_t.copy(), p=placed_t.astype(float))
    def pose_stats(label, mask_fn=None):
        Jp = runs[label]["Jp"]; rows = []
        for pi, p in enumerate(poses):
            m = (pose_of == pi) if mask_fn is None else ((pose_of == pi) & mask_fn)
            if m.sum() < 3: continue
            idx = np.nonzero(m)[0]; sel = L.select(Jp[idx], p["pid"], [vids[i] for i in idx]); r, nr = L.regret(Jt[idx], sel); pa, npair = L.pairwise_accuracy(Jp[idx], Jt[idx]); ties = int((Jp[idx] - Jp[idx].min() < 1e-9).sum())
            rows.append(dict(top1_success=float(succ[idx][sel]), mean_success=float(succ[idx].mean()), oracle_success=float(succ[idx].max()), regret=r, nregret=nr, pairwise=pa, n_pairs=npair, kendall=L.kendall(Jp[idx], Jt[idx]), tied_at_top=ties, exact_oracle=float(abs(Jt[idx][sel] - Jt[idx].min()) < 1e-9)))
        return rows
    def summarise(rows):
        g = lambda k: [r[k] for r in rows]; return dict(n_poses=len(rows), top1_success=L.boot_mean(g("top1_success")), random_choice_success=float(np.mean(g("mean_success"))), oracle_success=float(np.mean(g("oracle_success"))), regret=L.boot_mean(g("regret")), normalised_regret=L.boot_mean(g("nregret")),
                                            pairwise_accuracy=L.boot_mean(g("pairwise")), kendall_tau=L.boot_mean(g("kendall")), exact_oracle_match=L.boot_mean(g("exact_oracle")), frac_poses_with_tie_at_top=float(np.mean(np.array(g("tied_at_top")) > 1)))
    per = {}; fam_res = {}
    for lab in runs:
        rows = pose_stats(lab); per[lab] = rows; fam_res[lab] = {}
        for f in FAMS:
            mk = (fam == f) | (fam == "control"); r_f = pose_stats(lab, mk)
            if r_f: fam_res[lab][f] = summarise(r_f)
    out["models"] = {lab: summarise(per[lab]) for lab in runs}; out["per_family_within_family_ranking"] = fam_res; res_pairs = {}
    for lab in ("wm3b", "wm2_c0", "hybrid_posthoc", "mlp", "ridge"):
        res_pairs[lab + "_minus_mlp_blind(action-blind)"] = dict(top1_success=L.boot_paired([r["top1_success"] for r in per[lab]], [r["top1_success"] for r in per["mlp_blind(action-blind)"]]), regret=L.boot_paired([r["regret"] for r in per[lab]], [r["regret"] for r in per["mlp_blind(action-blind)"]]), pairwise=L.boot_paired([r["pairwise"] for r in per[lab]], [r["pairwise"] for r in per["mlp_blind(action-blind)"]]))
    for lab in ("wm3b", "hybrid_posthoc", "mlp", "ridge"): res_pairs[lab + "_minus_wm2_c0"] = dict(top1_success=L.boot_paired([r["top1_success"] for r in per[lab]], [r["top1_success"] for r in per["wm2_c0"]]), regret=L.boot_paired([r["regret"] for r in per[lab]], [r["regret"] for r in per["wm2_c0"]]), pairwise=L.boot_paired([r["pairwise"] for r in per[lab]], [r["pairwise"] for r in per["wm2_c0"]]))
    out["paired_differences_over_poses"] = res_pairs
    # classification at the fixed physical operating point + calibration of p(placed) + pairwise-confidence bins
    cls = {}
    for lab in PRED.keys():
        r = runs[lab]; yf = (~placed_t).astype(int); pf = (~r["placed_p"]).astype(int); tp = int(((pf == 1) & (yf == 1)).sum()); fp = int(((pf == 1) & (yf == 0)).sum()); fn = int(((pf == 0) & (yf == 1)).sum()); tn = int(((pf == 0) & (yf == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan"); rec = tp / (tp + fn) if tp + fn else float("nan")
        def stat(ix): 
            y_ = yf[ix]; s_ = r["Jp"][ix]; return L.auroc(y_, s_)
        rng = np.random.default_rng(0); pose_ix = [np.nonzero(pose_of == pi)[0] for pi in range(len(poses))]; bs = []
        for _ in range(500):
            ix = np.concatenate([pose_ix[j] for j in rng.integers(0, len(poses), len(poses))]); v = stat(ix)
            if np.isfinite(v): bs.append(v)
        ece, brier = L.ece_brier(placed_t.astype(float), r["p"]) if (np.isfinite(r["p"]).all() and not lab.startswith("ridge")) else (float("nan"), float("nan"))      # ridge event heads are stubbed here (no probabilities) => calibration not available
        bins = []
        for lo, hi in ((0, .01), (.01, .03), (.03, .1), (.1, 9)):
            acc = []
            for pi in range(len(poses)):
                ix = pose_ix[pi]
                for a in range(len(ix)):
                    for b_ in range(a + 1, len(ix)):
                        dt = Jt[ix[a]] - Jt[ix[b_]]; dp = r["Jp"][ix[a]] - r["Jp"][ix[b_]]
                        if abs(dt) >= L.MEANINGFUL_PAIR and lo <= abs(dp) < hi: acc.append(1.0 if dp * dt > 0 else 0.0)
            bins.append(dict(abs_pred_delta=[lo, hi], pairs=len(acc), accuracy=float(np.mean(acc)) if acc else None))
        cls[lab] = dict(predicted_fail_vs_true_fail=dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, accuracy=(tp + tn) / N), auroc_Jpred_for_true_failure=[L.auroc(yf, r["Jp"]), float(np.percentile(bs, 2.5)) if bs else None, float(np.percentile(bs, 97.5)) if bs else None], p_placed_ECE=ece, p_placed_Brier=brier, prevalence_true_placed=float(placed_t.mean()), pairwise_accuracy_by_predicted_confidence=bins)
    out["risk_classification_and_calibration"] = cls
    # decision stability with yaw handling (paired, same poses and candidates)
    stab = {}
    for base in ("wm3b", "wm2_c0"):
        for v in (" | yaw=persistence",):
            lab = base + v; sel0 = []; sel1 = []; taus = []
            for pi, p in enumerate(poses):
                ix = np.nonzero(pose_of == pi)[0]; s0 = L.select(runs[base]["Jp"][ix], p["pid"], [vids[i] for i in ix]); s1 = L.select(runs[lab]["Jp"][ix], p["pid"], [vids[i] for i in ix]); sel0.append(s0); sel1.append(s1); taus.append(L.kendall(runs[base]["Jp"][ix], runs[lab]["Jp"][ix]))
            r0 = np.array([r["regret"] for r in per[base]]); r1 = np.array([r["regret"] for r in per[lab]]); stab[lab] = dict(frac_poses_selected_candidate_changed=float(np.mean(np.array(sel0) != np.array(sel1))), kendall_tau_between_score_vectors=L.boot_mean(taus), top1_success=L.boot_mean([r["top1_success"] for r in per[lab]]), regret_difference_vs_model_yaw=L.boot_paired(r1, r0), max_abs_J_change=float(np.abs(runs[lab]["Jp"] - runs[base]["Jp"]).max()))
    stab["not_run"] = "Huber yaw head and rotation-vector head: their weights were not persisted by E1-lite and re-training is not allowed in this phase (PARTIAL for this item)"
    out["yaw_handling_stability"] = stab; return out, runs, cl, A, Tl, hist
if MODE == "run":
    try:
        data = {}
        for name in ("wm2_test", "I-1"): data[name] = load_dataset(name); guard("load " + name)
        for name in ("wm2_test", "I-1"): res[name], _r, cl_, A_, Tl_, hist_ = evaluate(name, data[name]); guard("evaluate " + name)
        # ACT-reachability diagnostic
        D = ROOT / "outputs/wm2/dataset"; idx = [e for e in json.loads((D / "index.json").read_text()) if e["kind"] == "observational" and e["agent"] == "act" and e["replay_ok"]]; feats = []
        def fe(a): da = np.diff(a[:, :7], axis=0); return [float(np.abs(da).mean()), float(np.abs(da).max()), float(np.abs(np.diff(a[:, :7], 2, axis=0)).mean()), float(a[:, 7].min())]
        for e in idx[::4]: feats.append(fe(np.load(D / "episodes" / f"{e['eid']}.npz")["a"]))
        feats = np.array(feats); lo_, hi_ = np.percentile(feats, 1, 0), np.percentile(feats, 99, 0); act_cov = {}
        for name in ("wm2_test", "I-1"):
            byf = {}
            for p in data[name]:
                for c in p["cands"]:
                    f = np.array(fe(c["a"])); ok = bool(((f >= lo_) & (f <= hi_)).all()); byf.setdefault(c["family"] or "control", []).append(ok)
            act_cov[name] = {k: dict(n=len(v), within_ACT_1_99_percentile_envelope=float(np.mean(v))) for k, v in byf.items()}
        res["ACT_reachability_diagnostic"] = dict(features=["mean |da| (arm)", "max |da| (arm)", "mean |d2a|", "min gripper cmd"], n_ACT_episodes=len(feats), p1=lo_.tolist(), p99=hi_.tolist(), coverage_by_family=act_cov, note="scripted variants are physically valid (clip + slew applied) but only those inside the ACT action envelope are plausible policy outputs")
        # latency (single thread)
        lat = {}
        poses = data["wm2_test"]; cl, A, Tl, hist = stack(poses[:1]); n1 = len(cl); W1 = dict(S_hist=hist["S"][:1], B_hist=hist["B"][:1], C_hist=hist["C"][:1], A_hist=A[:1, 0:4], A_fut=A[:1, 4:24], t=np.array([4]), Y_e=np.zeros((1, 4), np.float32), pose=np.zeros((1, 3), np.float32))
        for lab in ("wm3b", "wm2_c0", "hybrid_posthoc", "ridge", "mlp"):
            fn = PRED[lab]
            with torch.no_grad():
                for _ in range(5): fn(W1)
                ts = []
                for _ in range(60): t_ = time.perf_counter(); fn(W1); ts.append((time.perf_counter() - t_) * 1000)
                tk = []
                for _ in range(3): t_ = time.perf_counter(); L.rollout(fn, hist, A, Tl, [0, 0, 0]); tk.append((time.perf_counter() - t_) * 1000)
            lat[lab] = dict(per_candidate_one_block_ms_median=float(np.median(ts)), per_25_candidate_full_ranking_ms_median=float(np.median(tk)), candidates=n1, blocks=int(((Tl - 5) // 20).max())); guard("latency " + lab)
        res["latency_single_thread"] = lat
    except Stop as e: res["partial"].append(f"STOPPED by the CPU guard at: {e}")
    res["budget"] = dict(cap_cpu_hours=0.1, cpu_seconds=cpu() - C0, cpu_hours=(cpu() - C0) / 3600, wall_seconds=time.time() - T0, complete=not res["partial"]); (OUT / "results.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))); print(json.dumps(res["budget"]), res["partial"])
