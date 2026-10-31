#!/usr/bin/env python3
"""WM-2 runner.  Stages:  classical  -> persistence / constant-velocity baselines through the gate plumbing (no learned model; run BEFORE the gates are frozen)
                         learned    -> ridge / MLP / action-blind / outcome-only baselines and the GRU prototype (run only AFTER `wm2_gates.py --freeze` and the dataset freeze)
Only splits train/val are loaded for fitting and model selection (logged).  Run (ACT venv): python eval/wm2_run.py classical | learned"""
import sys, json, time, hashlib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "data"))
import wm2_gates as G
WIN = ROOT / "outputs/wm2/windows"; OUT = ROOT / "outputs/wm2"; SPLITS = ["train", "val", "test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]; EVAL = SPLITS[2:]; KEYS = ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut", "Y_s", "Y_b", "Y_c", "Y_e")
LOADED = []
def load(split):
    LOADED.append(split); z = np.load(WIN / f"{split}.npz"); return {k: z[k] for k in z.files}
def truth(W): return {k: W[k] for k in KEYS}
def classical_preds(W, prior_e):
    n = len(W["t"]); s_last = W["S_hist"][:, -1] - W["S_hist"][:, -2]; b_last = W["B_hist"][:, -1] - W["B_hist"][:, -2]
    pers = dict(Y_s=np.zeros((n, 20, 8), np.float32), Y_b=np.zeros((n, 20, 7), np.float32), Y_c=np.repeat(W["C_hist"][:, -1:], 20, 1), Y_e=np.tile(prior_e, (n, 1)).astype(np.float32))
    cv = dict(Y_s=np.repeat(s_last[:, None], 20, 1).astype(np.float32), Y_b=np.repeat(b_last[:, None], 20, 1).astype(np.float32), Y_c=pers["Y_c"], Y_e=pers["Y_e"]); return pers, cv
def table(preds, T, groups, active, models, splits):
    out = {}
    for m in models:
        out[m] = {}
        for sp in splits:
            out[m][sp] = {}
            for sub_name, mask in (("active", active[sp]), ("all", np.ones(len(groups[sp]), bool))):
                out[m][sp][sub_name] = {}
                for k in (1, 5, 10, 20):
                    e = G.errors(G.sub(preds[m][sp], mask), G.sub(T[sp], mask), k); out[m][sp][sub_name][f"K={k}"] = {met: float(G.macro(v, groups[sp][mask])) for met, v in e.items()}
    return out
def classical():
    tr = load("train"); prior = tr["Y_e"].mean(0); T = {}; groups = {}; active = {}; WIN_ = {}; preds = {"persistence": {}, "constvel": {}}
    for sp in EVAL:
        W = load(sp); T[sp] = truth(W); groups[sp] = W["group"]; active[sp] = G.active_mask(W); WIN_[sp] = W; p, c = classical_preds(W, prior); preds["persistence"][sp] = p; preds["constvel"][sp] = c
    tab = table(preds, T, groups, active, ["persistence", "constvel"], EVAL)
    # plumbing check on real data: persistence plays the model, constant-velocity the only baseline  (must FAIL G1: it cannot be 20 % better than itself)
    cfg = G.CONFIG; fake = {"gru": preds["persistence"], "persistence": preds["persistence"], "constvel": preds["constvel"]}
    g1 = G.gate_G1(cfg, fake, T, groups, active); g2 = G.gate_G2(cfg, {**fake, "ridge": preds["constvel"], "mlp": preds["constvel"]}, T, groups, active)
    fake["outcome_only"] = preds["persistence"]; masks = {(lab, sp): G.at_risk_mask(WIN_[sp], lab) for lab, sp in cfg["G3"]["required"]}; g3 = G.gate_G3(cfg, fake, T, groups, masks)
    res = dict(errors=tab, plumbing_check=dict(G1_verdict_persistence_vs_constvel=g1["verdict"], G1_detail=g1["detail"], G2_verdict=g2["verdict"], G3_evaluability={k: v["evaluable"] for k, v in g3["detail"].items()}, G3_sufficiency={k: v["sufficiency"] for k, v in g3["detail"].items()}),
               active_windows={sp: int(active[sp].sum()) for sp in EVAL}, groups={sp: int(len(np.unique(groups[sp]))) for sp in EVAL}, splits_loaded=sorted(set(LOADED)))
    (OUT / "classical_baselines.json").write_text(json.dumps(res, indent=1)); print(json.dumps({k: res[k] for k in ("plumbing_check", "active_windows", "groups")}, indent=1)[:3000])
    for sp in EVAL: print(sp, "persist K=20", {k: round(v, 3) for k, v in tab["persistence"][sp]["active"]["K=20"].items()}, "constvel K=20", {k: round(v, 3) for k, v in tab["constvel"][sp]["active"]["K=20"].items()})

import os, wm2_models as M, torch
DRY = bool(os.environ.get("WM2_DRY")); TAG = "dry_" if DRY else ""
PRED = OUT / "preds"; FROZEN = json.loads((OUT / "freeze.json").read_text())
def check_freeze():
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); bad = []
    for k, p in (("gates_config_sha256", OUT / "gates_config.json"), ("model_config_sha256", OUT / "model_config.json")):
        if sha(p) != FROZEN[k]: bad.append(k)
    if sha(OUT / "dataset/index.json") != FROZEN["dataset"]["index_sha256"]: bad.append("dataset_index")
    for sp, h in FROZEN["windows"].items():
        if sha(WIN / f"{sp}.npz") != h: bad.append(f"windows_{sp}")
    G.load_config(FROZEN["gates_config_sha256"])
    if bad: sys.exit(f"FROZEN FILES CHANGED: {bad}")
def gru_smoke(nrm, Wtr, Wva):
    """Smoke tests of the GRU prototype (frozen list): tiny-batch overfit, shapes/finiteness, determinism, action sensitivity, finite gradients."""
    res = {}; sub = {k: v[:256] for k, v in Wtr.items()}; log = []
    m = M.NN("gru"); m.fit(nrm, sub, {k: v[:256] for k, v in Wva.items()}, log, seed=0, epochs=100)          # 100 epochs x 2 batches = 200 steps (frozen smoke spec)
    l0, l1 = log[0]["train_loss"], min(r["train_loss"] for r in log); res["overfit_loss_first_to_best"] = [l0, l1, 1 - l1 / l0]
    m2 = m; log2 = log; res["overfit_200_steps_loss_drop"] = 1 - min(r["train_loss"] for r in log2) / log2[0]["train_loss"]
    p = m2.predict({k: v[:64] for k, v in Wtr.items()}); res["shapes"] = {k: list(v.shape) for k, v in p.items()}; res["finite"] = bool(all(np.isfinite(v).all() for v in p.values()))
    Wa = {k: v[:64].copy() for k, v in Wtr.items()}; Wb = {k: v.copy() for k, v in Wa.items()}; Wb["A_fut"] = Wb["A_fut"] + 0.1; pa, pb = m2.predict(Wa), m2.predict(Wb); res["action_sensitivity_mean_abs_change"] = float(np.abs(pa["Y_b"] - pb["Y_b"]).mean())
    def run_losses(seed):
        torch.manual_seed(seed); mm = M.NN("gru"); lg = []; mm.fit(nrm, sub, {k: v[:256] for k, v in Wva.items()}, lg, seed=seed, epochs=1, max_steps=5); return [r["train_loss"] for r in lg]
    a, b = run_losses(0), run_losses(0); res["determinism_same_seed_identical_loss"] = bool(a == b)
    mm = M.GRUModel(); x = (torch.tensor(nrm.hist(sub)), torch.tensor(nrm.n(sub["A_fut"], nrm.A))); y = torch.tensor(nrm.targets(sub)); pr, lg_ = mm(*x); L = M.loss_fn(pr, y, torch.tensor(sub["Y_e"]), lg_); L.backward(); res["gradients_finite"] = bool(all(torch.isfinite(p.grad).all() for p in mm.parameters() if p.grad is not None))
    res["n_params"] = int(sum(p.numel() for p in mm.parameters())); res["passed"] = bool(res["overfit_200_steps_loss_drop"] >= 0.8 and res["finite"] and res["action_sensitivity_mean_abs_change"] > 1e-6 and res["determinism_same_seed_identical_loss"] and res["gradients_finite"]); return res
def smoke():
    check_freeze(); tr, va = load("train"), load("val"); nrm = M.Norm(tr); r = gru_smoke(nrm, tr, va); (OUT / "gru_smoke.json").write_text(json.dumps(r, indent=1)); print(json.dumps(r, indent=1)); sys.exit(0 if r["passed"] else 1)
DRYKW = dict(epochs=1, max_steps=20) if DRY else {}
def learned():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    check_freeze(); T0 = time.time(); (OUT / "preds").mkdir(exist_ok=True); cfg = G.load_config(FROZEN["gates_config_sha256"]); LOADED.clear(); tr, va = load("train"), load("val"); nrm = M.Norm(tr); logs = []; times = {}; models = {}
    smoke_res = json.loads((OUT / "gru_smoke.json").read_text())
    def timed(name, fn):
        t = time.time(); m = fn(); times[name] = time.time() - t; print(f"[{name}] fitted in {times[name]:.0f}s", flush=True); models[name] = m
    timed("ridge", lambda: M.Ridge("cond").fit(nrm, tr)); timed("ridge_blind", lambda: M.Ridge("blind").fit(nrm, tr)); timed("outcome_only", lambda: M.OutcomeOnly().fit(nrm, tr))
    timed("mlp", lambda: M.NN("mlp_cond").fit(nrm, tr, va, logs, seed=0, **DRYKW)); timed("mlp_blind", lambda: M.NN("mlp_blind").fit(nrm, tr, va, logs, seed=0, **DRYKW)); timed("gru", lambda: M.NN("gru").fit(nrm, tr, va, logs, seed=0, **DRYKW))
    loaded_fit = sorted(set(LOADED)); prior = tr["Y_e"].mean(0)
    # ---- evaluation data (test splits only now)
    T = {}; groups = {}; active = {}; WIN_ = {}; preds = {m: {} for m in ["persistence", "constvel", "ridge", "mlp", "ridge_blind", "mlp_blind", "outcome_only", "gru", "gru_shuffle", "gru_actionzero"]}
    for sp in EVAL:
        W = load(sp); WIN_[sp] = W; T[sp] = truth(W); groups[sp] = W["group"]; active[sp] = G.active_mask(W); p, c = classical_preds(W, prior); preds["persistence"][sp] = p; preds["constvel"][sp] = c
        for name in ("ridge", "mlp", "ridge_blind", "mlp_blind", "outcome_only", "gru"): preds[name][sp] = models[name].predict(W)
        if sp in ("test_id",):
            part = G.shuffle_partner(W["group"], 0); Ws = dict(W); Ws["A_hist"], Ws["A_fut"] = W["A_hist"][part], W["A_fut"][part]; preds["gru_shuffle"][sp] = models["gru"].predict(Ws)
            Wz = dict(W); Wz["A_hist"], Wz["A_fut"] = np.zeros_like(W["A_hist"]), np.zeros_like(W["A_fut"]); preds["gru_actionzero"][sp] = models["gru"].predict(Wz)
    pz = np.load(WIN / "pairs.npz"); pairs = {k: pz[k] for k in pz.files}; pair_preds = {"gru": models["gru"].predict(pairs)}
    # ---- gates
    masks = {(lab, sp): G.at_risk_mask(WIN_[sp], lab) for lab, sp in cfg["G3"]["required"]}
    g1 = G.gate_G1(cfg, preds, T, groups, active); g2 = G.gate_G2(cfg, preds, T, groups, active); g3 = G.gate_G3(cfg, preds, T, groups, masks); g4 = G.gate_G4(cfg, preds, T, groups, active, pairs, pair_preds); g5 = G.gate_G5(cfg, preds, T, groups, masks, g3)
    # ---- latency (same process/VM; 1 thread; batch 1)
    Wb = {k: v[:1] for k, v in tr.items()}; g = models["gru"]; hist, af = g._inp(Wb); lat = M.latency(lambda x: g.model(*x), lambda: (hist, af)); lat.update(params=int(g.n_params), matmul_probe_s=float(np.mean([(lambda t: (np.random.rand(1500, 1500).astype(np.float32) @ np.random.rand(1500, 1500).astype(np.float32), time.perf_counter() - t)[1])(time.perf_counter()) for _ in range(3)])))
    for name in ("mlp", "ridge"):
        m = models[name]
        if name == "mlp": x = m._inp(Wb); lat[f"{name}_median_ms"] = M.latency(lambda xx: m.model(*xx), lambda: x)["median_ms"]
    lat["ACT_per_query_ms_reference"] = "70-86 ms median measured in xsweep/coverage evaluations (same VM family, 1 thread); not re-measured in this session"
    g6 = G.gate_G6(cfg, lat)
    # ---- G0
    idx = json.loads((OUT / "dataset/index.json").read_text()); man = json.loads((OUT / "dataset/dataset_manifest.json").read_text()); g2s = {}
    for e in idx: g2s.setdefault(tuple(e["group"]), set()).add(e["split"])
    I = man["integrity"]; g0info = dict(dataset_freeze_hash_matches=True, groups_in_multiple_splits_zero=sum(len(v) > 1 for v in g2s.values()) == 0, duplicate_action_hashes_zero=I["duplicate_action_hashes"] == 0, no_forbidden_seeds=not I["forbidden_seeds"], interventional_variants_share_split=I["poses_with_multiple_splits"] == 0,
                  no_interventional_group_overlap_with_observational=I["interventional_groups_overlapping_observational_groups"] == 0, window_tests_pass=window_checks(idx), training_loaded_only_train_val=set(loaded_fit) <= {"train", "val"}, originals_untouched=hashlib.sha256((ROOT / "outputs/wm1/dataset/index.json").read_bytes()).hexdigest() == FROZEN["wm1_index_sha256"])
    g0 = G.gate_G0(cfg, g0info)
    gates = dict(G0=g0, G1=g1, G2=g2, G3=g3, G4=g4, G5=g5, G6=g6)
    # ---- error tables, horizons, analysis
    allm = ["persistence", "constvel", "ridge", "mlp", "ridge_blind", "mlp_blind", "gru", "gru_shuffle", "gru_actionzero"]; tab = {}
    for m in allm:
        tab[m] = {}
        for sp in (EVAL if m not in ("gru_shuffle", "gru_actionzero") else ["test_id"]):
            tab[m][sp] = {}
            for sub_name, mask in (("active", active[sp]), ("all", np.ones(len(groups[sp]), bool))): tab[m][sp][sub_name] = {f"K={k}": {met: float(G.macro(v, groups[sp][mask])) for met, v in G.errors(G.sub(preds[m][sp], mask), G.sub(T[sp], mask), k).items()} for k in (1, 5, 10, 20)}
    fam = {}
    for sp in EVAL:
        W = WIN_[sp]; e = G.errors(preds["gru"][sp], T[sp], 20)["box_mm"]; ep = G.errors(preds["persistence"][sp], T[sp], 20)["box_mm"]
        for f, nm in ((-1, "observational"), (0, "control"), (1, "depth"), (2, "timing"), (3, "noise"), (4, "speed"), (5, "release_height")):
            m_ = W["family"] == f
            if m_.sum() > 20: fam.setdefault(sp, {})[nm] = dict(n=int(m_.sum()), gru_box_mm=float(e[m_].mean()), persistence_box_mm=float(ep[m_].mean()))
    results = dict(frozen=dict(gates_config_sha256=FROZEN["gates_config_sha256"], model_config_sha256=FROZEN["model_config_sha256"], dataset_index_sha256=FROZEN["dataset"]["index_sha256"]), gates={k: dict(verdict=v["verdict"], conditions=v["conditions"]) for k, v in gates.items()}, gate_details={k: {kk: vv for kk, vv in v.items() if kk not in ("conditions", "verdict")} for k, v in gates.items()},
                   errors=tab, error_by_family_gru_vs_persistence=fam, training_log=logs, fit_seconds=times, best_epochs={n: getattr(m, "best_epoch", None) for n, m in models.items() if hasattr(m, "best_epoch")}, smoke=smoke_res, splits_loaded_during_fit=loaded_fit, latency=lat, total_wall_s=time.time() - T0, n_windows={sp: int(len(groups[sp])) for sp in EVAL})
    (OUT / f"{TAG}wm2_results.json").write_text(json.dumps(results, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    np.savez_compressed(OUT / "preds/gru_pairs.npz", **{f"P_{k}": v for k, v in pair_preds["gru"].items()})
    # ---- plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 4)); Ks = list(range(1, 21)); COL = dict(persistence="gray", constvel="silver", ridge="C0", mlp="C1", ridge_blind="C9", mlp_blind="C3", gru="k", gru_shuffle="C2")
    for j, met in enumerate(("box_mm", "joint_rad", "aoffset_cm")):
        for m in ("persistence", "constvel", "ridge", "mlp", "ridge_blind", "mlp_blind", "gru", "gru_shuffle"):
            sp = "test_id"; mask = active[sp]; ys = [G.macro(G.errors(G.sub(preds[m][sp], mask), G.sub(T[sp], mask), k)[met], groups[sp][mask]) for k in Ks]; ax[j].plot(Ks, ys, color=COL[m], label=m, lw=2 if m == "gru" else 1)
        ax[j].set_xlabel("horizon k"); ax[j].set_ylabel(met); ax[j].set_title(f"test_id (active), {met}")
    ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT / f"{TAG}W1_horizon_errors.png", dpi=120); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8)); lg = [r for r in logs if r["model"] == "gru"]; ax[0].plot([r["train_loss"] for r in lg], label="train"); ax[0].plot([r["val_loss"] for r in lg], label="val"); ax[0].set_title("GRU loss"); ax[0].legend(); ax[0].set_xlabel("epoch")
    for nm in ("mlp_cond", "mlp_blind"): ll = [r for r in logs if r["model"] == nm]; ax[1].plot([r["val_loss"] for r in ll], label=nm + " val")
    ax[1].legend(); ax[1].set_title("MLP val loss"); plt.tight_layout(); plt.savefig(OUT / f"{TAG}W2_training_curves.png", dpi=120); plt.close()
    ev = [(k, v) for k, v in gates["G5"]["detail"].items() if v.get("reliability")]
    if ev:
        fig, ax = plt.subplots(1, len(ev), figsize=(4 * len(ev), 3.8), squeeze=False)
        for a_, (k, v) in zip(ax[0], ev): rr = [r for r in v["reliability"] if r["n"] > 0]; a_.plot([0, 1], [0, 1], "k:"); a_.plot([r["mean_pred"] for r in rr], [r["frac_pos"] for r in rr], "-o"); a_.set_title(f"{k}\nECE {v['ECE']:.3f} Brier {v['Brier']:.3f} prev {v['prevalence']:.3f}", fontsize=8); a_.set_xlabel("predicted"); a_.set_ylabel("observed")
        plt.tight_layout(); plt.savefig(OUT / f"{TAG}W3_reliability.png", dpi=120); plt.close()
    P, Tt = pair_preds["gru"], pairs; qa = lambda W_: (pairs["B_hist"][:, -1, 6] + np.cumsum(W_["Y_b"], 1)[:, 19, 6]) * 100; fig, ax = plt.subplots(1, 2, figsize=(10, 3.8)); m_ = (pairs["family"] == 1) | (pairs["vid"] == 0); lv = np.where(pairs["family"] == 1, pairs["level"], 0.0) * 100
    for lvl in sorted(set(lv[m_])): s_ = m_ & (lv == lvl); ax[0].scatter([lvl], [qa(Tt)[s_].mean()], color="k", s=30); ax[0].scatter([lvl], [qa(P)[s_].mean()], color="C3", marker="x", s=40); ax[1].scatter([lvl], [P["Y_e"][s_][:, 1].mean()], color="C3", marker="x"); ax[1].scatter([lvl], [Tt["Y_e"][s_][:, 1].mean()], color="k")
    ax[0].set_xlabel("extra TCP-forward grasp shift [cm]"); ax[0].set_ylabel("a-offset at t+K [cm]"); ax[0].set_title("paired depth family: true (black) vs GRU (red x)"); ax[1].set_ylabel("palm contact in horizon: true vs GRU probability"); ax[1].set_xlabel("depth level [cm]"); plt.tight_layout(); plt.savefig(OUT / f"{TAG}W4_paired_depth.png", dpi=120); plt.close()
    print(json.dumps({k: v["verdict"] for k, v in gates.items()}), f"total {time.time()-T0:.0f}s")
def window_checks(idx):
    rng = np.random.default_rng(0); bad = 0; n = 0
    for sp in ("train", "test_id", "test_ood_low_strict"):
        z = np.load(WIN / f"{sp}.npz"); N = len(z["t"]); eids = z["eid"]
        for i in rng.choice(N, 300, replace=False):
            ep = np.load(OUT / "dataset/episodes" / f"{eids[i]}.npz"); t = int(z["t"][i]); n += 1
            ok_ = np.array_equal(z["S_hist"][i][-1], ep["s"][t]) and np.array_equal(z["A_fut"][i][0], ep["a"][t]) and np.array_equal(z["A_hist"][i][-1], ep["a"][t - 1]) and t - 4 >= 0 and t + 20 <= len(ep["a"]) - 1 and np.allclose(z["Y_s"][i][0], ep["s"][t + 1] - ep["s"][t])
            bad += int(not ok_)
    return bad == 0

if __name__ == "__main__":
    if sys.argv[1] == "classical": classical()
    elif sys.argv[1] == "smoke": smoke()
    elif sys.argv[1] == "learned": learned()
