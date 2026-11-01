#!/usr/bin/env python3
"""INDEPENDENT held-out evaluation: MODEL SIDE.  Loads FROZEN models only (no training, no tuning, no re-fit of any learned network; the closed-form ridge / logistic baselines are re-fitted deterministically from the frozen
wm2 train windows and verified against the frozen weight hashes) and writes predictions for the independent windows into outputs/indep/preds.  The independent evaluator (eval/indep_evaluate.py) reads ONLY those files + the labels.
Modes:  python eval/indep_predict.py            -> independent set  (requires outputs/indep/protocol_freeze.json, verified)
        python eval/indep_predict.py --dry-run  -> DEVELOPMENT plumbing check on the frozen wm2 test windows, written to outputs/indep_dry/ and compared with the stored development predictions.
Run (ACT venv)."""
import os, sys, json, time, hashlib, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); DRY = "--dry-run" in sys.argv; sys.argv = [sys.argv[0], "none"]
import wm2_run as R, wm2_models as M, wm2_gates as G1, wm3b_model as B
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); FAMS = ["depth", "speed", "timing", "release_height", "noise"]; EVAL = ["test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]
BASE = ROOT / "outputs/indep_dry" if DRY else ROOT / "outputs/indep"; PRED = BASE / "preds"; WINDIR = R.WIN if DRY else ROOT / "outputs/indep/windows"; FAMF = ROOT / "outputs/wm3/family_windows.npz" if DRY else ROOT / "outputs/indep/family_windows.npz"
MODELS = {"wm2_c0": ROOT / "outputs/wm2_1/models/gru_c0.pt", "mlp": ROOT / "outputs/wm2_1/models/mlp.pt", "mlp_blind": ROOT / "outputs/wm2_1/models/mlp_blind.pt", "wm3b": ROOT / "outputs/wm3/models/wm3b.pt"}
def hash_ridge(r): return hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in (r.Wt, r.mu, r.sd))).hexdigest()  # deterministic closed-form part; logistic event heads (LBFGS float32) reproduce only to ~1e-3 in probability and are NOT hashed.hexdigest()
def hash_oo(o): return hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in (o.mu, o.sd))).hexdigest()
def save(name, d): np.savez_compressed(PRED / f"{name}.npz", **d)
def main():
    T0 = time.time(); PRED.mkdir(parents=True, exist_ok=True); F = None
    if not DRY:
        F = json.loads((BASE / "protocol_freeze.json").read_text()); bad = [k for k, v in F["files_sha256"].items() if sha(ROOT / k) != v]
        if bad: sys.exit(f"FROZEN FILES CHANGED: {bad}")
    R.LOADED.clear(); tr = R.load("train"); nrm = M.Norm(tr); torch.set_num_threads(3)
    ridge = M.Ridge("cond").fit(nrm, tr); ridge_blind = M.Ridge("blind").fit(nrm, tr); oo = M.OutcomeOnly().fit(nrm, tr); prior = tr["Y_e"].mean(0); hashes = dict(ridge=hash_ridge(ridge), ridge_blind=hash_ridge(ridge_blind), outcome_only=hash_oo(oo))
    if F: 
        for k, v in hashes.items():
            if v != F["closed_form_baselines_sha256"][k]: sys.exit(f"baseline {k} differs from frozen weights: {v}")
    def load_nn(kind, path):
        nnm = M.NN(kind); nnm.nrm = nrm; nnm.model = M.GRUModel() if kind == "gru" else M.MLP(280 if kind == "mlp_cond" else 88); nnm.model.load_state_dict(torch.load(path)); nnm.model.eval(); return nnm
    c0 = load_nn("gru", MODELS["wm2_c0"]); mlp = load_nn("mlp_cond", MODELS["mlp"]); mlpb = load_nn("mlp_blind", MODELS["mlp_blind"]); m = B.WM3B(ridge, nrm); m.model = B.GRUOC(); m.model.load_state_dict(torch.load(MODELS["wm3b"])); m.model.eval()
    def all_models(W, with_shuffle=False):
        out = {}; out["persistence"], out["constvel"] = R.classical_preds(W, prior); rp = ridge.predict(W); out["ridge"] = rp; out["ridge_blind"] = ridge_blind.predict(W); out["outcome_only"] = oo.predict(W)
        out["mlp"] = mlp.predict(W); out["mlp_blind"] = mlpb.predict(W); cp = c0.predict(W); out["wm2_c0"] = cp; out["hybrid_posthoc"] = dict(Y_s=rp["Y_s"], Y_b=cp["Y_b"], Y_c=cp["Y_c"], Y_e=cp["Y_e"]); out["wm3b"] = m.predict(W); return out
    for sp in EVAL:
        W = {k: v for k, v in np.load(WINDIR / f"{sp}.npz").items()}
        for n, d in all_models(W).items(): save(f"{n}_{sp}", d)
        if sp == "test_id":
            part = G1.shuffle_partner(W["group"], 0); Ws = dict(W); Ws["A_hist"], Ws["A_fut"] = W["A_hist"][part], W["A_fut"][part]; save("wm3b_shuffle_test_id", m.predict(Ws)); save("wm2_c0_shuffle_test_id", c0.predict(Ws))
            rn = m.tensors(W)[2].numpy(); save("wm3b_ablate_rfc_zero_test_id", m.predict(W, rfc_override=np.zeros_like(rn))); save("wm3b_ablate_rfc_shuffled_test_id", m.predict(W, rfc_override=rn[np.random.default_rng(0).permutation(len(rn))]))
    fw = np.load(FAMF)
    for f in FAMS:
        Wf = {k.split("/", 1)[1]: fw[k] for k in fw.files if k.startswith(f + "/")}; Wf["group"] = np.zeros(len(Wf["pid"]), np.int64); Wf["pose"] = np.zeros((len(Wf["pid"]), 3), np.float32); Wf["t"] = Wf["t"]
        for n, d in all_models(Wf).items(): save(f"{n}_family_{f}", d)
    # ---- latency (same machine, 1 thread, batch 1): WM3-B full path exactly as in the development benchmark; others via predict()
    Wb = {k: v[:1] for k, v in tr.items()}; hist, af, rn_, _ = m.tensors(Wb); Wt = ridge.Wt; mu, sd = ridge.mu, ridge.sd; xcond = nrm.feat(Wb, "cond").astype(np.float64)
    def full(x):
        xx = (xcond - mu) / sd; x1 = np.concatenate([xx, np.ones((1, 1))], 1); rf = (x1 @ Wt[:, :160]).astype(np.float32); rft = torch.tensor(((rf.reshape(1, 20, 8) - nrm.Ys[0]) / nrm.Ys[1]).astype(np.float32)); return m.model(hist, af, rft)
    lat = {"wm3b": M.latency(full, lambda: None)}; lat["wm3b"].update(params=int(sum(q.numel() for q in m.model.parameters())) + int(Wt[:, :160].size), note="ridge matmul (281x160) + GRU-OC, batch 1, 1 thread, identical to the development benchmark")
    for n, mm in (("wm2_c0", c0), ("mlp", mlp), ("mlp_blind", mlpb)): lat[n] = M.latency(lambda x, mm=mm: mm.predict(Wb), lambda: None); lat[n]["params"] = int(sum(q.numel() for q in mm.model.parameters()))
    lat["ridge"] = M.latency(lambda x: ridge.predict(Wb), lambda: None); torch.set_num_threads(3)
    rep = dict(mode="dry-run (development windows)" if DRY else "independent", closed_form_hashes=hashes, model_checkpoint_sha256={k: sha(p) for k, p in MODELS.items()}, latency=lat, splits_loaded_for_baseline_refit=sorted(set(R.LOADED)), total_wall_s=time.time() - T0, torch=torch.__version__, threads_latency=1)
    (BASE / "predict_report.json").write_text(json.dumps(rep, indent=1, default=str)); print(json.dumps({k: v for k, v in rep.items() if k != "latency"}, indent=1)); print({k: round(v["median_ms"], 2) for k, v in lat.items()})
    if DRY:                                    # plumbing check against the stored development predictions
        ref = ROOT / "outputs/wm3/preds"; worst = {}
        for f in sorted(PRED.glob("*.npz")):
            r = ref / f.name
            if not r.exists(): continue
            a, b = np.load(f), np.load(r); worst[f.stem] = max(float(np.abs(a[k] - b[k]).max()) for k in a.files if k in b.files)
        bad = {k: v for k, v in worst.items() if v > 1e-4}; print("dry-run max abs diff vs stored development predictions: files", len(worst), "worst", max(worst.values()), "files above 1e-4:", bad); (BASE / "dry_compare.json").write_text(json.dumps({"worst": worst, "above_1e-4": bad}, indent=1))
if __name__ == "__main__": main()
