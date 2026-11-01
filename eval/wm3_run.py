#!/usr/bin/env python3
"""WM3-B one-seed pilot: TRAINER side.  Loads ONLY train/val for fitting (logged); trains WM3-B (frozen configs, hashes checked); then writes predictions (standard windows, family windows, shuffle/ablation diagnostics)
for the candidate, the frozen WM2-C0, the post-hoc hybrid and the baselines into outputs/wm3/preds for the independent evaluator (eval/wm3_evaluate.py).  Run (ACT venv): python eval/wm3_run.py"""
import os, sys, json, time, hashlib, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); sys.argv = [sys.argv[0], "none"]
import wm2_run as R, wm2_models as M, wm2_gates as G1, wm3_gates as V, wm3b_model as B
OUT = ROOT / "outputs/wm3"; PRED = OUT / "preds"; REF = ROOT / "outputs/wm2_1/preds"; FROZEN = json.loads((OUT / "freeze.json").read_text()); FAMS = ["depth", "speed", "timing", "release_height", "noise"]
def check_freeze():
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); bad = []
    for k, p in (("gates_v2_config_sha256", OUT / "gates_v2_config.json"), ("model_config_wm3b_sha256", OUT / "model_config_wm3b.json"), ("family_windows_sha256", OUT / "family_windows.npz"), ("contact_scales_sha256", OUT / "contact_scales.json")):
        if sha(p) != FROZEN[k]: bad.append(k)
    if sha(ROOT / "outputs/wm2/dataset/index.json") != FROZEN["wm2_dataset_index_sha256"]: bad.append("wm2_index")
    for sp, h in FROZEN["wm2_windows_sha256"].items():
        if sha(R.WIN / f"{sp}.npz") != h: bad.append(f"windows_{sp}")
    V.load_config(FROZEN["gates_v2_config_sha256"])
    if bad: sys.exit(f"FROZEN FILES CHANGED: {bad}")
def save(name, d): np.savez_compressed(PRED / f"{name}.npz", **d)
def main():
    check_freeze(); T0 = time.time(); R.LOADED.clear(); tr, va = R.load("train"), R.load("val"); nrm = M.Norm(tr); logs = []; t = time.time(); ridge = M.Ridge("cond").fit(nrm, tr); ridge_s = time.time() - t
    print(f"ridge {ridge_s:.0f}s; training WM3-B ({len(tr['t'])} train windows)", flush=True); torch.set_num_threads(3); m = B.WM3B(ridge, nrm)
    if os.environ.get("WM3_REUSE"):          # reuse the weights trained in the first (crashed-after-training) run: per-epoch log was lost, best epoch/time from stdout
        m.model = B.GRUOC(); m.model.load_state_dict(torch.load(OUT / "models/wm3b.pt")); m.model.eval(); m.n_params = sum(q.numel() for q in m.model.parameters()); m.best_epoch = 19; m.train_seconds = 2137.0; logs = [dict(note="per-epoch training log LOST: first run crashed after training (MLP input-size bug in reference loading); weights reloaded; best epoch 19 and 2137 s taken from stdout")]
    else: m.fit(tr, va, logs, seed=0, epochs=20)
    train_s = m.train_seconds; loaded_fit = sorted(set(R.LOADED)); print(f"trained in {train_s:.0f}s best epoch {m.best_epoch}", flush=True)
    torch.save(m.model.state_dict(), OUT / "models/wm3b.pt"); ridge_blind = M.Ridge("blind").fit(nrm, tr); prior = tr["Y_e"].mean(0)
    # reference models for family windows (weights from the bit-exact WM-2.1 reproduction)
    def load_nn(kind, path):
        nnm = M.NN(kind); nnm.nrm = nrm; nnm.model = M.GRUModel() if kind == "gru" else M.MLP(280 if kind == "mlp_cond" else 88); nnm.model.load_state_dict(torch.load(path)); nnm.model.eval(); return nnm
    c0 = load_nn("gru", ROOT / "outputs/wm2_1/models/gru_c0.pt"); mlp = load_nn("mlp_cond", ROOT / "outputs/wm2_1/models/mlp.pt"); mlpb = load_nn("mlp_blind", ROOT / "outputs/wm2_1/models/mlp_blind.pt")
    # ---- standard windows
    for sp in R.EVAL:
        W = R.load(sp); pc, cv = R.classical_preds(W, prior); save(f"persistence_{sp}", pc); save(f"constvel_{sp}", cv)
        for nm in ("ridge", "mlp", "ridge_blind", "mlp_blind", "outcome_only"): save(f"{nm}_{sp}", {k: np.load(REF / f"{nm}_{sp}.npz")[k] for k in np.load(REF / f"{nm}_{sp}.npz").files})
        c0p = {k: np.load(REF / f"gru_{sp}.npz")[k] for k in ("Y_s", "Y_b", "Y_c", "Y_e")}; save(f"wm2_c0_{sp}", c0p); rp = np.load(REF / f"ridge_{sp}.npz"); save(f"hybrid_posthoc_{sp}", dict(Y_s=rp["Y_s"], Y_b=c0p["Y_b"], Y_c=c0p["Y_c"], Y_e=c0p["Y_e"]))
        save(f"wm3b_{sp}", m.predict(W))
        if sp == "test_id":
            part = G1.shuffle_partner(W["group"], 0); Ws = dict(W); Ws["A_hist"], Ws["A_fut"] = W["A_hist"][part], W["A_fut"][part]; save("wm3b_shuffle_test_id", m.predict(Ws)); c0s = c0.predict(Ws); save("wm2_c0_shuffle_test_id", c0s)
            rn = m.tensors(W)[2].numpy(); save("wm3b_ablate_rfc_zero_test_id", m.predict(W, rfc_override=np.zeros_like(rn))); rp_ = np.random.default_rng(0).permutation(len(rn)); save("wm3b_ablate_rfc_shuffled_test_id", m.predict(W, rfc_override=rn[rp_]))
    # ---- family windows (frozen file)
    fw = np.load(OUT / "family_windows.npz")
    for f in FAMS:
        Wf = {k.split("/", 1)[1]: fw[k] for k in fw.files if k.startswith(f + "/")}; n = len(Wf["pid"]); Wf["group"] = np.zeros(n, np.int64)
        save(f"wm3b_family_{f}", m.predict(Wf)); save(f"wm2_c0_family_{f}", c0.predict(Wf)); rpf = ridge.predict(Wf); c0f = c0.predict(Wf); save(f"hybrid_posthoc_family_{f}", dict(Y_s=rpf["Y_s"], Y_b=c0f["Y_b"], Y_c=c0f["Y_c"], Y_e=c0f["Y_e"]))
        save(f"ridge_family_{f}", rpf); save(f"mlp_family_{f}", mlp.predict(Wf)); save(f"ridge_blind_family_{f}", ridge_blind.predict(Wf)); save(f"mlp_blind_family_{f}", mlpb.predict(Wf))
        z = np.zeros((n, 20, 8), np.float32); pc, cvp = R.classical_preds(dict(Wf, S_hist=Wf["S_hist"], B_hist=Wf["B_hist"], C_hist=Wf["C_hist"]), prior); save(f"persistence_family_{f}", pc); save(f"constvel_family_{f}", cvp)
    # ---- latency (1 thread, batch 1) of the full WM3-B inference path (ridge matmul + GRU-OC)
    Wb = {k: v[:1] for k, v in tr.items()}; hist, af, rn, _ = m.tensors(Wb); Wt = ridge.Wt; mu, sd = ridge.mu, ridge.sd; xcond = m.nrm.feat(Wb, "cond").astype(np.float64)
    def full(x):
        xx = (xcond - mu) / sd; x1 = np.concatenate([xx, np.ones((1, 1))], 1); rf = (x1 @ Wt[:, :160]).astype(np.float32); rft = torch.tensor(((rf.reshape(1, 20, 8) - nrm.Ys[0]) / nrm.Ys[1]).astype(np.float32)); return m.model(hist, af, rft)
    lat = M.latency(full, lambda: None); lat.update(params=int(m.n_params) + int(Wt[:, :160].size), params_gru=int(m.n_params), params_ridge_joint_block=int(Wt[:, :160].size), note="median wall-clock of one full K=20 prediction: ridge matmul (281x160) + GRU-OC, batch 1, 1 thread")
    json.dump(dict(latency=lat, training_log=logs, train_seconds=train_s, ridge_seconds=ridge_s, best_epoch=m.best_epoch, splits_loaded_during_fit=loaded_fit, total_wall_s=time.time() - T0, n_params_gru=int(m.n_params)), open(OUT / "trainer_report.json", "w"), indent=1, default=str)
    print("done", f"{time.time()-T0:.0f}s", lat)
if __name__ == "__main__": main()
