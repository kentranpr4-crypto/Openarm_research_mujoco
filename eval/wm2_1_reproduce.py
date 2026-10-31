#!/usr/bin/env python3
"""WM-2.1 (diagnostics only): REPRODUCE the frozen WM2-C0 models (same frozen configs, same seed 0, same data) and save their weights and per-window predictions, which WM-2 did not persist.
Nothing is tuned; frozen hashes are checked first; nothing in outputs/wm2 is modified (writes to outputs/wm2_1).  Reproduction is verified against the WM-2 results (outputs/wm2/wm2_results.json).
Run (ACT venv): python eval/wm2_1_reproduce.py"""
import sys, json, time, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
sys.argv = [sys.argv[0], "none"]
import wm2_run as R, wm2_models as M, wm2_gates as G
OUT = ROOT / "outputs/wm2_1"; PRED = OUT / "preds"
def main():
    R.check_freeze(); t0 = time.time(); R.LOADED.clear(); tr, va = R.load("train"), R.load("val"); nrm = M.Norm(tr); logs = []; times = {}; models = {}
    def fit(name, fn): t = time.time(); models[name] = fn(); times[name] = time.time() - t; print(f"[{name}] {times[name]:.0f}s", flush=True)
    fit("ridge", lambda: M.Ridge("cond").fit(nrm, tr)); fit("ridge_blind", lambda: M.Ridge("blind").fit(nrm, tr)); fit("outcome_only", lambda: M.OutcomeOnly().fit(nrm, tr))
    fit("mlp", lambda: M.NN("mlp_cond").fit(nrm, tr, va, logs, seed=0)); fit("mlp_blind", lambda: M.NN("mlp_blind").fit(nrm, tr, va, logs, seed=0)); fit("gru", lambda: M.NN("gru").fit(nrm, tr, va, logs, seed=0))
    torch.save({k: v for k, v in models["gru"].model.state_dict().items()}, OUT / "models/gru_c0.pt"); torch.save(models["mlp"].model.state_dict(), OUT / "models/mlp.pt"); torch.save(models["mlp_blind"].model.state_dict(), OUT / "models/mlp_blind.pt")
    for nm in ("ridge", "ridge_blind"): np.savez(OUT / f"models/{nm}.npz", Wt=models[nm].Wt, lw=models[nm].lw, lb=models[nm].lb, mu=models[nm].mu, sd=models[nm].sd)
    prior = tr["Y_e"].mean(0); splits = ["val"] + R.EVAL
    for sp in splits:
        W = va if sp == "val" else R.load(sp)
        for name, m in models.items():
            p = m.predict(W); np.savez_compressed(PRED / f"{name}_{sp}.npz", **p)
        pc = R.classical_preds(W, prior); np.savez_compressed(PRED / f"persistence_{sp}.npz", **pc[0]); np.savez_compressed(PRED / f"constvel_{sp}.npz", **pc[1])
    pz = np.load(R.WIN / "pairs.npz"); pairs = {k: pz[k] for k in pz.files}; np.savez_compressed(PRED / "gru_pairs.npz", **models["gru"].predict(pairs)); np.savez_compressed(PRED / "ridge_pairs.npz", **models["ridge"].predict(pairs))
    # verification against WM-2
    ref = json.loads((ROOT / "outputs/wm2/wm2_results.json").read_text()); chk = {}
    for sp in R.EVAL:
        W = R.load(sp); P = {k: np.load(PRED / f"gru_{sp}.npz")[k] for k in ("Y_s", "Y_b", "Y_c", "Y_e")}; act = G.active_mask(W); e = G.errors(G.sub(P, act), G.sub(R.truth(W), act), 20)
        for met, v in e.items(): chk[f"{sp}/{met}"] = [float(G.macro(v, W["group"][act])), ref["errors"]["gru"][sp]["active"]["K=20"][met]]
    maxd = max(abs(a - b) / max(abs(b), 1e-9) for a, b in chk.values())
    json.dump(dict(fit_seconds=times, total_s=time.time() - t0, splits_loaded_during_fit=sorted(set(R.LOADED[:2])), training_log=logs, reproduction_check=chk, max_relative_difference=maxd, best_epochs={n: getattr(m, "best_epoch", None) for n, m in models.items() if hasattr(m, "best_epoch")}), open(OUT / "reproduction.json", "w"), indent=1)
    print("max relative difference vs WM-2 GRU errors:", maxd, f"total {time.time()-t0:.0f}s")
if __name__ == "__main__": main()
