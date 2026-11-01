#!/usr/bin/env python3
"""INDEPENDENT held-out evaluation: FREEZE.  Records sha256 of every model/config/code/manifest/protocol file and of the deterministic closed-form baselines BEFORE any independent data exist.  Refuses to overwrite.
Run (ACT venv): python eval/indep_freeze.py"""
import sys, json, hashlib, subprocess, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); sys.argv = [sys.argv[0], "none"]
import wm2_run as R, wm2_models as M
OUT = ROOT / "outputs/indep"; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
FILES = ["outputs/indep/manifest.json", "docs/ai-native/tasks/2026-11-01-indep-eval-protocol.md", "outputs/wm3/gates_v2_config.json", "outputs/wm3/model_config_wm3b.json", "outputs/wm2/model_config.json", "outputs/wm3/freeze.json", "outputs/wm3/contact_scales.json", "outputs/wm2/dataset/index.json",
         "outputs/wm2/windows/train.npz", "outputs/wm2/windows/val.npz", "outputs/wm2_1/models/gru_c0.pt", "outputs/wm2_1/models/mlp.pt", "outputs/wm2_1/models/mlp_blind.pt", "outputs/wm3/models/wm3b.pt",
         "eval/wm3_gates.py", "eval/wm2_gates.py", "eval/wm3b_model.py", "eval/wm2_models.py", "eval/wm2_common.py", "eval/wm2_run.py", "eval/wm3_data.py", "eval/indep_predict.py", "eval/indep_evaluate.py", "eval/indep_freeze.py",
         "data/indep_manifest.py", "data/indep_collect.py", "data/indep_build.py", "data/wm1_intervene.py", "data/wm1_sim.py", "data/wm1_windows.py", "control/safety_limits.py", "policies/baseline.py"]
def hr(r): return hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in (r.Wt, r.mu, r.sd))).hexdigest()
def main():
    if (OUT / "protocol_freeze.json").exists(): sys.exit("already frozen")
    torch.set_num_threads(3); tr = R.load("train"); nrm = M.Norm(tr); r = M.Ridge("cond").fit(nrm, tr); rb = M.Ridge("blind").fit(nrm, tr); oo = M.OutcomeOnly().fit(nrm, tr)
    cf = dict(ridge=hr(r), ridge_blind=hr(rb), outcome_only=hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in (oo.mu, oo.sd))).hexdigest())
    ref = ROOT / "outputs/wm2_1/preds"; te = R.load("test_id"); chk = float(np.abs(r.predict(te)["Y_b"] - np.load(ref / "ridge_test_id.npz")["Y_b"]).max()); chk2 = float(np.abs(rb.predict(te)["Y_s"] - np.load(ref / "ridge_blind_test_id.npz")["Y_s"]).max())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(); dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    F = dict(frozen_at=__import__("datetime").datetime.utcnow().isoformat() + "Z", code_commit=commit, uncommitted_at_freeze=dirty, status="INDEPENDENT held-out protocol frozen BEFORE collection; no independent data exist at this time", files_sha256={f: sha(ROOT / f) for f in FILES},
             model_checkpoint_sha256={"wm2_c0": sha(ROOT / "outputs/wm2_1/models/gru_c0.pt"), "mlp": sha(ROOT / "outputs/wm2_1/models/mlp.pt"), "mlp_blind": sha(ROOT / "outputs/wm2_1/models/mlp_blind.pt"), "wm3b": sha(ROOT / "outputs/wm3/models/wm3b.pt")},
             closed_form_baselines_sha256=cf, closed_form_reproduction_max_abs_diff_vs_stored_dev_preds=dict(ridge_Yb=chk, ridge_blind_Ys=chk2), independent_data_exist_at_freeze=(OUT / "raw").exists() or (OUT / "dataset").exists())
    assert not F["independent_data_exist_at_freeze"]; (OUT / "protocol_freeze.json").write_text(json.dumps(F, indent=1)); print(json.dumps({k: v for k, v in F.items() if k != "files_sha256"}, indent=1)); print("protocol_freeze sha256", sha(OUT / "protocol_freeze.json"))
if __name__ == "__main__": main()
