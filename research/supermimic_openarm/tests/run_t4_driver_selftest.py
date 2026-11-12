#!/usr/bin/env python3
"""Manual end-to-end SELF-TEST of the T4 driver control flow against the fake CLI (CPU, no Colab, no GPU; minutes).  Not part of the quick pytest suite (set P1_RUN_SLOW=1 and run tests/test_t4_driver_selftest.py, or run this file).
Covers: hash verification, bundle content policy, upload + remote hash check, setup, smoke, benchmark, decision NO-GO on CPU, forced pilot branch with a stub trainer, gate evaluation at iteration 400 on a real checkpoint, STOP/INCOMPLETE handling, periodic + final sync, artifact verification, release.
Run from the repo root:  python research/supermimic_openarm/tests/run_t4_driver_selftest.py"""
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]; PY = sys.executable; T = REPO / "research/supermimic_openarm/tests"
def main():
    try: return _main()
    finally:
        for x in ("/content/p1", "/content/p1_inputs"): shutil.rmtree(x, ignore_errors=True)                                          # only the fake-VM paths this test created
        for x in ("/content/p1_bundle.tar.gz", "/content/p1_sync.tgz"): Path(x).unlink(missing_ok=True)
def _main():
    work = REPO / "outputs/p1_validation/colab/selftest_work"
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True); z = np.load(REPO / "outputs/p1_validation/wm_eval_dataset_v1.npz", allow_pickle=True); mini = work / "mini_dataset.npz"; np.savez_compressed(mini, train=np.array(list(z["train"])[:6], dtype=object), val=np.array(list(z["val"])[:12], dtype=object), allow_pickle=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_t4_package", REPO / "research/supermimic_openarm/scripts/make_t4_package.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); reg = dict(m.REGISTERED); reg["dataset_sha256"] = hashlib.sha256(mini.read_bytes()).hexdigest(); (work / "expected.json").write_text(json.dumps(reg))
    pkg_rel = "outputs/p1_validation/colab/selftest_work/pkg"; subprocess.run([PY, "research/supermimic_openarm/scripts/make_t4_package.py", "--out-dir", pkg_rel, "--dataset", str(mini.relative_to(REPO)), "--expected-json", str(work / "expected.json")], cwd=REPO, check=True)
    pkg = REPO / pkg_rel; env = dict(os.environ, PATH=str(Path(PY).parent) + os.pathsep + os.environ["PATH"], PYTHONPATH="/content/p1", FAKE_COLAB_PYTHON=PY, FAKE_COLAB_STOPPED=str(work / "stopped.flag"), P1_SKIP_INSTALL="1", P1_UPSTREAM=str(REPO / "sim/openarm_mujoco"), P1_TEST_OVERRIDES="world_model.hidden=128,world_model.layers=2,world_model.batch_size=64,imagination.batch_size=128,imagination.chunk_size=64,actor.hidden=128,actor.layers=2,critic.hidden=64,critic.layers=2,train.warmup_steps=16", P1_PILOT_POLL_S="2")
    run_dir = "/content/p1/outputs/supermimic_openarm/p1_pilot_b0_t4/seed1000"; src = "/content/p1/outputs/supermimic_openarm/p1_t4_smoke/seed1000/checkpoints/latest.pt"
    env["P1_PILOT_TRAIN_CMD"] = f"{PY} /content/p1/research/supermimic_openarm/tests/fake_train.py --run-dir {run_dir} --src-ckpt {src}"
    if any(Path(x).exists() for x in ("/content/p1", "/content/p1_inputs", "/content/p1_bundle.tar.gz", "/content/p1_sync.tgz")): sys.exit("refusing to run: /content/p1* already exists (this self-test uses those exact paths as the fake VM)")
    r = subprocess.run([PY, str(pkg / "scripts/colab_t4_run.py"), "--package-dir", str(pkg), "--results-dir", str(work / "results"), "--colab", str(T / "fake_colab.py"), "--test-env", "--test-force-go", "--budget-s", "600", "--poll-s", "6", "--sync-every-s", "20"], env=env, cwd=work, text=True, capture_output=True)
    (work / "driver_stdout.txt").write_text(r.stdout + r.stderr); print(r.stdout[-6000:]); print(r.stderr[-1500:])
    s = json.loads((work / "results/run_summary.json").read_text()); print("EXIT", r.returncode, "| stage", s["stage"], "| pilot_started", s.get("pilot_started"), "| gates", [(g["iteration"], g["exit_code"]) for g in (s.get("pilot_status") or {}).get("gates_done", [])], "| reason", (s.get("pilot_status") or {}).get("reason"), "| artifacts", s.get("artifact_verification"), "| released", s.get("session_released")); return r.returncode
if __name__ == "__main__": sys.exit(main())
