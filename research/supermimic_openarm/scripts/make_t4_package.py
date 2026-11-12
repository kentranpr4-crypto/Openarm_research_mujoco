#!/usr/bin/env python3
"""Builds the package that is copied to the machine running the Colab CLI (the owner's workstation): the code bundle, the registered validation dataset, the registered full hashes, the driver and a README.  Local, read-only for the repository; writes ONE new directory and one tarball.  Uploads nothing.
Registered full SHA-256 values (checked by the driver BEFORE anything is uploaded or run): gates (owner's message), A/B protocol and validation dataset (recorded in docs/ai-native/tasks/2026-11-11-supermimic-p1-benchmark-actor-protocol-report.md).  If a build input does not hash to its registered value this script refuses to build.
Usage (repo root):  python research/supermimic_openarm/scripts/make_t4_package.py [--out-dir outputs/p1_validation/colab/p1_t4_package] [--dataset outputs/p1_validation/wm_eval_dataset_v1.npz] [--expected-json PATH]"""
import argparse, hashlib, json, shutil, subprocess, sys, tarfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
REGISTERED = dict(gates_sha256="a7dc91ca66f6350d4798602aa86258a4c01777d9d123f2e724519499d24a8dca", protocol_sha256="126266d5f4235249a1cfde2070fee2b86d89f1efd09c0d332abc1b335d5611f2", dataset_sha256="92b9e5185a313e1095e49729c5a974bb6cf45fba4e6be975bad11d2561bf23c6")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default="outputs/p1_validation/colab/p1_t4_package"); ap.add_argument("--dataset", default="outputs/p1_validation/wm_eval_dataset_v1.npz"); ap.add_argument("--expected-json", help="override the registered hashes (self-tests only)"); a = ap.parse_args(argv)
    out = REPO / a.out_dir; reg = json.loads(Path(a.expected_json).read_text()) if a.expected_json else REGISTERED
    if out.exists(): sys.exit(f"refusing to overwrite {out}")
    for k, f in (("gates_sha256", "research/supermimic_openarm/gates/p1_pilot_gates.yaml"), ("protocol_sha256", "research/supermimic_openarm/experiments/p1_actor_ab_protocol.yaml"), ("dataset_sha256", a.dataset)):
        if sha(REPO / f) != reg[k]: sys.exit(f"REFUSING to build: {f} hashes to {sha(REPO / f)}, registered {reg[k]}")
    out.mkdir(parents=True); subprocess.run([sys.executable, str(REPO / "research/supermimic_openarm/scripts/make_colab_bundle.py"), "--out", str((out / "p1_bundle.tar.gz").relative_to(REPO))], check=True)
    shutil.copy(REPO / a.dataset, out / "wm_eval_dataset_v1.npz"); (out / "scripts").mkdir(); shutil.copy(REPO / "research/supermimic_openarm/scripts/colab_t4_run.py", out / "scripts/colab_t4_run.py"); (out / "EXPECTED_HASHES.json").write_text(json.dumps(reg, indent=1))
    (out / "README.txt").write_text("""P1 on a Colab T4 -- package (built from the working tree; includes UNCOMMITTED P1 files, see bundle manifest)
1. Copy this directory to the machine where `colab` is installed and authenticated (python >= 3.12 only needed for the CLI itself; the driver uses the standard library).
2. Local checks only (touches nothing remote):      python3 scripts/colab_t4_run.py --verify-only
3. Benchmark only:                                   python3 scripts/colab_t4_run.py --benchmark-only
4. Benchmark + B0 pilot if the conditions pass:      python3 scripts/colab_t4_run.py
Results land in results_<timestamp>/ (run_summary.json, decision.json, smoke.json, benchmark.json, sync_*.tgz, SHA256SUMS).  Nothing is committed, pushed or posted.
""")
    tgz = out.parent / (out.name + ".tar.gz")
    with tarfile.open(tgz, "w:gz") as t: t.add(out, arcname=out.name)
    print(f"package {out} -> {tgz} ({tgz.stat().st_size / 1e6:.1f} MB); sha256 {sha(tgz)}"); return 0
if __name__ == "__main__": sys.exit(main())
