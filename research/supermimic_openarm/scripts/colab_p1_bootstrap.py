"""Runs ON a remote VM (Colab) via `colab exec -s NAME -f colab_p1_bootstrap.py --timeout 0`.  BENCHMARK ONLY: it never starts training.
Steps: verify + unpack the uploaded bundle (sha256 of every file) -> clone upstream openarm_mujoco at the pinned SHA -> install the few missing packages at the lock versions (the VM's own torch/CUDA build is kept) -> list GPUs -> pick the idle GPU by UUID -> run p1_benchmark with a hard budget and VRAM cap.
Configuration (optional, pass with `colab exec --env KEY=VALUE`): P1_BUDGET_S (300) P1_NUM_ENVS (2) P1_STEPS (400) P1_UPDATES (5) P1_MAX_VRAM_GB (8) P1_MODE (gpu | cpu-selftest | setup = unpack + verify + install only, no benchmark) P1_ROOT (/content/p1) P1_BUNDLE (/content/p1_bundle.tar.gz) P1_SKIP_INSTALL (0) P1_UPSTREAM (path of an existing openarm_mujoco clone to link instead of cloning).
The Colab GPU is NOT the pilot's RTX 5090: this run is a functional / order-of-magnitude check of the benchmark path, never a sizing result for gate G7."""
import hashlib, json, os, re, subprocess, sys, tarfile
from pathlib import Path
E = os.environ.get; ROOT = Path(E("P1_ROOT", "/content/p1")); BUNDLE = Path(E("P1_BUNDLE", "/content/p1_bundle.tar.gz")); MODE = E("P1_MODE", "gpu")
def sh(cmd, check=True, env=None):
    print("$", cmd, flush=True); p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT) if ROOT.exists() else None, env=env)
    out = []
    for line in p.stdout: print(line, end="", flush=True); out.append(line)
    p.wait()
    if check and p.returncode != 0: raise SystemExit(f"command failed ({p.returncode}): {cmd}")
    return "".join(out), p.returncode
def main():
    if not BUNDLE.is_file(): raise SystemExit(f"bundle not found: {BUNDLE} (upload it first: colab upload -s NAME <local bundle> {BUNDLE})")
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(BUNDLE) as t: t.extractall(ROOT, filter="data")
    man = json.loads((ROOT / "BUNDLE_MANIFEST.json").read_text()); bad = [f for f, h in man["files"].items() if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
    if bad: raise SystemExit(f"bundle integrity check FAILED for {len(bad)} files, e.g. {bad[:3]}")
    print(f"bundle ok: {len(man['files'])} files, git {man['git_commit'][:10]} ({man['git_branch']}), {len(man['dirty_paths_in_bundle'])} uncommitted paths included", flush=True)
    up = ROOT / "sim" / "openarm_mujoco"
    if not up.exists():
        if E("P1_UPSTREAM"): up.symlink_to(E("P1_UPSTREAM"))
        else:
            sha = re.search(r'SHA="([0-9a-f]{40})"', (ROOT / "research/supermimic_openarm/scripts/setup_ubuntu.sh").read_text()).group(1)
            sh(f"git clone --quiet https://github.com/enactic/openarm_mujoco.git {up} && git -C {up} checkout --quiet {sha}")
    if E("P1_SKIP_INSTALL", "0") != "1": sh(f"{sys.executable} -m pip install --quiet mujoco==3.17.0 pyyaml==6.0.3 imageio==2.38.0 scipy")
    if MODE == "setup": print("setup finished (no benchmark run)", flush=True); return
    env = dict(os.environ, PYTHONPATH=str(ROOT), MUJOCO_GL=E("MUJOCO_GL", "egl")); tool = f"{sys.executable} -m research.supermimic_openarm.p1_benchmark"
    out_json = ROOT / "outputs/p1_validation/colab_benchmark.json"; out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_json.exists(): out_json.unlink()                                                                                   # a fresh VM never has one; refuse stale results on a kept session
    sh(f"{tool} --list", check=False, env=env)
    common = f"--budget-s {E('P1_BUDGET_S', '300')} --num-envs {E('P1_NUM_ENVS', '2')} --steps {E('P1_STEPS', '400')} --updates {E('P1_UPDATES', '5')} --max-vram-gb {E('P1_MAX_VRAM_GB', '8')} --min-spare-cores 0 --out {out_json}"
    if MODE == "cpu-selftest": cmd = f"{tool} --cpu-selftest {common}"
    else:
        txt, rc = sh(f"{tool} --pick-idle ''", check=False, env=env)
        uuids = [l.split("\t")[0] for l in txt.splitlines() if l.startswith("GPU-")]
        if rc != 0 or len(uuids) != 1: raise SystemExit(f"expected exactly one idle GPU on this VM, found {uuids}; nothing was benchmarked")
        cmd = f"{tool} --gpu-uuid {uuids[0]} {common}"
    sh(cmd, env=env)
    info = ROOT / "outputs/p1_validation/colab_run_info.json"; info.write_text(json.dumps(dict(bundle_sha256=hashlib.sha256(BUNDLE.read_bytes()).hexdigest(), bundle_git_commit=man["git_commit"], bundle_git_branch=man["git_branch"], bundle_created_utc=man["created_utc"], uncommitted_paths_in_bundle=man["dirty_paths_in_bundle"], mode=MODE, note="Colab GPU is not the pilot RTX 5090: functional check only"), indent=1))
    print(f"\nRESULT FILES: {out_json} and {info}\nDownload them with:  colab download -s <NAME> {out_json} ./colab_benchmark.json  &&  colab download -s <NAME> {info} ./colab_run_info.json", flush=True)
main()
