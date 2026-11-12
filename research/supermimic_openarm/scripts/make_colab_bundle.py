#!/usr/bin/env python3
"""Builds ONE tarball of the code needed to run the P1 benchmark on a remote VM (e.g. a Colab GPU runtime).  Local, read-only for the repository; writes ONE new file (refuses to overwrite).
Contents: sim/ (without the 140 MB upstream clone `sim/openarm_mujoco`, which the VM clones at the pinned SHA), control/, research/__init__.py, research/supermimic_openarm/ (working tree, including uncommitted P1 changes), plus BUNDLE_MANIFEST.json (git commit, dirty flag, sha256 of every file).
Never included: outputs/, checkpoints, datasets, logs, .git, virtualenvs (.venv*, venv), node_modules, research/openarm_wam/, any __pycache__.  Nothing is uploaded by this script.
Usage (repo root):  python research/supermimic_openarm/scripts/make_colab_bundle.py [--out outputs/p1_validation/colab/p1_bundle.tar.gz]"""
import argparse, hashlib, io, json, subprocess, sys, tarfile, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]; INCLUDE = ["sim", "control", "research/__init__.py", "research/supermimic_openarm"]; SKIP_PARTS = {"__pycache__", "openarm_mujoco", ".git", "outputs", "openarm_wam"}; SKIP_SUFFIX = (".pyc", ".pt", ".npz", ".tar.gz", ".log")
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs/p1_validation/colab/p1_bundle.tar.gz"); a = ap.parse_args(argv); out = REPO / a.out
    if out.exists(): sys.exit(f"refusing to overwrite {out}")
    files = []
    for inc in INCLUDE:
        p = REPO / inc
        for f in ([p] if p.is_file() else sorted(p.rglob("*"))):
            rel = f.relative_to(REPO)
            if f.is_file() and not (set(rel.parts) & SKIP_PARTS) and not any(x.startswith(".venv") or x == "venv" or x == "node_modules" for x in rel.parts) and not str(rel).endswith(SKIP_SUFFIX): files.append(rel)
    g = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()
    man = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), git_commit=g("rev-parse", "HEAD"), git_branch=g("rev-parse", "--abbrev-ref", "HEAD"), dirty_paths_in_bundle=sorted(l[3:] for l in g("status", "--porcelain").splitlines() if any(l[3:].startswith(i) for i in INCLUDE)),
               upstream_note="sim/openarm_mujoco is NOT in the bundle; the VM clones it at the pin in scripts/setup_ubuntu.sh", files={str(r): hashlib.sha256((REPO / r).read_bytes()).hexdigest() for r in files})
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as t:
        for r in files: t.add(REPO / r, arcname=str(r), recursive=False)
        b = json.dumps(man, indent=1).encode(); ti = tarfile.TarInfo("BUNDLE_MANIFEST.json"); ti.size = len(b); ti.mtime = int(time.time()); t.addfile(ti, io.BytesIO(b))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(files)} files); git {man['git_commit'][:10]} on {man['git_branch']}; {len(man['dirty_paths_in_bundle'])} uncommitted paths inside the bundle; sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}"); return 0
if __name__ == "__main__": sys.exit(main())
