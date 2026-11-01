#!/usr/bin/env python3
"""INDEPENDENT held-out collection (simulation only): runs the FROZEN manifest (outputs/indep/manifest.json: 50 new initial-state groups x 25 development variants = 1 250 episodes) through the same scripted-expert /
intervention / clip+slew pipeline as the development collection (data/wm1_intervene.py::run_one).  Refuses to start unless outputs/indep/protocol_freeze.json exists and its hashes match.  Existing raw episodes are never overwritten.
Run (ACT venv, detached): python data/indep_collect.py [--workers 4]"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, hashlib, argparse, time
from pathlib import Path
from multiprocessing import Pool
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "data")):
    if p not in sys.path: sys.path.insert(0, p)
import wm1_intervene as WI
OUT = ROOT / "outputs/indep"; RAW = OUT / "raw"; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    fz = OUT / "protocol_freeze.json"
    if not fz.exists(): sys.exit("protocol not frozen: run eval/indep_freeze.py first")
    F = json.loads(fz.read_text()); bad = [k for k, v in F["files_sha256"].items() if sha(ROOT / k) != v]
    if bad: sys.exit(f"FROZEN FILES CHANGED: {bad}")
    man = json.loads((OUT / "manifest.json").read_text()); assert man["eval_seed_label"] == 7001 and not set(man["forbidden_seeds"]) & {7001} and len(man["poses"]) * len(man["variants"]) == 1250 <= man["cap_episodes"]
    WI.OUT = RAW; RAW.mkdir(parents=True, exist_ok=True); jobs = [dict(pose=p, variant=v) for p in man["poses"] for v in man["variants"]]; t0 = time.time()
    with Pool(a.workers) as pool: R = pool.map(WI.run_one, jobs, chunksize=4)
    ip = RAW / "index.json"; old = json.loads(ip.read_text())["episodes"] if ip.exists() else []; have = {r["name"]: r for r in old}; [have.__setitem__(r["name"], r) for r in R if r["status"] == "new"]
    ip.write_text(json.dumps(dict(manifest_sha256=sha(OUT / "manifest.json"), protocol_freeze_sha256=sha(fz), episodes=list(have.values())), indent=1, default=str)); print(f"{len(jobs)} jobs in {time.time()-t0:.0f}s; new {sum(r['status']=='new' for r in R)}; total indexed {len(have)}")
if __name__ == "__main__": main()
