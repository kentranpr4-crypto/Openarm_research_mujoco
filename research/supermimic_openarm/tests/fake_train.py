#!/usr/bin/env python3
"""TEST DOUBLE of the training process for the supervisor self-test (writes metrics + checkpoint manifest like the real trainer; no training happens)."""
import argparse, hashlib, json, time
from pathlib import Path
import torch
ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); ap.add_argument("--src-ckpt", required=True); ap.add_argument("--iterations", default="200,400,600"); a = ap.parse_args(); rd = Path(a.run_dir); (rd / "checkpoints").mkdir(parents=True); ck = torch.load(a.src_ckpt, map_location="cpu", weights_only=False); man = {}
for it in [int(x) for x in a.iterations.split(",")]:
    time.sleep(3); ck["iteration"] = it; p = rd / "checkpoints" / "latest.pt"; torch.save(ck, p); man["latest.pt"] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), iteration=it); (rd / "checkpoints" / "manifest.json").write_text(json.dumps(man))
    with open(rd / "metrics.jsonl", "a") as f: f.write(json.dumps(dict(iteration=it, env_steps=it * 96, t_collect=0.1, t_wm=0.1, t_actor=0.1, bad_grads=0)) + "\n")
time.sleep(600)
