#!/usr/bin/env python3
"""x-sweep (evaluation only): runs ONE frozen checkpoint on the explicit pose table of outputs/xsweep/xsweep_config.json.
Everything except the box-pose source is the unchanged eval/evaluate.py path (run_agent_episode, labels, summaries, rollout files).
The pose source is replaced by monkey-patching policies.baseline.sample_box_pose inside each worker; nothing on disk is modified.
Run (ACT venv): python eval/xsweep_run.py --ckpt outputs/checkpoints/e0_c0_seed1/last --out outputs/xsweep/eval/c0_seed1 --workers 4"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, argparse, hashlib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path: sys.path.insert(0, p)
CFG = ROOT / "outputs/xsweep/xsweep_config.json"; EXPECT = "6c59292d6d951673ab813fc50eba38612c01777534ccdad2a6f3c5797b1f596c"

def worker(job):
    ids, a = job; os.environ.setdefault("LP_NUM_THREADS", "1")
    import policies.baseline as PB
    from lab_env import LabEnv
    from control.action_adapter import ActionAdapter
    from policies.act_infer import ACTRunner
    from eval.evaluate import run_agent_episode
    poses = {p["idx"]: p for p in json.loads(CFG.read_text())["poses"]}; cur = {}
    PB.sample_box_pose = lambda rng, region="train": (cur["p"]["x"], cur["p"]["y"], cur["p"]["yaw"])
    probe = LabEnv(cameras=()); ad = ActionAdapter(probe.m, 20); probe.close()
    runner = ACTRunner(a["ckpt"], ad, n_action_steps=None, threads=1, temporal_ensemble=None); cams = tuple(runner.cams)
    env = LabEnv(image_hw=(120, 160), control_hz=20, render_profile="fast", cameras=cams); env.runner = runner; env.state_perturb = None
    expert = PB.ScriptedExpert(env.m, control_hz=20); out = Path(a["out"]) / "act" / "sweep"; res = []
    for idx in ids:
        cur["p"] = poses[idx]
        r = run_agent_episode("act", a["ckpt"], None, env, expert, idx, 21, "sweep", {}, 200, out / "videos", out / "rollouts", False)
        r.update(sweep_x=poses[idx]["x"], sweep_y=poses[idx]["y"], sweep_yaw=poses[idx]["yaw"], sweep_block=poses[idx]["block"])
        print(f"[act/sweep] idx {idx} x={poses[idx]['x']:.2f} block {poses[idx]['block']}: {r['outcome']}", flush=True); res.append(r)
    env.close(); return res

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--ids", type=int, nargs="*", help="smoke test only: subset of pose indices"); a = ap.parse_args()
    if hashlib.sha256(CFG.read_bytes()).hexdigest() != EXPECT: sys.exit("x-sweep config changed after freeze")
    cfg = json.loads(CFG.read_text()); ids = list(a.ids) if a.ids else [p["idx"] for p in cfg["poses"]]; chunks = [ids[i::a.workers] for i in range(a.workers)]
    args = dict(ckpt=a.ckpt, out=str(a.out))
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool: recs = sum(pool.map(worker, [(c, args) for c in chunks if c]), [])
    recs.sort(key=lambda r: r["episode_index"]); d = a.out / "act" / "sweep"; d.mkdir(parents=True, exist_ok=True)
    (d / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in recs)); print(f"{len(recs)}/{len(ids)} episodes; task success {sum(r['task_success'] for r in recs)}/{len(recs)}", flush=True)

if __name__ == "__main__": main()
