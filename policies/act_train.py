#!/usr/bin/env python3
"""Train a LeRobot ACT policy on the converted OpenArm dataset (run inside the ACT venv: /home/user/.venvs/act).

Inputs  : observation.state (16) + camera images (default: front, wrist_left)      [NO object coordinates, NO velocities]
Outputs : action chunk (chunk_size x 16) = absolute joint-position targets in simulator ctrl order (see control/action_adapter.py)
Design  : use_vae=False (the scripted expert is deterministic given the observation, so the CVAE latent adds nothing and
          costs CPU); ResNet18 backbone trained from scratch (ImageNet weights cannot be downloaded in this sandbox);
          normalisation via LeRobot pre/post processors with dataset statistics of the TRAIN episodes only.
Split   : by EPISODE, from <dataset>/openarm_splits.json. --overfit N trains and validates on the first N train episodes.
Usage   : python policies/act_train.py --dataset outputs/lerobot/x --out outputs/checkpoints/x --steps 2000 [--overfit 3]"""
import sys, json, time, argparse, subprocess, random
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent

def build(meta, cams, a):
    from lerobot.configs.types import FeatureType
    from lerobot.utils.feature_utils import dataset_to_policy_features
    from lerobot.policies.act.configuration_act import ACTConfig
    feats = dataset_to_policy_features(meta.features)
    out = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    inp = {k: v for k, v in feats.items() if k == "observation.state" or k in [f"observation.images.{c}" for c in cams]}
    return ACTConfig(input_features=inp, output_features=out, chunk_size=a.chunk, n_action_steps=a.chunk, use_vae=False, dim_model=a.dim_model,
                     n_heads=a.heads, dim_feedforward=a.dim_ff, n_encoder_layers=a.enc_layers, pretrained_backbone_weights=None, push_to_hub=False,
                     device="cpu", optimizer_lr=a.lr, optimizer_lr_backbone=a.lr, dropout=a.dropout)

@torch.no_grad()
def evaluate(policy, pre, loader, max_batches=8):
    policy.eval(); tot, n = 0.0, 0
    for i, b in enumerate(loader):
        if i >= max_batches: break
        l, _ = policy.forward(pre(b)); tot += l.item(); n += 1
    policy.train(); return tot / max(n, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--chunk", type=int, default=20); ap.add_argument("--cams", nargs="+", default=["front", "wrist_left"])
    ap.add_argument("--dim-model", type=int, default=256); ap.add_argument("--heads", type=int, default=4); ap.add_argument("--dim-ff", type=int, default=1024)
    ap.add_argument("--enc-layers", type=int, default=2); ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--overfit", type=int, default=0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val-every", type=int, default=100); ap.add_argument("--save-every", type=int, default=500); ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()): sys.exit(f"{a.out} is not empty; choose a new --out (existing checkpoints are never overwritten)")
    a.out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if a.threads: torch.set_num_threads(a.threads)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    meta = LeRobotDatasetMetadata("local/openarm_lab_pickplace", root=a.dataset); fps = meta.fps
    split = json.loads((a.dataset / "openarm_splits.json").read_text()); tr, va = split["train"], split["val"]
    if a.overfit: tr = tr[: a.overfit]; va = tr
    dts = {"action": [i / fps for i in range(a.chunk)]}
    ds_tr = LeRobotDataset("local/openarm_lab_pickplace", root=a.dataset, episodes=tr, delta_timestamps=dts)
    ds_va = LeRobotDataset("local/openarm_lab_pickplace", root=a.dataset, episodes=va, delta_timestamps=dts) if va else None
    # statistics of the TRAIN episodes only (no leakage from validation episodes)
    cfg = build(meta, a.cams, a); policy = ACTPolicy(cfg); policy.train()
    pre, post = make_pre_post_processors(cfg, dataset_stats=ds_tr.meta.stats)
    mk = lambda ds, sh: torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=sh, num_workers=a.workers, drop_last=sh, persistent_workers=a.workers > 0)
    dl, dlv = mk(ds_tr, True), (mk(ds_va, False) if ds_va else None)
    opt = torch.optim.AdamW(policy.get_optim_params(), lr=a.lr, weight_decay=1e-4)
    nparam = sum(p.numel() for p in policy.parameters())
    info = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, n_params=nparam, train_episodes=tr, val_episodes=va, fps=fps,
                n_train_frames=len(ds_tr), torch=torch.__version__, commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    (a.out / "train_info.json").write_text(json.dumps(info, indent=1)); print(f"params {nparam/1e6:.1f}M | train frames {len(ds_tr)} | episodes train {len(tr)} val {len(va)}", flush=True)
    log = open(a.out / "train_log.jsonl", "w"); t0 = time.time(); step = 0; it = iter(dl); ema = None
    def save(tag):
        d = a.out / tag; policy.save_pretrained(d); pre.save_pretrained(d); post.save_pretrained(d)
    while step < a.steps:
        try: b = next(it)
        except StopIteration: it = iter(dl); b = next(it)
        loss, ld = policy.forward(pre(b)); opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0); opt.step(); step += 1
        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        rec = dict(step=step, loss=loss.item(), loss_ema=ema, grad_norm=float(gn), elapsed_s=time.time() - t0)
        if dlv is not None and step % a.val_every == 0: rec["val_l1"] = evaluate(policy, pre, dlv)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if step % 20 == 0 or "val_l1" in rec: print({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}, flush=True)
        if step % a.save_every == 0: save(f"step_{step:06d}")
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes: print("time budget reached", flush=True); break
    save("last"); print("saved", a.out / "last", f"after {step} steps, {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
