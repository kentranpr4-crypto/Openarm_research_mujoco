"""MLP behaviour cloning baseline (reaching only).  Demonstrations come from the scripted Jacobian controller on the NEW task (training seeds), matched observation/action spaces and evaluation seeds."""
import json, time
from pathlib import Path
import numpy as np, torch
from .. import utils
from ..checkpoint import save_checkpoint
from ..envs import layout as L
from ..envs.tasks import make_task
from ..models import Actor
from .common import evaluate_policy
from .expert import reach_delta_action
def collect_demos(cfg, task, n_demos, seed_base):
    from ..envs.openarm_env import OpenArmTaskEnv
    if cfg.env.task != "reaching" or cfg.actor.action_param != "delta" or cfg.actor.action_scale != 1.0: raise ValueError("BC baseline: only task=reaching with actor.action_param=delta, action_scale=1 is implemented")
    env = OpenArmTaskEnv(cfg.env.task, cfg.env.task_params, task.max_steps, cfg.env.control_hz); O, Y = [], []
    for k in range(n_demos):
        s, g, tcp = env.reset(seed_base + k); done = False
        while not done:
            a = reach_delta_action(env); O.append(task.actor_obs(torch.as_tensor(s)[None], torch.as_tensor(g)[None], torch.as_tensor(tcp)[None])[0].numpy()); Y.append(a)
            prev = s[L.PREV]; tgt = prev + a * env.adapter.limits.max_speed[:8] / env.hz; o = env.step(tgt); s, tcp = o["state"], o["tcp"]; done = o["terminated"] or o["truncated"]
    return np.stack(O).astype(np.float32), np.stack(Y).astype(np.float32)
def train_bc(cfg, run_dir):
    from ..trainer import resolve_device, build_fk_and_adapter
    device = resolve_device(cfg); utils.seed_everything(cfg.experiment.seed); task = make_task(cfg.env.task, **cfg.env.task_params)
    if cfg.env.max_steps: task.max_steps = cfg.env.max_steps
    _, adapter, _ = build_fk_and_adapter(cfg, device); t0 = time.time(); O, Y = collect_demos(cfg, task, cfg.bc.demos, cfg.train.train_seed_base); n_demo_steps = len(O)
    actor = Actor(task.actor_obs_dim, cfg.bc.hidden, 2).to(device); actor.obs_norm.update(torch.as_tensor(O, device=device)); opt = torch.optim.Adam(actor.parameters(), lr=cfg.bc.lr); X, T = torch.as_tensor(O, device=device), torch.as_tensor(Y, device=device); losses = []
    for ep in range(cfg.bc.epochs):
        perm = torch.randperm(len(X), device=device)
        for ch in perm.split(cfg.bc.batch_size): loss = ((actor(X[ch]) - T[ch]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    seeds = [cfg.train.eval_seed_base + i for i in range(cfg.train.eval_episodes)]; res = evaluate_policy(cfg, task, adapter, actor, seeds, device); res.update(algo="bc", demo_env_transitions=n_demo_steps, train_seconds=time.time() - t0, final_loss=losses[-1])
    save_checkpoint(Path(run_dir) / "checkpoints", "latest.pt", dict(iteration=cfg.bc.epochs, algo="bc", actor=actor.state_dict(), config=cfg.to_dict())); utils.atomic_write_json(Path(run_dir) / "eval" / "eval_final.json", res); utils.atomic_write_json(Path(run_dir) / "COMPLETED", dict(algo="bc"))
    return dict(algo="bc", success_rate=res["success_rate"], mean_return=res["mean_return"], demo_env_transitions=n_demo_steps)
