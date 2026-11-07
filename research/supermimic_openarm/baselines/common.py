"""Shared evaluation protocol for all methods: same env, goal seeds, success criteria, observation function and action adapter."""
import time
import numpy as np, torch
from .. import utils
from ..envs import layout as L
def evaluate_policy(cfg, task, adapter, act_fn, seeds, device, save_traj=0):
    """act_fn(obs_tensor (1,obs_dim)) -> action in [-1,1]^8 tensor.  Runs complete episodes on held-out seeds; returns metrics (+ per-episode rows)."""
    from ..envs.openarm_env import OpenArmTaskEnv
    utils.assert_seeds_allowed(*seeds); env = OpenArmTaskEnv(cfg.env.task, cfg.env.task_params, task.max_steps, cfg.env.control_hz); rows = []; lat = []
    for sd in seeds:
        s, g, tcp = env.reset(sd); done = False; ret = 0.0
        while not done:
            obs = task.actor_obs(torch.as_tensor(s, device=device)[None], torch.as_tensor(g, device=device)[None], torch.as_tensor(tcp, device=device)[None]); t = time.perf_counter()
            with torch.no_grad(): a = act_fn(obs).float()
            lat.append(time.perf_counter() - t); tgt = adapter.target_from_action(a, torch.as_tensor(s[L.PREV], device=device)[None])[0].cpu().numpy().astype(np.float64)
            o = env.step(tgt); s, tcp = o["state"], o["tcp"]; ret += o["reward"]; done = o["terminated"] or o["truncated"]
        err = np.linalg.norm(tcp - g) if cfg.env.task == "reaching" else np.linalg.norm(s[L.BOX_POS][:2] - g[:2])
        rows.append(dict(seed=int(sd), success=bool(env.ever_success), ret=float(ret), length=int(env.t), final_goal_error_m=float(err), box_max_rise_m=float(env.box_max_rise)))
    return dict(success_rate=float(np.mean([r["success"] for r in rows])), mean_return=float(np.mean([r["ret"] for r in rows])), mean_final_goal_error_m=float(np.mean([r["final_goal_error_m"] for r in rows])), n_episodes=len(rows),
                inference_latency_ms=1e3 * float(np.mean(lat)), seeds=[int(x) for x in seeds], episodes=rows, privileged_state=True)
