"""Model-free PPO baseline on the same environment, observations (task.actor_obs), action adapter, goals and evaluation protocol."""
import json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from .. import utils
from ..checkpoint import save_checkpoint
from ..envs import layout as L
from ..envs.tasks import make_task
from ..envs.vector import make_vec_env
from ..models import RunningNorm, mlp
from .common import evaluate_policy
class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden, init_log_std):
        super().__init__(); self.norm = RunningNorm(obs_dim); self.pi = mlp(obs_dim, hidden, 2, L.ACT_DIM, out_gain=0.01); self.v = mlp(obs_dim, hidden, 2, 1); self.log_std = nn.Parameter(torch.full((L.ACT_DIM,), float(init_log_std)))
    def dist(self, obs): return torch.distributions.Normal(torch.tanh(self.pi(self.norm(obs))), self.log_std.exp())
    def value(self, obs): return self.v(self.norm(obs)).squeeze(-1)
    def forward(self, obs): return torch.tanh(self.pi(self.norm(obs)))              # deterministic action for evaluation
def train_ppo(cfg, run_dir, max_iterations=None):
    from ..trainer import resolve_device, build_fk_and_adapter
    device = resolve_device(cfg); utils.seed_everything(cfg.experiment.seed); task = make_task(cfg.env.task, **cfg.env.task_params)
    if cfg.env.max_steps: task.max_steps = cfg.env.max_steps
    _, adapter, _ = build_fk_and_adapter(cfg, device); p = cfg.ppo; n = cfg.env.num_envs; vec = make_vec_env(cfg.env.backend, n, cfg.env.task, cfg.env.task_params, task.max_steps, cfg.env.control_hz, cfg.train.train_seed_base)
    ac = PPOActorCritic(task.actor_obs_dim, p.hidden, p.init_log_std).to(device); opt = torch.optim.Adam(ac.parameters(), lr=p.lr); state, goal, tcp = vec.reset(); t0 = time.time(); env_steps = 0; iters = cfg.train.iterations if max_iterations is None else min(cfg.train.iterations, max_iterations); log = []
    T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    for it in range(1, iters + 1):
        O, A, LP, R, V, DN, TR = [], [], [], [], [], [], []
        for _ in range(p.rollout_steps):
            obs = task.actor_obs(T(state), T(goal), T(tcp)); ac.norm.update(obs)
            with torch.no_grad(): d = ac.dist(obs); a = d.sample(); lp = d.log_prob(a).sum(-1); v = ac.value(obs)
            tgt = adapter.target_from_action(a.clamp(-1, 1), T(state[:, L.PREV])).cpu().numpy().astype(np.float64); o = vec.step(tgt); env_steps += n
            with torch.no_grad(): v_next = ac.value(task.actor_obs(T(o["next_state"]), T(o["goal_used"]), T(_tcp_of(o))))
            r = T(o["reward"]) + p.gamma * v_next * T(o["truncated"].astype(np.float32))            # time-limit bootstrap; terminal -> no bootstrap
            O.append(obs); A.append(a); LP.append(lp); R.append(r); V.append(v); DN.append(T((o["terminated"] | o["truncated"]).astype(np.float32))); state, goal, tcp = o["state"].astype(np.float32), o["goal"].astype(np.float32), o["tcp"]
        with torch.no_grad(): last_v = ac.value(task.actor_obs(T(state), T(goal), T(tcp)))
        adv = torch.zeros(p.rollout_steps, n, device=device); g = torch.zeros(n, device=device)
        for t in reversed(range(p.rollout_steps)):
            nv = last_v if t == p.rollout_steps - 1 else V[t + 1]; delta = R[t] + p.gamma * nv * (1 - DN[t]) - V[t]; g = delta + p.gamma * p.lam * (1 - DN[t]) * g; adv[t] = g
        Vt = torch.stack(V); ret = adv + Vt; Ob, Ab, LPb = torch.cat(O).reshape(-1, O[0].shape[-1]), torch.stack(A).reshape(-1, L.ACT_DIM), torch.stack(LP).reshape(-1); advf = (adv.reshape(-1) - adv.mean()) / (adv.std() + 1e-8); retf = ret.reshape(-1)
        for _ in range(p.epochs):
            perm = torch.randperm(len(Ob), device=device)
            for ch in perm.chunk(p.minibatches):
                d = ac.dist(Ob[ch]); lp = d.log_prob(Ab[ch]).sum(-1); ratio = (lp - LPb[ch]).exp(); pl = -torch.min(ratio * advf[ch], ratio.clamp(1 - p.clip, 1 + p.clip) * advf[ch]).mean(); vl = ((ac.value(Ob[ch]) - retf[ch]) ** 2).mean()
                loss = pl + p.vf_coef * vl - p.ent_coef * d.entropy().sum(-1).mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ac.parameters(), 1.0); opt.step()
        log.append(dict(iteration=it, env_steps=env_steps, policy_loss=float(pl.detach()), value_loss=float(vl.detach()), mean_reward=float(torch.stack(R).mean()), time_s=round(time.time() - t0, 2)))
    vec.close(); seeds = [cfg.train.eval_seed_base + i for i in range(cfg.train.eval_episodes)]; res = evaluate_policy(cfg, task, adapter, ac, seeds, device); res.update(algo="ppo", env_transitions=env_steps, train_seconds=time.time() - t0)
    rd = Path(run_dir); (rd / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log)); save_checkpoint(rd / "checkpoints", "latest.pt", dict(iteration=iters, algo="ppo", actor_critic=ac.state_dict(), config=cfg.to_dict())); utils.atomic_write_json(rd / "eval" / "eval_final.json", res)
    if iters >= cfg.train.iterations: utils.atomic_write_json(rd / "COMPLETED", dict(algo="ppo"))
    return dict(algo="ppo", success_rate=res["success_rate"], mean_return=res["mean_return"], env_transitions=env_steps)
def _tcp_of(o):
    """TCP of the transition's next state: the vector env returns the post-reset TCP for finished envs, so recompute it from the stored final state is not possible without FK; use the env-provided tcp for running envs
    and the final tcp recorded in the episode summary for finished ones."""
    tcp = o["tcp"].copy()
    ends = {i: e for i, e in zip(np.where(o["done"])[0], o["episodes"])}
    for i, e in ends.items(): tcp[i] = np.asarray(e["final_tcp"], np.float32)
    return tcp
