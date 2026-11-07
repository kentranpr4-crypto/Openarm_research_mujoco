"""Training orchestrator: collect -> fit world model -> imagine (actor + critic) -> evaluate -> checkpoint.  CPU debugging and single-GPU CUDA."""
import json, math, os, time
from pathlib import Path
import numpy as np, torch
from . import utils
from .adapter import TorchAdapter
from .checkpoint import save_checkpoint, load_checkpoint
from .config import Config
from .envs import layout as L
from .envs.tasks import make_task
from .envs.vector import make_vec_env, SyncVectorEnv
from .imagination import imagine, survival_weighted_return, critic_loss_on, set_requires_grad
from .models import WorldModel, Actor, Critic, critic_features
from .replay import Replay
def resolve_device(cfg):
    d = cfg.system.device
    if d == "auto": d = "cuda" if torch.cuda.is_available() else "cpu"
    if d.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("system.device is CUDA but torch.cuda.is_available() is False (driver / CUDA runtime / PyTorch build). Run scripts/check_gpu.sh")
    if cfg.system.precision == "bf16" and d.startswith("cuda") and not torch.cuda.is_bf16_supported(): raise RuntimeError("bf16 requested but not supported on this GPU; use precision: fp32")
    return torch.device(d)
def build_fk_and_adapter(cfg, device):
    """loads the MuJoCo model once in this process (needed for the differentiable FK and the limits); returns (fk, adapter, model info)"""
    from .envs.openarm_env import _load_sim
    from .kinematics import TorchFK
    M = _load_sim(); m, d = M["lab_scene"].compile_lab(with_cameras=False); A = M["ActionAdapter"](m, cfg.env.control_hz)
    act_of_qadr = {int(q): i for i, q in enumerate(A.qadr[:8])}; fk = TorchFK(m, "openarm_left_hand_tcp", act_of_qadr).to(device)
    adapter = TorchAdapter(A.limits, 1.0 / cfg.env.control_hz, cfg.actor.action_param, cfg.actor.action_scale, device=device)
    return fk, adapter, dict(model_nq=int(m.nq), model_nu=int(m.nu), timestep=float(m.opt.timestep), lo=A.limits.lo[:8].tolist(), hi=A.limits.hi[:8].tolist(), max_speed=A.limits.max_speed[:8].tolist())
class Trainer:
    def __init__(self, cfg: Config, run_dir, resume=False, log=print):
        cfg.validate(); self.cfg, self.run_dir, self.log = cfg, Path(run_dir), log; self.device = resolve_device(cfg); self._setup_torch()
        self.task = make_task(cfg.env.task, **cfg.env.task_params); 
        if cfg.env.max_steps: self.task.max_steps = cfg.env.max_steps
        self.fk, self.adapter, self.model_info = build_fk_and_adapter(cfg, self.device); self.gd = self.task.goal_dim; w, a, c = cfg.world_model, cfg.actor, cfg.critic
        self.wm = WorldModel(self.gd, w.hidden, w.layers, w.contact_head, w.reward_head).to(self.device); self.actor = Actor(self.task.actor_obs_dim, a.hidden, a.layers).to(self.device)
        self.critic = Critic(L.STATE_DIM + self.gd + 3, c.hidden, c.layers).to(self.device)
        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=w.lr); self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=a.lr); self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=c.lr)
        self.replay = Replay(cfg.env.num_envs, cfg.replay.capacity_per_env, self.gd, seed=cfg.experiment.seed); self.iteration = 0; self.env_steps = 0; self.best = -float("inf"); self.resumes = 0; self.vec = None; self.history = []
        self.ckpt_dir = self.run_dir / "checkpoints"; self.metrics_path = self.run_dir / "metrics.jsonl"; self.autocast = (self.cfg.system.precision == "bf16")
        if resume: self._load_latest()
    # -------------------------------------------------------------------------------------------------------------------------------------
    def _setup_torch(self):
        s = self.cfg.system
        if s.torch_threads > 0: torch.set_num_threads(s.torch_threads)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = bool(s.tf32); torch.backends.cudnn.allow_tf32 = bool(s.tf32); total = torch.cuda.get_device_properties(self.device).total_memory / 2 ** 30
            torch.cuda.set_per_process_memory_fraction(min(1.0, s.max_vram_gb / total), self.device)            # OOM becomes an explicit error instead of exhausting the card
        utils.seed_everything(self.cfg.experiment.seed)
    def _ac(self): return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.autocast)
    def _t(self, x): return torch.as_tensor(x, dtype=torch.float32, device=self.device)
    # ---- data collection --------------------------------------------------------------------------------------------------------------
    def _make_vec(self):
        e = self.cfg.env; base = self.cfg.train.train_seed_base + 1_000_003 * self.resumes
        self.vec = make_vec_env(e.backend, e.num_envs, e.task, e.task_params, self.task.max_steps, e.control_hz, base); self.cur_state, self.cur_goal, self.cur_tcp = self.vec.reset()
        self.cur_state = self.cur_state.astype(np.float32); self.cur_goal = self.cur_goal.astype(np.float32); self.ep_stats = []
    def policy_targets(self, state, goal, tcp, noise_std, random=False):
        n = state.shape[0]
        if random: a = np.random.uniform(-1, 1, (n, L.ACT_DIM)).astype(np.float32)
        else:
            with torch.no_grad(), self._ac(): a = self.actor(self.task.actor_obs(self._t(state), self._t(goal), self._t(tcp))).float().cpu().numpy()
            if noise_std > 0: a = np.clip(a + noise_std * np.random.randn(*a.shape), -1, 1)
        with torch.no_grad(): tgt = self.adapter.target_from_action(torch.as_tensor(a, device=self.device), torch.as_tensor(state[:, L.PREV], device=self.device)).cpu().numpy().astype(np.float64)
        return tgt
    def collect(self, steps, random=False):
        for _ in range(steps):
            tgt = self.policy_targets(self.cur_state, self.cur_goal, self.cur_tcp, self.cfg.actor.noise_std, random); o = self.vec.step(tgt)
            self.replay.add(self.cur_state, o["goal_used"], o["executed"], o["next_state"], o["reward"], o["terminated"], o["truncated"], o["contact"])
            self.cur_state, self.cur_goal, self.cur_tcp = o["state"].astype(np.float32), o["goal"].astype(np.float32), o["tcp"]; self.env_steps += self.cfg.env.num_envs; self.ep_stats += o["episodes"]
    # ---- world model ------------------------------------------------------------------------------------------------------------------
    def update_world_model(self):
        w = self.cfg.world_model; k = w.window; losses, terml, contl = [], [], []; self.wm.train()
        big = self.replay.sample_transitions(min(self.replay.size * self.cfg.env.num_envs, 4096)); self.wm.update_norms(self._t(big["state"]), self._t(big["action"]), self._t(big["next_state"]))
        for _ in range(w.updates_per_iter):
            b = self.replay.sample_windows(w.batch_size, k); S, A = self._t(b["states"]), self._t(b["actions"]); goal = self._t(b["goal"]); pred = S[:, 0]; loss = 0.0
            with self._ac():
                for j in range(k): pred = self.wm(pred, A[:, j]); loss = loss + self.wm.state_loss(pred.float(), S[:, j + 1])
            loss = loss / k; t = self.replay.sample_transitions(w.batch_size); ns, tg, tt = self._t(t["next_state"]), self._t(t["goal"]), self._t(t["term"])
            lt = torch.nn.functional.binary_cross_entropy_with_logits(self.wm.termination_logit(ns, tg), tt, pos_weight=torch.tensor(5.0, device=self.device)); loss = loss + w.term_weight * lt; terml.append(float(lt.detach()))
            if self.wm.contact is not None: lc = torch.nn.functional.mse_loss(self.wm.predict_contact(ns), self._t(t["contact"])); loss = loss + w.contact_weight * lc; contl.append(float(lc.detach()))
            if self.wm.reward is not None: lr = torch.nn.functional.mse_loss(self.wm.predict_reward(self._t(t["state"]), self._t(t["action"]), tg), self._t(t["reward"])); loss = loss + w.reward_weight * lr
            self.wm_opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss): loss.backward(); torch.nn.utils.clip_grad_norm_(self.wm.parameters(), w.grad_clip); self.wm_opt.step(); losses.append(float(loss.detach()))
        return dict(wm_loss=float(np.mean(losses)) if losses else float("nan"), wm_term_loss=float(np.mean(terml)) if terml else 0.0, wm_contact_loss=float(np.mean(contl)) if contl else 0.0, wm_updates=len(losses))
    @torch.no_grad()
    def wm_window_error(self, n=512):
        k = self.cfg.world_model.window; b = self.replay.sample_windows(n, k); S, A = self._t(b["states"]), self._t(b["actions"]); pred = S[:, 0]; errs = []; self.wm.eval()
        for j in range(k): pred = self.wm(pred, A[:, j]); errs.append((((self.wm.state_norm(pred) - self.wm.state_norm(S[:, j + 1])) ** 2) * self.wm.loss_w).sum(-1).div(self.wm.loss_w.sum()).mean())
        return float(torch.stack(errs).mean().sqrt()), float(torch.stack(errs)[0].sqrt())
    # ---- actor / critic through imagination ---------------------------------------------------------------------------------------------------
    def actor_policy(self): return lambda obs, t: self.actor(obs)
    def update_actor_critic(self):
        im = self.cfg.imagination; H = im.horizon; total = im.batch_size; chunk = max(1, min(im.chunk_size, total)); s0, g0 = self.replay.sample_starts(total); s0, g0 = self._t(s0), self._t(g0)
        set_requires_grad(self.wm, False); set_requires_grad(self.critic, False); set_requires_grad(self.actor, True); self.wm.eval(); self.actor_opt.zero_grad(set_to_none=True)
        objs, rsum, bad = 0.0, 0.0, 0; trajs = []; term_mean = 0.0
        for a in range(0, total, chunk):
            b = min(a + chunk, total); wgt = (b - a) / total
            with torch.enable_grad(), self._ac():
                tr = imagine(self.wm, self.actor_policy(), self.task, self.fk, self.adapter, s0[a:b], g0[a:b], H)
                v_last = self.critic(critic_features(self.task, tr["states"][-1], g0[a:b], tr["tcps"][-1])); J, surv = survival_weighted_return(tr["rewards"], tr["terms"], v_last.float(), im.gamma); loss = -(J.float().mean()) * wgt
            if torch.isfinite(loss): loss.backward()
            else: bad += 1
            bad += sum(int((~torch.isfinite(p.grad)).sum()) for p in self.actor.parameters() if p.grad is not None)
            objs += float(J.detach().mean()) * wgt; rsum += float(torch.stack(tr["rewards"]).detach().sum(0).mean()) * wgt; term_mean += float(torch.stack(tr["terms"]).detach().mean()) * wgt
            trajs.append(dict(states=[x.detach() for x in tr["states"]], tcps=[x.detach() for x in tr["tcps"]], rewards=[x.detach() for x in tr["rewards"]], terms=[x.detach() for x in tr["terms"]], goal=g0[a:b]))
        gn = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), im.grad_clip); gnorm = float(gn)
        if bad == 0 and math.isfinite(gnorm):
            for p in self.actor.parameters():
                if p.grad is not None: p.grad.clamp_(-im.grad_clip_elementwise, im.grad_clip_elementwise)
            self.actor_opt.step()
        else: self.actor_opt.zero_grad(set_to_none=True)
        set_requires_grad(self.wm, True); set_requires_grad(self.critic, True)
        traj = dict(states=[torch.cat([t["states"][i] for t in trajs]) for i in range(H + 1)], tcps=[torch.cat([t["tcps"][i] for t in trajs]) for i in range(H + 1)], rewards=[torch.cat([t["rewards"][i] for t in trajs]) for i in range(H)], terms=[torch.cat([t["terms"][i] for t in trajs]) for i in range(H)])
        cl = critic_loss_on(traj, self.critic, self.task, torch.cat([t["goal"] for t in trajs]), im.gamma, im.lam, self.cfg.critic.epochs, self.cfg.critic.minibatches, self.critic_opt)
        obs_batch = self.task.actor_obs(s0, g0, self.fk(s0[:, :8])); self.actor.obs_norm.update(obs_batch.detach())
        return dict(imag_objective=objs, imag_reward_sum=rsum, imag_term_prob=term_mean, actor_grad_norm=gnorm, bad_grads=bad, critic_loss=cl)
    # ---- evaluation ------------------------------------------------------------------------------------------------------------------------------------
    def evaluate(self, n_episodes=None, seeds=None, save_traj=0, tag="eval", actor=None):
        from .envs.openarm_env import OpenArmTaskEnv
        cfg = self.cfg; n = n_episodes or cfg.train.eval_episodes; seeds = seeds or [cfg.train.eval_seed_base + i for i in range(n)]; utils.assert_seeds_allowed(*seeds)
        env = OpenArmTaskEnv(cfg.env.task, cfg.env.task_params, self.task.max_steps, cfg.env.control_hz); actor = actor or self.actor; actor.eval(); self.wm.eval(); rows = []; os_err, ro_err = [], []; trajs = []
        for ei, sd in enumerate(seeds):
            s, g, tcp = env.reset(sd); S, TCP, EX, R, C, SUCC, ACT = [s], [tcp], [], [], [], [], []; done = False; ep_ret = 0.0
            while not done:
                with torch.no_grad(): a = actor(self.task.actor_obs(self._t(s)[None], self._t(g)[None], self._t(tcp)[None])).float(); tgt = self.adapter.target_from_action(a, self._t(s)[None][:, L.PREV])[0].cpu().numpy().astype(np.float64)
                o = env.step(tgt); s, tcp = o["state"], o["tcp"]; S.append(s); TCP.append(tcp); EX.append(o["executed"]); R.append(o["reward"]); C.append(o["contact"]); SUCC.append(o["info"]["success"]); ACT.append(a[0].cpu().numpy()); ep_ret += o["reward"]; done = o["terminated"] or o["truncated"]
            S, EX = np.stack(S), np.stack(EX); gnum = g
            with torch.no_grad():
                St, Et = self._t(S), self._t(EX); one = self.wm(St[:-1], Et); e1 = (((self.wm.state_norm(one) - self.wm.state_norm(St[1:])) ** 2) * self.wm.loss_w).sum(-1).div(self.wm.loss_w.sum()); ro = [St[0:1]]
                for j in range(len(EX)): ro.append(self.wm(ro[-1], Et[j:j + 1]))
                ro = torch.cat(ro[1:]); er = (((self.wm.state_norm(ro) - self.wm.state_norm(St[1:])) ** 2) * self.wm.loss_w).sum(-1).div(self.wm.loss_w.sum())
            os_err.append(float(e1.mean().sqrt())); ro_err.append(float(er.mean().sqrt()))
            rows.append(dict(seed=int(sd), success=bool(env.ever_success), ret=float(ep_ret), length=int(env.t), final_goal_error_m=float(self._goal_error(env, S[-1], TCP[-1], gnum)), box_max_rise_m=float(env.box_max_rise), terminated=bool(o["terminated"]), max_finger_force_N=0.0))
            if ei < save_traj: trajs.append(dict(seed=sd, states=S, executed=EX, goal=gnum, tcp=np.stack(TCP), reward=np.array(R), contact=np.stack(C), success=np.array(SUCC), wm_one_step=one.cpu().numpy(), wm_rollout=ro.cpu().numpy(), actions=np.stack(ACT)))
        res = dict(tag=tag, n_episodes=len(rows), success_rate=float(np.mean([r["success"] for r in rows])), mean_return=float(np.mean([r["ret"] for r in rows])), mean_length=float(np.mean([r["length"] for r in rows])), mean_final_goal_error_m=float(np.mean([r["final_goal_error_m"] for r in rows])),
                   wm_one_step_rmse_norm=float(np.mean(os_err)), wm_rollout_rmse_norm=float(np.mean(ro_err)), seeds=[int(x) for x in seeds], episodes=rows, task=cfg.env.task, privileged_state=True, note="privileged-state simulation evaluation; held-out goals (seeds disjoint from training); not Sim2Real")
        self.actor.train(); return res, trajs
    def _goal_error(self, env, s, tcp, g):
        if self.cfg.env.task == "reaching": return np.linalg.norm(np.asarray(tcp) - np.asarray(g))
        return np.linalg.norm(np.asarray(s[L.BOX_POS][:2]) - np.asarray(g[:2]))
    # ---- checkpoints -------------------------------------------------------------------------------------------------------------------------------------
    def payload(self):
        return dict(iteration=self.iteration, env_steps=self.env_steps, best=self.best, resumes=self.resumes, wm=self.wm.state_dict(), actor=self.actor.state_dict(), critic=self.critic.state_dict(), wm_opt=self.wm_opt.state_dict(), actor_opt=self.actor_opt.state_dict(), critic_opt=self.critic_opt.state_dict(),
                    config=self.cfg.to_dict(), rng=utils.rng_state(), replay_rng=self.replay.rng.bit_generator.state, git=utils.git_info(), history=self.history[-200:])
    def save(self, name="latest.pt"):
        info = save_checkpoint(self.ckpt_dir, name, self.payload())
        if self.cfg.train.save_replay and name == "latest.pt": np.savez_compressed(self.run_dir / "replay_latest.npz", **self.replay.state_dict())
        return info
    def _load_latest(self):
        p = self.ckpt_dir / "latest.pt"
        if not p.is_file(): raise FileNotFoundError(f"nothing to resume: {p} does not exist")
        ck = load_checkpoint(p, self.device); self.wm.load_state_dict(ck["wm"]); self.actor.load_state_dict(ck["actor"]); self.critic.load_state_dict(ck["critic"]); self.wm_opt.load_state_dict(ck["wm_opt"]); self.actor_opt.load_state_dict(ck["actor_opt"]); self.critic_opt.load_state_dict(ck["critic_opt"])
        self.iteration, self.env_steps, self.best, self.resumes, self.history = ck["iteration"], ck["env_steps"], ck["best"], ck["resumes"] + 1, ck.get("history", []); utils.set_rng_state(ck["rng"]); self.replay.rng.bit_generator.state = ck["replay_rng"]
        rp = self.run_dir / "replay_latest.npz"
        if rp.is_file():
            z = np.load(rp); self.replay.load_state_dict({k: z[k] for k in z.files}); self.log(f"resumed replay buffer: {self.replay.size} steps/env")
        self.log(f"resumed from iteration {self.iteration} (env steps {self.env_steps})")
    # ---- main loop ---------------------------------------------------------------------------------------------------------------------------------------
    def log_metrics(self, row):
        row["time_s"] = round(time.time() - self.t0, 2); self.history.append(row); utils.atomic_write_bytes(self.metrics_path, (self.metrics_path.read_bytes() if self.metrics_path.is_file() else b"") + (json.dumps(row, default=utils._jd) + "\n").encode())
    def train(self, max_iterations=None):
        cfg, tc = self.cfg, self.cfg.train; self.t0 = time.time(); self.run_dir.mkdir(parents=True, exist_ok=True); self._make_vec()
        if self.replay.size * cfg.env.num_envs < cfg.world_model.batch_size // 8 or self.replay.size < cfg.world_model.window + 2:
            self.collect(max(tc.warmup_steps, cfg.world_model.window + 2), random=True)
        last = tc.iterations if max_iterations is None else min(tc.iterations, self.iteration + max_iterations)
        try:
            while self.iteration < last:
                self.iteration += 1; t1 = time.time(); self.collect(tc.steps_per_iter); t2 = time.time(); wmm = self.update_world_model(); t3 = time.time(); acm = self.update_actor_critic(); t4 = time.time()
                row = dict(iteration=self.iteration, env_steps=self.env_steps, **wmm, **acm, t_collect=round(t2 - t1, 3), t_wm=round(t3 - t2, 3), t_actor=round(t4 - t3, 3))
                if self.ep_stats: row["train_ep_return"] = float(np.mean([e["ret"] for e in self.ep_stats[-20:]])); row["train_ep_success"] = float(np.mean([e["success"] for e in self.ep_stats[-20:]]))
                if self.iteration % tc.eval_interval == 0 or self.iteration == tc.iterations:
                    res, trajs = self.evaluate(save_traj=2, tag=f"it{self.iteration:06d}"); row.update(eval_success=res["success_rate"], eval_return=res["mean_return"], eval_goal_err=res["mean_final_goal_error_m"], eval_wm_rmse=res["wm_rollout_rmse_norm"])
                    (self.run_dir / "eval").mkdir(exist_ok=True); utils.atomic_write_json(self.run_dir / "eval" / f"eval_it{self.iteration:06d}.json", res); (self.run_dir / "trajectories").mkdir(exist_ok=True)
                    for k, t in enumerate(trajs): np.savez_compressed(self.run_dir / "trajectories" / f"eval_it{self.iteration:06d}_ep{k}.npz", **{kk: (np.asarray(v) if not isinstance(v, int) else np.array(v)) for kk, v in t.items()})
                    score = res["mean_return"]
                    if score > self.best: self.best = score; self.save("best.pt")
                if self.iteration % tc.log_interval == 0: self.log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items() if k in ("iteration", "env_steps", "wm_loss", "imag_objective", "actor_grad_norm", "bad_grads", "critic_loss", "eval_success", "eval_return", "time_s")))
                self.log_metrics(row)
                if self.iteration % tc.checkpoint_interval == 0 or self.iteration == last: self.save("latest.pt")
        finally:
            if self.vec is not None: self.vec.close()
        if self.iteration >= tc.iterations: utils.atomic_write_json(self.run_dir / "COMPLETED", dict(iteration=self.iteration, env_steps=self.env_steps, finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        return dict(iteration=self.iteration, env_steps=self.env_steps, best=self.best)
