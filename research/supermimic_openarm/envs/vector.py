"""Auto-resetting vectorised environments: `SyncVectorEnv` (same process) and `SubprocVectorEnv` (one spawned worker process per env; MuJoCo CPU).
Episode seeds are a deterministic function of (base_seed, env_index, episode_index); protected seeds 31-33 are rejected by config validation."""
import numpy as np, multiprocessing as mp
from .openarm_env import OpenArmTaskEnv
def episode_seed(base, env_idx, ep_idx): return int(np.random.SeedSequence([int(base), int(env_idx), int(ep_idx)]).generate_state(1)[0])
class AutoResetEnv:
    def __init__(self, task_name, task_params, max_steps, control_hz, base_seed, env_idx):
        self.env = OpenArmTaskEnv(task_name, task_params, max_steps, control_hz); self.base, self.idx, self.ep = base_seed, env_idx, 0; self.state = self.goal = self.tcp = None
    def reset(self, seed=None):
        seed = episode_seed(self.base, self.idx, self.ep) if seed is None else int(seed); self.ep += 1; self.state, self.goal, self.tcp = self.env.reset(seed); return self.state, self.goal, self.tcp
    def step(self, target):
        o = self.env.step(target); done = o["terminated"] or o["truncated"]; o["done"] = done; o["goal_used"] = o["goal"].copy()
        if done: o["episode"] = dict(ret=self.env.episode_return, length=self.env.t, success=self.env.ever_success, **{k: v for k, v in self.env.episode_info.items() if k == "seed"}, final_tcp=o["tcp"].tolist(), box_max_rise=self.env.box_max_rise); o["final_state"] = o["state"]; self.reset()
        o["next_state"] = o["final_state"] if done else o["state"]; o["state"], o["goal"], o["tcp"] = (self.state, self.goal, self.tcp) if done else (o["state"], o["goal"], o["tcp"]); return o
def _worker(conn, args):
    env = AutoResetEnv(*args)
    try:
        while True:
            cmd, x = conn.recv()
            if cmd == "reset": conn.send(env.reset(x))
            elif cmd == "step": conn.send(env.step(x))
            elif cmd == "close": break
    except (EOFError, KeyboardInterrupt): pass
    finally: conn.close()
class _VecBase:
    def _stack(self, outs):
        keys = ["state", "next_state", "goal", "goal_used", "tcp", "contact", "executed"]; r = {k: np.stack([o[k] for o in outs]) for k in keys}
        r["reward"] = np.array([o["reward"] for o in outs], np.float32); r["terminated"] = np.array([o["terminated"] for o in outs]); r["truncated"] = np.array([o["truncated"] for o in outs]); r["done"] = np.array([o["done"] for o in outs])
        r["success_so_far"] = np.array([o["info"]["ever_success"] for o in outs]); r["episodes"] = [o["episode"] for o in outs if "episode" in o]; r["info"] = [o["info"] for o in outs]; return r
class SyncVectorEnv(_VecBase):
    def __init__(self, n, task_name, task_params, max_steps, control_hz, base_seed):
        self.envs = [AutoResetEnv(task_name, task_params, max_steps, control_hz, base_seed, i) for i in range(n)]; self.n = n
    def reset(self, seeds=None):
        outs = [e.reset(None if seeds is None else seeds[i]) for i, e in enumerate(self.envs)]; return np.stack([o[0] for o in outs]), np.stack([o[1] for o in outs]), np.stack([o[2] for o in outs])
    def step(self, targets): return self._stack([e.step(targets[i]) for i, e in enumerate(self.envs)])
    def close(self): pass
class SubprocVectorEnv(_VecBase):
    def __init__(self, n, task_name, task_params, max_steps, control_hz, base_seed):
        ctx = mp.get_context("spawn"); self.n = n; self.conns = []; self.procs = []
        for i in range(n):
            a, b = ctx.Pipe(); p = ctx.Process(target=_worker, args=(b, (task_name, task_params, max_steps, control_hz, base_seed, i)), daemon=True); p.start(); b.close(); self.conns.append(a); self.procs.append(p)
    def reset(self, seeds=None):
        for i, c in enumerate(self.conns): c.send(("reset", None if seeds is None else seeds[i]))
        outs = [c.recv() for c in self.conns]; return np.stack([o[0] for o in outs]), np.stack([o[1] for o in outs]), np.stack([o[2] for o in outs])
    def step(self, targets):
        for i, c in enumerate(self.conns): c.send(("step", targets[i]))
        return self._stack([c.recv() for c in self.conns])
    def close(self):
        for c in self.conns:
            try: c.send(("close", None))
            except Exception: pass
        for p in self.procs: p.join(timeout=5)
def make_vec_env(backend, n, task_name, task_params, max_steps, control_hz, base_seed):
    return (SyncVectorEnv if backend == "sync" else SubprocVectorEnv)(n, task_name, task_params, max_steps, control_hz, base_seed)
