"""Per-environment ring buffer with episode-aware window sampling.  A window never crosses an episode boundary (checked via episode ids).
Stores the EXECUTED action (after clip + slew) so the world model is trained on what the simulator really received."""
import numpy as np, torch
from .envs import layout as L
class Replay:
    def __init__(self, n_envs, capacity, goal_dim, seed=0):
        self.n, self.cap, self.gd = n_envs, capacity, goal_dim; self.rng = np.random.default_rng(seed); D = L.STATE_DIM
        z = lambda *s, dt=np.float32: np.zeros((n_envs, capacity) + s, dt)
        self.state, self.next_state, self.goal, self.action, self.reward, self.term, self.trunc, self.contact, self.ep = z(D), z(D), z(goal_dim), z(L.ACT_DIM), z(), z(dt=np.bool_), z(dt=np.bool_), z(L.CONTACT_DIM), z(dt=np.int64)
        self.ptr = 0; self.size = 0; self.ep_counter = np.zeros(n_envs, np.int64); self.total = 0
    def add(self, state, goal, action, next_state, reward, terminated, truncated, contact):
        p = self.ptr; self.state[:, p] = state; self.goal[:, p] = goal; self.action[:, p] = action; self.next_state[:, p] = next_state; self.reward[:, p] = reward; self.term[:, p] = terminated; self.trunc[:, p] = truncated; self.contact[:, p] = contact
        self.ep[:, p] = self.ep_counter; self.ep_counter += (np.asarray(terminated) | np.asarray(truncated)).astype(np.int64)
        self.ptr = (self.ptr + 1) % self.cap; self.size = min(self.size + 1, self.cap); self.total += self.n
    def _idx(self, i): return (self.ptr - self.size + i) % self.cap                      # logical (oldest = 0) -> physical
    def sample_windows(self, B, k):
        """-> dict(states (B,k+1,D), actions (B,k,A), goal (B,g), contact (B,k,3), reward (B,k), term (B,k)); windows lie inside one episode"""
        if self.size < k: raise RuntimeError("replay buffer holds fewer steps than the world-model window")
        out = []; need = B; tries = 0
        while need > 0 and tries < 50:
            m = max(need * 3, 8); e = self.rng.integers(0, self.n, m); i = self.rng.integers(0, self.size - k + 1, m)
            first = self._idx(i); last = self._idx(i + k - 1); ok = self.ep[e, first] == self.ep[e, last]
            for ee, ii in zip(e[ok][:need], i[ok][:need]): out.append((ee, ii))
            need = B - len(out); tries += 1
        if need > 0: raise RuntimeError("could not sample enough in-episode windows (episodes shorter than the window?)")
        E = np.array([o[0] for o in out]); I = np.array([o[1] for o in out]); pos = self._idx(I[:, None] + np.arange(k)[None])           # (B,k)
        S = self.state[E[:, None], pos]; nxt = self.next_state[E, pos[:, -1]]
        return dict(states=np.concatenate([S, nxt[:, None]], 1), actions=self.action[E[:, None], pos], goal=self.goal[E, pos[:, 0]], contact=self.contact[E[:, None], pos], reward=self.reward[E[:, None], pos], term=self.term[E[:, None], pos])
    def sample_transitions(self, B):
        e = self.rng.integers(0, self.n, B); i = self._idx(self.rng.integers(0, self.size, B)); return dict(state=self.state[e, i], next_state=self.next_state[e, i], goal=self.goal[e, i], action=self.action[e, i], term=self.term[e, i], contact=self.contact[e, i], reward=self.reward[e, i])
    def sample_starts(self, B):
        t = self.sample_transitions(B); return t["state"], t["goal"]
    def state_dict(self):
        s = {k: getattr(self, k)[:, :self.size] if k not in ("ep_counter",) and getattr(self, k).ndim > 1 else getattr(self, k) for k in ("state", "next_state", "goal", "action", "reward", "term", "trunc", "contact", "ep")}
        order = np.array([self._idx(i) for i in range(self.size)], np.int64) if self.size else np.zeros(0, np.int64)
        return {**{k: getattr(self, k)[:, order] for k in ("state", "next_state", "goal", "action", "reward", "term", "trunc", "contact", "ep")}, "ep_counter": self.ep_counter, "total": np.array(self.total)}
    def load_state_dict(self, d):
        n = d["state"].shape[1]
        for k in ("state", "next_state", "goal", "action", "reward", "term", "trunc", "contact", "ep"): getattr(self, k)[:, :n] = d[k]
        self.size = n; self.ptr = n % self.cap; self.ep_counter = d["ep_counter"].copy(); self.total = int(d["total"])
