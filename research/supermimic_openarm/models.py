"""Neural modules: running normaliser, differentiable world model (residual dynamics on the EXECUTED action), actor, critic.
   next_state = world_model(state, executed_action)   with prev_ctrl(next) := executed_action enforced structurally (it is known exactly)."""
import torch, torch.nn as nn
from .envs import layout as L
class RunningNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, clip=10.0):
        super().__init__(); self.register_buffer("mean", torch.zeros(dim)); self.register_buffer("var", torch.ones(dim)); self.register_buffer("count", torch.tensor(1e-4)); self.eps, self.clip = eps, clip
    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).float(); n = x.shape[0]
        if n == 0: return
        bm, bv = x.mean(0), x.var(0, unbiased=False); d = bm - self.mean; tot = self.count + n
        self.mean += d * n / tot; self.var.copy_((self.var * self.count + bv * n + d ** 2 * self.count * n / tot) / tot); self.count.copy_(tot)
    def forward(self, x): return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)
    def denorm_delta(self, z): return z * torch.sqrt(self.var + self.eps) + self.mean
def mlp(i, h, layers, o, act=nn.SiLU, out_gain=None):
    mods = []; d = i
    for _ in range(layers): mods += [nn.Linear(d, h), act()]; d = h
    last = nn.Linear(d, o); 
    if out_gain is not None: nn.init.orthogonal_(last.weight, out_gain); nn.init.zeros_(last.bias)
    return nn.Sequential(*mods, last)
class WorldModel(nn.Module):
    """Action-conditioned residual dynamics + termination head (+ optional contact / reward heads).  PRIVILEGED simulator state in, state out."""
    def __init__(self, goal_dim, hidden=256, layers=3, contact_head=True, reward_head=False):
        super().__init__(); D, A = L.STATE_DIM, L.ACT_DIM; self.goal_dim = goal_dim
        self.state_norm, self.action_norm, self.delta_norm = RunningNorm(D), RunningNorm(A), RunningNorm(D)
        self.dyn = mlp(D + A, hidden, layers, D, out_gain=0.1)
        self.term = mlp(D + goal_dim, max(64, hidden // 2), 2, 1); self.contact = mlp(D, max(64, hidden // 2), 2, L.CONTACT_DIM) if contact_head else None
        self.reward = mlp(D + A + goal_dim, max(64, hidden // 2), 2, 1) if reward_head else None
        w = torch.ones(D); w[L.PREV] = 0.0; self.register_buffer("loss_w", w)             # prev_ctrl is structural, excluded from the loss
    def forward(self, s, a_exec):
        z = self.dyn(torch.cat([self.state_norm(s), self.action_norm(a_exec)], -1)); nxt = s + self.delta_norm.denorm_delta(z)
        out = torch.cat([nxt[:, :L.PREV.start], a_exec, nxt[:, L.PREV.stop:]], -1); return out
    def state_loss(self, pred, target):
        d = (self.state_norm(pred) - self.state_norm(target)) ** 2; return (d * self.loss_w).sum(-1).mean() / self.loss_w.sum()
    def termination_logit(self, s, goal): return self.term(torch.cat([self.state_norm(s), goal], -1)).squeeze(-1)
    def termination_prob(self, s, goal): return torch.sigmoid(self.termination_logit(s, goal))
    def predict_contact(self, s): return torch.sigmoid(self.contact(self.state_norm(s)))
    def predict_reward(self, s, a, goal): return self.reward(torch.cat([self.state_norm(s), self.action_norm(a), goal], -1)).squeeze(-1)
    @torch.no_grad()
    def update_norms(self, s, a, s_next):
        self.state_norm.update(s); self.action_norm.update(a); self.delta_norm.update(s_next - s)
class Actor(nn.Module):
    """Deterministic continuous actor: obs -> tanh in [-1,1]^8.  Exploration noise is added by the collector, never inside imagination."""
    def __init__(self, obs_dim, hidden=256, layers=3):
        super().__init__(); self.obs_norm = RunningNorm(obs_dim); self.net = mlp(obs_dim, hidden, layers, L.ACT_DIM, out_gain=0.01)
    def forward(self, obs): return torch.tanh(self.net(self.obs_norm(obs)))
class Critic(nn.Module):
    """Goal-conditioned state value V(s, goal) on the full (privileged) critic features."""
    def __init__(self, feat_dim, hidden=256, layers=2):
        super().__init__(); self.norm = RunningNorm(feat_dim); self.net = mlp(feat_dim, hidden, layers, 1)
    def forward(self, feat): return self.net(self.norm(feat)).squeeze(-1)
def critic_features(task, s, goal, tcp): return torch.cat([s, goal, tcp], -1)
