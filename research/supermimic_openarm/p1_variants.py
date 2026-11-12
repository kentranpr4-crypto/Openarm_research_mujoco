"""OPT-IN actor-experiment variants for P1 pushing.  B0 is NOT in this file: B0 is the original SuperMimic trainer (`train.py` / `trainer.Trainer`), untouched and still the default.  Nothing in the tracked source tree imports this module.
B1  action trust region : in the imagined rollouts the policy action is a_ref(obs) + clamp(a(obs) - a_ref(obs), -rho, +rho), where a_ref is a slowly moving (EMA, factor `ema`) copy of the actor.  Gradients vanish for actions outside the radius, which bounds how far the actor can
                          move away from its lagged self per update (the closed-loop analogue of the +/-0.1 open-loop trust region that cut the proxy optimism from +8.4 to +2.8).  The collector still uses the unclamped actor.
B2  ensemble penalty    : `members` world models (the primary one plus members-1 extra) are trained on the same replay; each imagined step uses the primary actor's action in all of them, the imagined reward is the MEAN reward over members minus lam * 100 * ||std over members of the predicted box position|| (1 cm of disagreement = lam reward units).
B3  contact gating      : the world model's box dynamics are gated: box_next = box + g (wm_box_next - box), g = max(sigmoid(k (r0 - |tcp - box|)), sigmoid(kv (|v_box| - v0))), with r0, k, v0, kv learned.  With a still box and the pusher out of reach the predicted box change is the ungated change times g (g ~ 0.05 at initialisation because of the rest-velocity branch, learned afterwards; the proxy ablation measured 0.79 mm drift at h=16 after training).
Checkpoints carry `variant` (and the extras) so p1_wm_eval / p1_actor_eval can rebuild the right world model.  Defaults (`DEFAULTS`) are the values used in the proxy ablations; every one can be set with --variant-arg on the CLI (p1_train_variant.py)."""
import math
from copy import deepcopy
import torch
import torch.nn as nn
from .envs import layout as L
from .models import WorldModel, Actor
from . import trainer as T
from .checkpoint import load_checkpoint
VARIANTS = ("b0", "b1", "b2", "b3")
DEFAULTS = dict(rho=0.1, ema=0.02, lam=3.0, members=3, gate_r0=0.10, gate_k=60.0, gate_v0=0.01, gate_kv=300.0)
class GatedWorldModel(WorldModel):
    """WorldModel whose BOX outputs (position, quaternion, velocity) are gated by pusher proximity or existing box motion (see module docstring)."""
    def __init__(self, goal_dim, hidden, layers, contact_head, reward_head, fk, r0=0.10, k=60.0, v0=0.01, kv=300.0):
        super().__init__(goal_dim, hidden, layers, contact_head, reward_head); object.__setattr__(self, "_fk", fk)                  # fk is shared with the trainer; not registered as a submodule (keeps state_dict small and portable)
        self.r0 = nn.Parameter(torch.tensor(float(r0))); self.logk = nn.Parameter(torch.tensor(math.log(k))); self.v0 = nn.Parameter(torch.tensor(float(v0))); self.logkv = nn.Parameter(torch.tensor(math.log(kv)))
    def gate(self, s):
        d = (self._fk(s[:, :8]) - s[:, L.BOX_POS]).norm(dim=-1); gp = torch.sigmoid(self.logk.exp() * (self.r0 - d)); gv = torch.sigmoid(self.logkv.exp() * (s[:, L.BOX_VEL].norm(dim=-1) - self.v0)); return torch.maximum(gp, gv)[:, None]
    def forward(self, s, a_exec):
        out = super().forward(s, a_exec); b0 = L.BOX_POS.start; g = self.gate(s); return torch.cat([out[:, :b0], s[:, b0:] + g * (out[:, b0:] - s[:, b0:])], -1)
def build_world_model(variant, goal_dim, w, fk, device, **gate):
    """the world model class a checkpoint of `variant` was trained with (w = cfg.world_model)"""
    if variant == "b3": return GatedWorldModel(goal_dim, w.hidden, w.layers, w.contact_head, w.reward_head, fk, **{k[5:]: v for k, v in {**DEFAULTS, **gate}.items() if k.startswith("gate_")}).to(device)
    return WorldModel(goal_dim, w.hidden, w.layers, w.contact_head, w.reward_head).to(device)
def load_actor(cfg, ck, device, task):
    """actor + torch adapter from a checkpoint of ANY variant, without touching the world model (so a gated / ensemble checkpoint loads too)"""
    fk, adapter, _ = T.build_fk_and_adapter(cfg, device); a = Actor(task.actor_obs_dim, cfg.actor.hidden, cfg.actor.layers).to(device); a.load_state_dict(ck["actor"]); a.eval(); return a, adapter
def make_ensemble_imagine(members, lam, stats):
    """drop-in replacement of imagination.imagine for B2 (same signature and return structure)"""
    def imagine_e(wm, policy, task, fk, adapter, s0, goal, horizon):
        ws = [wm] + list(members); S = [s0 for _ in ws]; s = s0; tcp = fk(s[:, :8]); states, tcps, rewards, terms, actions, execs = [s], [tcp], [], [], [], []
        for t in range(horizon):
            obs = task.actor_obs(s, goal, tcp); a = policy(obs, t); a_exec = adapter.act(a, s[:, L.PREV])
            S = [w(S[i], adapter.act(a, S[i][:, L.PREV])) for i, w in enumerate(ws)]; s_next = S[0]; tcp_next = fk(s_next[:, :8])
            r_mean = sum(task.reward_torch(S[i], a_exec, goal, fk(S[i][:, :8])) for i in range(len(ws))) / len(ws); dis = torch.stack([x[:, L.BOX_POS] for x in S]).std(0).norm(dim=-1) * 100.0; stats.append(float(dis.detach().mean()))
            states.append(s_next); tcps.append(tcp_next); rewards.append(r_mean - lam * dis); terms.append(wm.termination_prob(s_next, goal)); actions.append(a); execs.append(a_exec); s, tcp = s_next, tcp_next
        return dict(states=states, tcps=tcps, rewards=rewards, terms=terms, actions=actions, executed=execs)
    return imagine_e
class P1VariantTrainer(T.Trainer):
    def __init__(self, cfg, run_dir, variant, resume=False, log=print, **vargs):
        if variant not in VARIANTS[1:]: raise ValueError(f"variant must be one of {VARIANTS[1:]} (B0 is the original Trainer); got {variant!r}")
        unknown = set(vargs) - set(DEFAULTS)
        if unknown: raise ValueError(f"unknown variant argument(s): {sorted(unknown)}; known: {sorted(DEFAULTS)}")
        super().__init__(cfg, run_dir, resume=False, log=log); self.variant = variant; self.v = {**DEFAULTS, **{k: type(DEFAULTS[k])(x) for k, x in vargs.items()}}; w = cfg.world_model; self._stats = []
        if variant == "b3": self.wm = build_world_model("b3", self.gd, w, self.fk, self.device, **{k: self.v[k] for k in self.v if k.startswith("gate_")}); self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=w.lr)
        if variant == "b2":
            self.members = [WorldModel(self.gd, w.hidden, w.layers, w.contact_head, w.reward_head).to(self.device) for _ in range(int(self.v["members"]) - 1)]; self.member_opts = [torch.optim.Adam(m.parameters(), lr=w.lr) for m in self.members]
        if variant == "b1": self.ref_net = deepcopy(self.actor.net); [p.requires_grad_(False) for p in self.ref_net.parameters()]
        if resume: self._load_latest()
    # ---- checkpoints --------------------------------------------------------------------------------------------------------------------------
    def payload(self):
        d = super().payload(); d["variant"] = self.variant; d["variant_args"] = dict(self.v)
        if self.variant == "b2": d["wm_members"] = [m.state_dict() for m in self.members]; d["member_opts"] = [o.state_dict() for o in self.member_opts]
        if self.variant == "b1": d["ref_net"] = self.ref_net.state_dict()
        return d
    def _load_latest(self):
        p = self.ckpt_dir / "latest.pt"
        if not p.is_file(): raise FileNotFoundError(f"nothing to resume: {p} does not exist")
        ck = load_checkpoint(p, self.device)
        if ck.get("variant") != self.variant: raise ValueError(f"checkpoint was written by variant {ck.get('variant')!r}, not {self.variant!r}")        # checked BEFORE any state is loaded
        super()._load_latest()
        if self.variant == "b2":
            for m, sd in zip(self.members, ck["wm_members"]): m.load_state_dict(sd)
            for o, sd in zip(self.member_opts, ck["member_opts"]): o.load_state_dict(sd)
        if self.variant == "b1": self.ref_net.load_state_dict(ck["ref_net"])
    # ---- world model ------------------------------------------------------------------------------------------------------------------------------
    def update_world_model(self):
        m = super().update_world_model()
        if self.variant == "b2":
            for i, (mem, opt) in enumerate(zip(self.members, self.member_opts)):
                keep = (self.wm, self.wm_opt); self.wm, self.wm_opt = mem, opt
                try: mm = super().update_world_model()
                finally: self.wm, self.wm_opt = keep
                m[f"wm_loss_member{i + 1}"] = mm["wm_loss"]
        return m
    # ---- actor ----------------------------------------------------------------------------------------------------------------------------------------
    def actor_policy(self):
        if self.variant != "b1": return super().actor_policy()
        rho = float(self.v["rho"])
        def pol(obs, t):
            a = self.actor(obs)
            with torch.no_grad(): ref = torch.tanh(self.ref_net(self.actor.obs_norm(obs)))
            dev = a - ref; self._stats.append((float(dev.detach().abs().mean()), float((dev.detach().abs() > rho).float().mean())))
            return ref + dev.clamp(-rho, rho)
        return pol
    def update_actor_critic(self):
        self._stats = []
        if self.variant == "b1":
            with torch.no_grad():
                for pr, p in zip(self.ref_net.parameters(), self.actor.net.parameters()): pr.mul_(1.0 - float(self.v["ema"])).add_(p.detach(), alpha=float(self.v["ema"]))
            out = super().update_actor_critic(); arr = self._stats
            out["b1_mean_abs_dev_from_ref"] = float(sum(x[0] for x in arr) / max(len(arr), 1)); out["b1_frac_outside_radius"] = float(sum(x[1] for x in arr) / max(len(arr), 1)); return out
        if self.variant == "b2":
            real = T.imagine; T.imagine = make_ensemble_imagine(self.members, float(self.v["lam"]), self._stats); [p.requires_grad_(False) for m in self.members for p in m.parameters()]
            for m in self.members: m.eval()
            try: out = super().update_actor_critic()
            finally:
                T.imagine = real; [p.requires_grad_(True) for m in self.members for p in m.parameters()]
            out["b2_mean_box_disagreement_cm"] = float(sum(self._stats) / max(len(self._stats), 1)); return out
        return super().update_actor_critic()
