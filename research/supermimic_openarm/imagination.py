"""Imagination-based policy optimisation:  differentiable H-step rollouts of the frozen world model, actor actions conditioned on imagined states,
analytic rewards of imagined states, survival-weighted return + critic bootstrap, backpropagation THROUGH the dynamics into the actor.
   J = sum_t gamma^t S_t r_t + gamma^H S_H V(s_H),   S_0 = 1,  S_{t+1} = S_t (1 - d(s_{t+1})),  d = predicted termination probability."""
import torch
from .envs import layout as L
from .models import critic_features
def set_requires_grad(mod_or_params, flag):
    ps = mod_or_params.parameters() if hasattr(mod_or_params, "parameters") else mod_or_params
    for p in ps: p.requires_grad_(flag)
def imagine(wm, policy, task, fk, adapter, s0, goal, horizon):
    """policy(obs, t) -> action in [-1,1]^8 (gradient-carrying).  Returns lists over time; everything is differentiable w.r.t. the policy parameters."""
    s = s0; tcp = fk(s[:, :8]); states, tcps, rewards, terms, actions, execs = [s], [tcp], [], [], [], []
    for t in range(horizon):
        obs = task.actor_obs(s, goal, tcp); a = policy(obs, t); prev = s[:, L.PREV]; a_exec = adapter.act(a, prev)            # same clip + slew as the simulator path
        s_next = wm(s, a_exec); tcp_next = fk(s_next[:, :8]); r = task.reward_torch(s_next, a_exec, goal, tcp_next); d = wm.termination_prob(s_next, goal)
        states.append(s_next); tcps.append(tcp_next); rewards.append(r); terms.append(d); actions.append(a); execs.append(a_exec); s, tcp = s_next, tcp_next
    return dict(states=states, tcps=tcps, rewards=rewards, terms=terms, actions=actions, executed=execs)
def survival_weighted_return(rewards, terms, v_last, gamma):
    surv = torch.ones_like(rewards[0]); total = torch.zeros_like(rewards[0])
    for t, (r, d) in enumerate(zip(rewards, terms)): total = total + (gamma ** t) * surv * r; surv = surv * (1.0 - d)
    return total + (gamma ** len(rewards)) * surv * v_last, surv
def lambda_returns(rewards, values, continues, gamma, lam):
    """rewards (H,B), values (H+1,B) [V(s_0..s_H)], continues (H,B) in [0,1] (0 = terminal after the step).  Terminal steps do not bootstrap."""
    G = values[-1]; out = []
    for t in reversed(range(rewards.shape[0])): G = rewards[t] + gamma * continues[t] * ((1.0 - lam) * values[t + 1] + lam * G); out.append(G)
    return torch.stack(out[::-1])
def critic_loss_on(traj, critic, task, goal, gamma, lam, epochs, minibatches, optimizer, grad_clip=1.0):
    """fits V to lambda-returns of the (detached) imagined trajectory; returns mean loss"""
    with torch.no_grad():
        S = torch.stack(traj["states"]); T = torch.stack(traj["tcps"]); H1, B = S.shape[0], S.shape[1]; g = goal.unsqueeze(0).expand(H1, -1, -1)
        feats = critic_features(task, S.reshape(H1 * B, -1), g.reshape(H1 * B, -1), T.reshape(H1 * B, -1)); critic.norm.update(feats)
        v = critic(feats).reshape(H1, B); R = torch.stack(traj["rewards"]); C = 1.0 - torch.stack(traj["terms"]); targets = lambda_returns(R, v, C, gamma, lam)
        X = feats.reshape(H1, B, -1)[:-1].reshape((H1 - 1) * B, -1); Y = targets.reshape(-1)
    losses = []; n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=X.device)
        for chunk in perm.chunk(minibatches):
            loss = ((critic(X[chunk]) - Y[chunk]) ** 2).mean(); optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss): continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip); optimizer.step(); losses.append(float(loss.detach()))
    return sum(losses) / max(len(losses), 1)
