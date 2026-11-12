"""P1 (pushing) WORLD-MODEL READINESS diagnostics.  Bounded, CPU-friendly, read-only with respect to every existing run (writes one NEW json; refuses to overwrite).  Uses the SuperMimic WorldModel / actor / imagination code unchanged.
IMPORTANT LIMIT: the world model here is fitted on SCRIPTED-pusher + noisy + random-exploration data (a proxy for the early replay buffer), not on P1 on-policy data; the real readiness evidence is this same script run against the P1 pilot's own checkpoint (--checkpoint) once it exists.
Diagnostics: (1) executed-action alignment, (2) one-step and multi-step object prediction by contact regime (free-space / first-contact / sustained-contact) against persistence and constant-velocity baselines, (3) actor gradients through the imagined dynamics
(non-zero, finite, BPTT depth, sensitivity of predicted box motion to the actions), (4) analytic-reward error of imagined states vs the simulator, (5) model exploitation: optimise an open-loop action sequence by gradient ascent on the IMAGINED return, then execute it in the REAL simulator from the same snapshot and compare.
Usage (repo root):  python -m research.supermimic_openarm.p1_wm_diagnostics --out outputs/p1_validation/wm_diagnostics.json [--episodes 60 --wm-updates 1500] [--checkpoint RUN/checkpoints/latest.pt]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
def parse(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--episodes", type=int, default=60); ap.add_argument("--val-episodes", type=int, default=20); ap.add_argument("--wm-updates", type=int, default=1500); ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--checkpoint", help="use the world model of an existing P1 run instead of fitting one (no parameter is updated)"); ap.add_argument("--horizon", type=int, default=16); ap.add_argument("--exploit-states", type=int, default=12); ap.add_argument("--exploit-steps", type=int, default=60)
    ap.add_argument("--seed-base", type=int, default=23_200_000); ap.add_argument("--device", default="cpu"); return ap.parse_args(argv)
def collect(env, expert, seeds, rng, MAX_STEP, L, mode):
    """transitions from ONE env: expert with action noise (mode 'expert'), expert with strong noise + bursts ('noisy'), uniform random ('random').  Returns per-episode dicts with arrays and contact flags."""
    eps = []
    for sd in seeds:
        env.reset(sd); S = [env.state_vec()]; A = []; C = []; R = []; TCP = [env.d.xpos[env.tcp_id].astype(np.float32).copy()]; done = False; burst = (int(rng.integers(10, 60)), 6) if mode == "noisy" else None; t = 0
        while not done:
            ae = expert.act(env); a = ae.copy()
            if mode == "expert": a[:7] += rng.normal(0, 0.05, 7)
            elif mode == "noisy": a[:7] += rng.normal(0, 0.25, 7) + (rng.normal(0, 0.9, 7) if burst[0] <= t < burst[0] + burst[1] else 0)
            else: a = rng.uniform(-1, 1, 8)
            a = np.clip(a, -1, 1); o = env.step(env.state_vec()[L.PREV] + a * MAX_STEP); S.append(o["state"].copy()); A.append(o["executed"].copy()); C.append(o["contact"].copy()); R.append(o["reward"]); TCP.append(o["tcp"].copy()); done = o["terminated"] or o["truncated"]; t += 1
        eps.append(dict(seed=sd, S=np.stack(S), A=np.stack(A), C=np.stack(C), R=np.array(R, np.float32), TCP=np.stack(TCP), goal=env.goal.copy(), mode=mode))
    return eps
def main(argv=None):
    a = parse(argv); out = Path(a.out)
    if out.exists(): sys.exit(f"refusing to overwrite {out}")
    import torch
    from .config import Config, from_dict
    from .utils import assert_seeds_allowed, repro_metadata, atomic_write_json
    from .envs.openarm_env import OpenArmTaskEnv
    from .envs import layout as L
    from .envs.tasks import make_task
    from .baselines.push_expert import PushExpert, MAX_STEP
    from .models import WorldModel, Actor, Critic, critic_features
    from .imagination import imagine, survival_weighted_return, set_requires_grad
    from .trainer import build_fk_and_adapter
    import mujoco
    seeds_tr = [a.seed_base + i for i in range(a.episodes)]; seeds_va = [a.seed_base + 50_000 + i for i in range(a.val_episodes)]; assert_seeds_allowed(*seeds_tr, *seeds_va); torch.manual_seed(0); rng = np.random.default_rng(0); dev = torch.device(a.device)
    cfg = Config(); cfg.env.task = "pushing"; cfg.world_model.hidden = 256; cfg.world_model.layers = 3; cfg.actor.hidden = 256; cfg.actor.layers = 3; cfg.system.device = a.device
    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location=dev, weights_only=False); cfg = from_dict(ck["config"]); cfg.system.device = a.device
    env = OpenArmTaskEnv("pushing"); task = make_task("pushing"); fk, adapter, _ = build_fk_and_adapter(cfg, dev); expert = PushExpert(); rep = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), limit="proxy data (scripted pusher / noisy / random); see module docstring" if not a.checkpoint else f"world model of {a.checkpoint}")
    t0 = time.time(); tr = collect(env, expert, seeds_tr[: int(0.5 * a.episodes)], rng, MAX_STEP, L, "expert") + collect(env, expert, seeds_tr[int(0.5 * a.episodes): int(0.8 * a.episodes)], rng, MAX_STEP, L, "noisy") + collect(env, expert, seeds_tr[int(0.8 * a.episodes):], rng, MAX_STEP, L, "random")
    va = collect(env, expert, seeds_va[: a.val_episodes // 2], rng, MAX_STEP, L, "expert") + collect(env, expert, seeds_va[a.val_episodes // 2:], rng, MAX_STEP, L, "noisy"); rep["data"] = dict(train_episodes=len(tr), val_episodes=len(va), train_transitions=int(sum(len(e["A"]) for e in tr)), seconds=round(time.time() - t0, 1), val_seeds_disjoint_from_train=not (set(seeds_tr) & set(seeds_va)))
    # ------------------------------------------------------------------------------------------------------ (1) executed-action alignment
    err = max(float(np.abs(e["S"][1:, L.PREV] - e["A"]).max()) for e in tr + va); al_prev = max(float(np.abs(e["S"][1:, L.Q] - e["S"][:-1, L.Q]).max()) for e in tr[:3]) > 0
    rep["alignment"] = dict(max_abs_err_next_prev_ctrl_vs_recorded_executed_action=err, passed=bool(err < 1e-6), note="state[t+1].prev_ctrl must equal the executed action t (the world model is trained and rolled out on EXECUTED actions)", states_change=al_prev)
    # ------------------------------------------------------------------------------------------------------ world model: fit (or load)
    wm = WorldModel(2, cfg.world_model.hidden, cfg.world_model.layers, True, False).to(dev)
    if a.checkpoint: wm.load_state_dict(ck["wm"]); fit = "loaded"
    else:
        S_all = torch.as_tensor(np.concatenate([e["S"][:-1] for e in tr]), device=dev); A_all = torch.as_tensor(np.concatenate([e["A"] for e in tr]), device=dev); N_all = torch.as_tensor(np.concatenate([e["S"][1:] for e in tr]), device=dev); wm.update_norms(S_all, A_all, N_all); opt = torch.optim.Adam(wm.parameters(), 3e-4); K = a.window; losses = []
        starts = [(i, t) for i, e in enumerate(tr) for t in range(0, len(e["A"]) - K + 1)]
        for u in range(a.wm_updates):
            idx = rng.integers(0, len(starts), 256); Sb = np.stack([tr[starts[i][0]]["S"][starts[i][1]: starts[i][1] + K + 1] for i in idx]); Ab = np.stack([tr[starts[i][0]]["A"][starts[i][1]: starts[i][1] + K] for i in idx]); Sb, Ab = torch.as_tensor(Sb, device=dev), torch.as_tensor(Ab, device=dev); pred = Sb[:, 0]; loss = 0
            for j in range(K): pred = wm(pred, Ab[:, j]); loss = loss + wm.state_loss(pred, Sb[:, j + 1])
            loss = loss / K; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0); opt.step(); losses.append(float(loss.detach()))
        fit = dict(updates=a.wm_updates, first_loss=float(np.mean(losses[:20])), last_loss=float(np.mean(losses[-20:])), seconds=round(time.time() - t0, 1))
    wm.eval(); set_requires_grad(wm, False); rep["world_model_fit"] = fit
    # ------------------------------------------------------------------------------------------------------ (2) prediction error by contact regime
    def regime(C):   # per-step regime label of the transition t -> t+1: 0 free-space, 1 first contact (onset +-2), 2 sustained contact
        c = (C[:, 0] > 0) | (C[:, 2] > 0) | (C[:, 1] > 0); r = np.zeros(len(c), int); r[c] = 2; on = np.where(c & ~np.concatenate([[False], c[:-1]]))[0]
        for t in on: r[max(0, t - 2): t + 3] = 1
        return r
    hs = (1, 5, 10, a.horizon); res = {h: {r: dict(n=0, wm=0.0, persist=0.0, constvel=0.0, vel_wm=0.0, vel_persist=0.0) for r in (0, 1, 2)} for h in hs}; box_mm = []
    with torch.no_grad():
        for e in va:
            S = torch.as_tensor(e["S"], device=dev); A = torch.as_tensor(e["A"], device=dev); reg = regime(e["C"]); T = len(A)
            for t0_ in range(0, T - max(hs), 2):
                cur = S[t0_:t0_ + 1]; kmax = max(hs); preds = []
                for k in range(kmax): cur = wm(cur, A[t0_ + k:t0_ + k + 1]); preds.append(cur)
                for h in hs:
                    p = preds[h - 1][0]; tgt = S[t0_ + h]; r = int(reg[t0_:t0_ + h].max()) if reg[t0_:t0_ + h].max() < 2 else (2 if (reg[t0_:t0_ + h] == 2).sum() >= max(1, h // 2) else 1)
                    bw = (p[L.BOX_POS] - tgt[L.BOX_POS]).norm().item(); bp = (S[t0_][L.BOX_POS] - tgt[L.BOX_POS]).norm().item(); bc = (S[t0_][L.BOX_POS] + S[t0_][L.BOX_VEL] * 0.05 * h - tgt[L.BOX_POS]).norm().item()
                    x = res[h][r]; x["n"] += 1; x["wm"] += bw; x["persist"] += bp; x["constvel"] += bc; x["vel_wm"] += (p[L.BOX_VEL] - tgt[L.BOX_VEL]).norm().item(); x["vel_persist"] += (S[t0_][L.BOX_VEL] - tgt[L.BOX_VEL]).norm().item()
    table = {}
    for h in hs:
        for r, nm in ((0, "free_space"), (1, "first_contact"), (2, "sustained_contact")):
            x = res[h][r]; n = max(x["n"], 1); table[f"h{h}_{nm}"] = dict(n=x["n"], box_pos_err_mm=1000 * x["wm"] / n, persistence_mm=1000 * x["persist"] / n, const_velocity_mm=1000 * x["constvel"] / n, ratio_vs_persistence=(x["wm"] / x["persist"] if x["persist"] > 1e-12 else None), box_vel_err_m_s=x["vel_wm"] / n, vel_persistence_m_s=x["vel_persist"] / n)
    rep["box_prediction_by_regime"] = table; rep["box_prediction_note"] = "box position error (mm) of the open-loop world-model rollout vs the simulator, per horizon h (control steps) and contact regime; ratio < 1 means better than predicting that the box does not move"
    # ------------------------------------------------------------------------------------------------------ (3) actor gradients through the imagined dynamics + sensitivity
    actor = Actor(task.actor_obs_dim, cfg.actor.hidden, cfg.actor.layers).to(dev); critic = Critic(L.STATE_DIM + 2 + 3, 128, 2).to(dev); starts_s = torch.as_tensor(np.stack([e["S"][min(5, len(e["S"]) - 2)] for e in tr[:64]]), device=dev); goals_s = torch.as_tensor(np.stack([e["goal"] for e in tr[:64]]), device=dev)
    for p in actor.net[-1].parameters(): torch.nn.init.normal_(p, std=0.05)                       # the output layer is near-zero by design; make the actor non-trivial for the probe
    set_requires_grad(critic, False); tr_i = imagine(wm, lambda o, t: actor(o), task, fk, adapter, starts_s, goals_s, a.horizon); v = critic(critic_features(task, tr_i["states"][-1], goals_s, tr_i["tcps"][-1])); J, _ = survival_weighted_return(tr_i["rewards"], tr_i["terms"], v, 0.99); actor.zero_grad(); (-J.mean()).backward()
    gn = float(torch.sqrt(sum((p.grad ** 2).sum() for p in actor.parameters() if p.grad is not None))); fin = bool(all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None))
    acts = torch.zeros(len(starts_s), a.horizon, 8, device=dev, requires_grad=True); tr_b = imagine(wm, lambda o, t: acts[:, t], task, fk, adapter, starts_s, goals_s, a.horizon); tr_b["rewards"][-1].sum().backward(); bptt = float(acts.grad[:, 0].abs().mean())
    acts2 = torch.zeros(len(starts_s), a.horizon, 8, device=dev, requires_grad=True); tr_c = imagine(wm, lambda o, t: acts2[:, t], task, fk, adapter, starts_s, goals_s, a.horizon); tr_c["states"][-1][:, L.BOX_POS].sum().backward(); sens_box = acts2.grad.abs().mean(dim=(0, 1)).cpu().numpy()
    rep["actor_gradients"] = dict(grad_norm=gn, finite=fin, bptt_last_reward_to_first_action_mean_abs=bptt, nonzero=bool(gn > 0 and bptt > 0), sensitivity_of_final_predicted_box_position_to_actions_per_joint_mean_abs=[float(x) for x in sens_box], note="in free space the box does not respond to the arm; a flat sensitivity here means no push gradient reaches the actor")
    # ------------------------------------------------------------------------------------------------------ (4) analytic reward error of imagined states, (5) model exploitation
    rerr = []; rcor = []
    with torch.no_grad():
        for e in va:
            S = torch.as_tensor(e["S"], device=dev); A = torch.as_tensor(e["A"], device=dev); g = torch.as_tensor(e["goal"], device=dev)[None]; cur = S[0:1]; pr = []
            for k in range(len(A)): cur = wm(cur, A[k:k + 1]); tcp = fk(cur[:, :8]); pr.append(float(task.reward_torch(cur, A[k:k + 1], g, tcp)))
            rr = e["R"]; rerr.append(float(np.mean(np.abs(np.array(pr) - rr)))); rcor.append(float(np.corrcoef(pr, rr)[0, 1]) if np.std(pr) > 0 and np.std(rr) > 0 else float("nan"))
    rep["reward_prediction"] = dict(open_loop_mean_abs_error_per_step=float(np.mean(rerr)), mean_correlation=float(np.nanmean(rcor)), reward_scale_mean_abs=float(np.mean([np.abs(e["R"]).mean() for e in va])), note="the imagined reward is the analytic task reward evaluated on predicted states (no learned reward head)")
    gaps = []; Hh = a.horizon
    for e in va[:a.exploit_states]:
        t_s = min(30, len(e["A"]) - Hh - 1); env.reset(e["seed"]); [env.step(env.state_vec()[L.PREV] + np.clip(expert.act(env), -1, 1) * MAX_STEP) for _ in range(t_s)]
        spec = mujoco.mjtState.mjSTATE_INTEGRATION; st = np.empty(mujoco.mj_stateSize(env.m, spec)); mujoco.mj_getState(env.m, env.d, st, spec); py = (env.prev_ctrl.copy(), env.t, env.held, env.ever_success, env.box_max_rise, env.tcp_prev.copy()); s0 = torch.as_tensor(env.state_vec(), device=dev)[None]; g = torch.as_tensor(env.goal, device=dev)[None]
        u = torch.zeros(1, Hh, 8, device=dev, requires_grad=True); opt = torch.optim.Adam([u], 0.1)
        for it in range(a.exploit_steps):
            trj = imagine(wm, lambda o, t: torch.tanh(u[:, t]), task, fk, adapter, s0, g, Hh); ret = sum(trj["rewards"]); opt.zero_grad(); (-ret.mean()).backward(); opt.step()
        imag = float(ret.detach()); acts_np = torch.tanh(u).detach()[0].cpu().numpy()
        mujoco.mj_setState(env.m, env.d, st, spec); env.prev_ctrl, env.t, env.held, env.ever_success, env.box_max_rise, env.tcp_prev = py[0].copy(), py[1], py[2], py[3], py[4], py[5].copy(); mujoco.mj_forward(env.m, env.d); real = 0.0
        for k in range(Hh): real += env.step(env.state_vec()[L.PREV] + acts_np[k] * MAX_STEP)["reward"]
        base = 0.0; mujoco.mj_setState(env.m, env.d, st, spec); env.prev_ctrl, env.t, env.held, env.ever_success, env.box_max_rise, env.tcp_prev = py[0].copy(), py[1], py[2], py[3], py[4], py[5].copy(); mujoco.mj_forward(env.m, env.d)
        for k in range(Hh): base += env.step(env.state_vec()[L.PREV] + np.clip(expert.act(env), -1, 1) * MAX_STEP)["reward"]
        gaps.append(dict(seed=e["seed"], imagined_return_optimised=imag, real_return_of_the_same_actions=float(real), expert_real_return_same_horizon=float(base), gap=imag - float(real)))
    G = np.array([x["gap"] for x in gaps]); rep["model_exploitation"] = dict(horizon=Hh, n=len(gaps), mean_gap_imagined_minus_real=float(G.mean()), median_gap=float(np.median(G)), frac_states_imagined_exceeds_real_by_more_than_expert_margin=float(np.mean([x["imagined_return_optimised"] - x["real_return_of_the_same_actions"] > max(1.0, abs(x["expert_real_return_same_horizon"]) * 0.2) for x in gaps])),
        mean_real_return_optimised=float(np.mean([x["real_return_of_the_same_actions"] for x in gaps])), mean_real_return_expert=float(np.mean([x["expert_real_return_same_horizon"] for x in gaps])), episodes=gaps, note="actions optimised by gradient ascent on the imagined return, then replayed in the real simulator from the same snapshot: a large positive gap means the policy optimiser exploits world-model errors")
    rep["seconds_total"] = round(time.time() - t0, 1); rep["metadata"] = repro_metadata(dict(no_training_of_any_existing_run=True, protected_seeds_used=False)); atomic_write_json(out, rep)
    print(json.dumps({k: rep[k] for k in ("alignment", "actor_gradients", "reward_prediction")}, indent=1)[:2500]); print("model_exploitation:", {k: rep["model_exploitation"][k] for k in ("n", "mean_gap_imagined_minus_real", "mean_real_return_optimised", "mean_real_return_expert")}); print("written:", out); return 0
if __name__ == "__main__": sys.exit(main())
