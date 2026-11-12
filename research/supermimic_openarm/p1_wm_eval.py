"""P1 (pushing) WORLD-MODEL re-evaluation on a regime-labelled diagnostic dataset.  Bounded, CPU only, read-only for every existing run; writes NEW files (refuses to overwrite).  The world model, actor and imagination code are used UNCHANGED.
LIMIT (stated in every output): unless --checkpoint is given, the world model is fitted here on SCRIPTED-pusher + noisy + fast-poke + random data (a proxy for an early replay buffer).  It is not the P1 pilot's world model; run this script with --checkpoint on the pilot's checkpoint for the real readiness evidence.
Contents
  (1) dataset: transition regimes from the finger / hand contact flags: free (no contact for >= 3 states), first_contact (no contact -> contact), sustained (contact for >= 3 consecutive states), contact_loss (contact -> no contact).
      Multi-step windows are labelled by the highest-priority event inside them: first_contact > contact_loss > sustained (>= half of the steps) > free; everything else = mixed (reported, not tabulated).
  (2) one-step and multi-step box prediction (position, orientation, velocity) vs STATIONARY (box stays) and CONSTANT-VELOCITY (box_pos + v*dt*h) baselines, paired bootstrap CI of the error difference per regime.
  (3) imagined vs actual return on MATCHED trajectories: the same start state and the same executed actions are rolled in the world model and were executed in the simulator; gap = imagined - actual; by regime and by reward term.
  (4) optimism under optimisation: the action sequence is optimised by gradient ascent on the IMAGINED return (warm-started from the recorded actions), then executed in the real simulator from the same snapshot.
  (5) action-response fidelity: a 4-step action perturbation is applied in the simulator and in the world model; agreement of the predicted box displacement (correlation, sign agreement); plus actor-gradient finiteness through the imagined dynamics.
Usage (repo root):  python -m research.supermimic_openarm.p1_wm_eval --out outputs/p1_validation/wm_eval.json [--train-episodes 360 --val-episodes 120 --wm-updates 6000 --dataset-out outputs/p1_validation/wm_eval_dataset.npz]"""
import argparse, sys, time
from pathlib import Path
import numpy as np
MODES = ("expert", "noisy", "poke", "random"); HS = (1, 4, 8, 16); REG = ("free", "first_contact", "sustained", "contact_loss")
def parse(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--dataset-out"); ap.add_argument("--dataset", help="reuse a dataset npz written by an earlier run"); ap.add_argument("--train-episodes", type=int, default=360); ap.add_argument("--val-episodes", type=int, default=120)
    ap.add_argument("--wm-updates", type=int, default=6000); ap.add_argument("--window", type=int, default=8); ap.add_argument("--horizon", type=int, default=16); ap.add_argument("--seed-base", type=int, default=25_000_000); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--opt-states", type=int, default=24); ap.add_argument("--opt-steps", type=int, default=40); ap.add_argument("--fd-states", type=int, default=24); ap.add_argument("--checkpoint"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--actor-episodes", type=int, default=0, help="with --checkpoint: also roll the checkpoint's ACTOR in the simulator and compare imagined vs real return on ITS OWN trajectories"); ap.add_argument("--actor-seed-base", type=int, default=27_300_000); ap.add_argument("--grad-states", type=int, default=0, help="states for the return-gradient check (WM gradient vs real return along +grad / -grad)"); return ap.parse_args(argv)
# ------------------------------------------------------------------------------------------------------------------------------ data collection (worker processes)
def _collect(args):
    seed, mode, nseed = args
    from .envs.openarm_env import OpenArmTaskEnv
    from .envs import layout as L
    from .baselines.push_expert import PushExpert, MAX_STEP
    global _E
    if "_E" not in globals(): _E = OpenArmTaskEnv("pushing", None, 150, 20)
    env = _E; rng = np.random.default_rng(nseed); ex = PushExpert(v_push=0.40, v_descend=0.40) if mode == "poke" else PushExpert(); env.reset(seed); d = env.d
    S = [env.state_vec()]; A, Tg, NF, HF, R, TCP = [], [], [], [], [], [env.tcp()]; burst = (int(rng.integers(10, 70)), int(rng.integers(3, 9))) if mode == "noisy" else None; pause = (int(rng.integers(30, 90)), int(rng.integers(4, 10))) if mode == "expert" and rng.random() < 0.5 else None; done = False; t = 0
    while not done:
        a = ex.act(env).astype(np.float64)
        if mode == "expert": a[:7] += rng.normal(0, 0.05, 7)
        elif mode == "noisy": a[:7] += rng.normal(0, 0.25, 7) + (rng.normal(0, 0.9, 7) if burst[0] <= t < burst[0] + burst[1] else 0)
        elif mode == "poke": a[:7] += rng.normal(0, 0.10, 7)
        else: a = rng.uniform(-1, 1, 8)
        if pause and pause[0] <= t < pause[0] + pause[1]: a[:7] = 0.0                                                                  # stop while (possibly) in contact: produces contact-loss / re-contact events
        a = np.clip(a, -1, 1); tgt = env.state_vec()[L.PREV].astype(np.float64) + a * MAX_STEP; o = env.step(tgt); S.append(o["state"].copy()); A.append(o["executed"].copy()); Tg.append(tgt); NF.append(o["info"]["finger_contacts"]); HF.append(o["info"]["hand_force_N"]); R.append(o["reward"]); TCP.append(o["tcp"].copy()); done = o["terminated"] or o["truncated"]; t += 1
    inf = env.episode_info; return dict(seed=seed, mode=mode, S=np.stack(S), A=np.stack(A), Tg=np.stack(Tg), NF=np.array(NF), HF=np.array(HF, np.float32), R=np.array(R, np.float32), TCP=np.stack(TCP), goal=env.goal.copy(), mass=inf["mass_kg"], mu=0.8 * inf["friction_scale"])
def labels(e):
    """per-transition regime labels (strings) of transition t: S[t] -> S[t+1]"""
    c0 = np.concatenate([[False], (e["NF"] > 0) | (e["HF"] > 1e-6)]); k5 = np.ones(5, int)                                         # raw contact flag is sampled at the end of each 50 ms step and flickers while pushing:
    c = (np.convolve((np.convolve(c0.astype(int), k5, "same") > 0).astype(int), k5, "same") == 5) | c0                              # morphological closing: gaps of <= 4 steps inside a contact episode are filled
    T = len(e["A"]); lab = []; run = np.zeros(T + 1, int)
    for i in range(1, T + 1): run[i] = run[i - 1] + 1 if c[i] else 0
    for t in range(T):
        b, af = c[t], c[t + 1]
        if not b and af: lab.append("first_contact")
        elif b and not af: lab.append("contact_loss")
        elif b and af and run[t] >= 2: lab.append("sustained")
        elif not c[max(0, t - 3): t + 2].any(): lab.append("free")
        else: lab.append("other")
    return np.array(lab)
def window_label(lab, t0, h):
    w = lab[t0:t0 + h]
    if (w == "first_contact").any(): return "first_contact"
    if (w == "contact_loss").any(): return "contact_loss"
    if (w == "sustained").sum() >= max(1, h // 2): return "sustained"
    if (w == "free").all(): return "free"
    return "mixed"
def boot_ci(x, n=1000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2: return [float("nan"), float("nan")]
    r = np.random.default_rng(seed); m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(n)]; return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
def quat_angle_deg(q1, q2):
    d = np.abs(np.sum(q1 * q2, -1)).clip(0, 1); return np.degrees(2 * np.arccos(d))
def main(argv=None):
    a = parse(argv); out = Path(a.out)
    if out.exists(): sys.exit(f"refusing to overwrite {out}")
    if a.dataset_out and Path(a.dataset_out).exists(): sys.exit(f"refusing to overwrite {a.dataset_out}")
    import torch, multiprocessing as mp
    from .config import Config, from_dict
    from .utils import assert_seeds_allowed, repro_metadata, atomic_write_json, sha256_file
    from .envs.openarm_env import OpenArmTaskEnv
    from .envs import layout as L
    from .envs.tasks import make_task
    from .baselines.push_expert import MAX_STEP
    from .models import Actor
    from .p1_variants import build_world_model, load_actor
    from .imagination import imagine, set_requires_grad
    from .trainer import build_fk_and_adapter
    import mujoco
    torch.manual_seed(0); rng = np.random.default_rng(0); dev = torch.device(a.device); t_start = time.time(); rep = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), limit="proxy world model fitted on scripted-pusher / noisy / poke / random data" if not a.checkpoint else f"world model of {a.checkpoint} (no parameter updated)")
    cfg = Config(); cfg.env.task = "pushing"; cfg.world_model.hidden = 256; cfg.world_model.layers = 3; cfg.actor.hidden = 256; cfg.actor.layers = 3; cfg.system.device = a.device
    if a.checkpoint: ck = torch.load(a.checkpoint, map_location=dev, weights_only=False); cfg = from_dict(ck["config"]); cfg.system.device = a.device
    # ------------------------------------------------------------------------------------------------------------------------ (1) dataset
    def split_modes(n, fr): k = [int(round(n * f)) for f in fr]; k[0] += n - sum(k); return dict(zip(MODES, k))
    def make_jobs(base, n):
        jobs = []; i = 0
        for mode, k in split_modes(n, (0.45, 0.25, 0.20, 0.10)).items():
            for _ in range(k): jobs.append((base + i, mode, 1000 + base + i)); i += 1
        return jobs
    if a.dataset: rep["dataset_sha256"] = sha256_file(a.dataset)
    if a.checkpoint: rep["checkpoint"] = dict(name=Path(a.checkpoint).name, sha256=sha256_file(a.checkpoint), iteration=ck.get("iteration"), variant=ck.get("variant", "b0"))
    if a.dataset:
        z = np.load(a.dataset, allow_pickle=True); tr, va = list(z["train"]), list(z["val"])
    else:
        jt, jv = make_jobs(a.seed_base, a.train_episodes), make_jobs(a.seed_base + 100_000, a.val_episodes); assert_seeds_allowed(jt[0][0], jt[-1][0], jv[0][0], jv[-1][0])
        with mp.get_context("fork").Pool(a.workers) as pool: tr = pool.map(_collect, jt, chunksize=2); va = pool.map(_collect, jv, chunksize=2)
        if a.dataset_out: np.savez_compressed(a.dataset_out, train=np.array(tr, dtype=object), val=np.array(va, dtype=object), allow_pickle=True)
    for e in tr + va: e["lab"] = labels(e)
    def counts(eps):
        c = {r: 0 for r in REG + ("other",)}
        for e in eps:
            for r in e["lab"]: c[r] += 1
        return c
    rep["dataset"] = dict(train_episodes=len(tr), val_episodes=len(va), train_transitions=int(sum(len(e["A"]) for e in tr)), val_transitions=int(sum(len(e["A"]) for e in va)), train_regime_transitions=counts(tr), val_regime_transitions=counts(va),
                          modes={m: sum(e["mode"] == m for e in tr) for m in MODES}, val_seeds_disjoint_from_train=not ({e["seed"] for e in tr} & {e["seed"] for e in va}), seconds=round(time.time() - t_start, 1), dataset_file=a.dataset_out)
    err = max(float(np.abs(e["S"][1:, L.PREV] - e["A"]).max()) for e in tr + va); rep["alignment"] = dict(max_abs_err_next_prev_ctrl_vs_recorded_executed_action=err, passed=bool(err < 1e-6))
    # ------------------------------------------------------------------------------------------------------------------------ world model
    env = OpenArmTaskEnv("pushing"); task = make_task("pushing"); fk, adapter, _ = build_fk_and_adapter(cfg, dev); variant = ck.get("variant", "b0") if a.checkpoint else "b0"; wm = build_world_model(variant, 2, cfg.world_model, fk, dev, **{k: v for k, v in (ck.get("variant_args") or {}).items() if k.startswith("gate_")} if a.checkpoint else {})
    if a.checkpoint: wm.load_state_dict(ck["wm"]); fit = "loaded"
    else:
        S_all = torch.as_tensor(np.concatenate([e["S"][:-1] for e in tr]), device=dev); A_all = torch.as_tensor(np.concatenate([e["A"] for e in tr]), device=dev); N_all = torch.as_tensor(np.concatenate([e["S"][1:] for e in tr]), device=dev); wm.update_norms(S_all, A_all, N_all)
        opt = torch.optim.Adam(wm.parameters(), 3e-4); K = a.window; starts = [(i, t) for i, e in enumerate(tr) for t in range(0, len(e["A"]) - K + 1)]; losses = []; t1 = time.time()
        for u in range(a.wm_updates):
            idx = rng.integers(0, len(starts), 256); Sb = torch.as_tensor(np.stack([tr[starts[i][0]]["S"][starts[i][1]: starts[i][1] + K + 1] for i in idx]), device=dev); Ab = torch.as_tensor(np.stack([tr[starts[i][0]]["A"][starts[i][1]: starts[i][1] + K] for i in idx]), device=dev); pred = Sb[:, 0]; loss = 0
            for j in range(K): pred = wm(pred, Ab[:, j]); loss = loss + wm.state_loss(pred, Sb[:, j + 1])
            loss = loss / K; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0); opt.step(); losses.append(float(loss.detach()))
        fit = dict(updates=a.wm_updates, window=K, first_loss=float(np.mean(losses[:20])), last_loss=float(np.mean(losses[-20:])), seconds=round(time.time() - t1, 1))
    wm.eval(); set_requires_grad(wm, False); rep["world_model_fit"] = fit; print("world model ready", fit, flush=True)
    # ------------------------------------------------------------------------------------------------------------------------ (2) box prediction by regime vs baselines  +  (3) matched imagined-vs-actual returns
    kmax = max(HS); H = a.horizon; P = {(h, r): dict(wm=[], st=[], cv=[], wm_ang=[], st_ang=[], wm_v=[], st_v=[], cv_v=[]) for h in HS for r in REG}; mixed = {h: 0 for h in HS}; MR = []   # matched returns
    TERMS = ("dist", "tcp", "bonus", "lift")
    def terms_of(bp, tcp, goal):
        d = np.linalg.norm(bp[..., :2] - goal, axis=-1); dt = np.linalg.norm(tcp - bp, axis=-1); lift = np.maximum(bp[..., 2] - 0.3805 - 0.01, 0) / 0.01
        return np.stack([-3 * d, -0.5 * dt, 2 * np.exp(-(d / 0.04) ** 2), -0.1 * lift], -1)
    with torch.no_grad():
        for e in va:
            S = torch.as_tensor(e["S"], device=dev); A = torch.as_tensor(e["A"], device=dev); T = len(e["A"]); goal = torch.as_tensor(e["goal"], device=dev)[None]
            for t0 in range(0, T - kmax + 1, 2):
                cur = S[t0:t0 + 1]; preds = []
                for k in range(kmax): cur = wm(cur, A[t0 + k:t0 + k + 1]); preds.append(cur[0])
                for h in HS:
                    r = window_label(e["lab"], t0, h)
                    if r == "mixed" or r == "other": mixed[h] += 1; continue
                    p = preds[h - 1].cpu().numpy(); tg = e["S"][t0 + h]; s0 = e["S"][t0]; x = P[(h, r)]
                    x["wm"].append(1000 * np.linalg.norm(p[L.BOX_POS] - tg[L.BOX_POS])); x["st"].append(1000 * np.linalg.norm(s0[L.BOX_POS] - tg[L.BOX_POS])); x["cv"].append(1000 * np.linalg.norm(s0[L.BOX_POS] + s0[L.BOX_VEL] * 0.05 * h - tg[L.BOX_POS]))
                    x["wm_ang"].append(float(quat_angle_deg(p[L.BOX_QUAT] / np.linalg.norm(p[L.BOX_QUAT]), tg[L.BOX_QUAT]))); x["st_ang"].append(float(quat_angle_deg(s0[L.BOX_QUAT], tg[L.BOX_QUAT])))
                    x["wm_v"].append(float(np.linalg.norm(p[L.BOX_VEL] - tg[L.BOX_VEL]))); x["st_v"].append(float(np.linalg.norm(tg[L.BOX_VEL]))); x["cv_v"].append(float(np.linalg.norm(s0[L.BOX_VEL] - tg[L.BOX_VEL])))
                # matched return over H steps (same start state, same executed actions)
                pj = torch.stack(preds[:H]).cpu().numpy(); tcp_i = fk(torch.as_tensor(pj, device=dev)[:, :8]).cpu().numpy(); real_s = e["S"][t0 + 1:t0 + H + 1]; real_tcp = e["TCP"][t0 + 1:t0 + H + 1]; g = e["goal"]
                ti = terms_of(pj[:, L.BOX_POS], tcp_i, g); tr_ = terms_of(real_s[:, L.BOX_POS], real_tcp, g); imag = float(ti.sum()); real = float(e["R"][t0:t0 + H].sum()); ch = float(np.abs(tr_.sum(-1) - e["R"][t0:t0 + H]).max())
                MR.append(dict(mode=e["mode"], regime=window_label(e["lab"], t0, H), imag=imag, real=real, gap=imag - real, term_gap=(ti.sum(0) - tr_.sum(0)).tolist(), term_check_err=ch, box_err_end_mm=1000 * float(np.linalg.norm(pj[-1, L.BOX_POS] - real_s[-1, L.BOX_POS]))))
    table = {}
    for (h, r), x in P.items():
        n = len(x["wm"]); key = f"h{h}_{r}"
        if n == 0: table[key] = dict(n=0, note="no windows of this regime in the validation set"); continue
        wm_, st_, cv_ = map(np.array, (x["wm"], x["st"], x["cv"])); table[key] = dict(n=n, wm_mm=dict(mean=float(wm_.mean()), median=float(np.median(wm_))), stationary_mm=dict(mean=float(st_.mean()), median=float(np.median(st_))), constant_velocity_mm=dict(mean=float(cv_.mean()), median=float(np.median(cv_))),
            wm_minus_stationary_mm_mean_ci95=[float((wm_ - st_).mean())] + boot_ci(wm_ - st_), wm_minus_constvel_mm_mean_ci95=[float((wm_ - cv_).mean())] + boot_ci(wm_ - cv_), wm_better_than_stationary_frac=float((wm_ < st_).mean()), wm_better_than_constvel_frac=float((wm_ < cv_).mean()),
            box_orientation_err_deg=dict(wm=float(np.mean(x["wm_ang"])), stationary=float(np.mean(x["st_ang"]))), box_velocity_err_m_s=dict(wm=float(np.mean(x["wm_v"])), zero_velocity=float(np.mean(x["st_v"])), constant_velocity=float(np.mean(x["cv_v"]))))
    rep["box_prediction"] = dict(table=table, mixed_windows_excluded=mixed, note="box position error (mm) of the open-loop world-model rollout from the true start state, same executed actions as the simulator; negative wm_minus_* means the world model is closer to the truth; CI = paired bootstrap of the mean difference")
    G = np.array([m["gap"] for m in MR]); IM = np.array([m["imag"] for m in MR]); RE = np.array([m["real"] for m in MR])
    def summ(sel):
        g = np.array([m["gap"] for m in sel]); im = np.array([m["imag"] for m in sel]); re = np.array([m["real"] for m in sel])
        if len(g) < 3: return dict(n=len(g))
        return dict(n=len(g), mean_gap=float(g.mean()), mean_gap_ci95=boot_ci(g), mean_abs_gap=float(np.abs(g).mean()), optimistic_fraction=float((g > 0).mean()), real_return_mean=float(re.mean()), imagined_return_mean=float(im.mean()), corr_imag_real=float(np.corrcoef(im, re)[0, 1]) if im.std() > 0 and re.std() > 0 else None,
                    term_gap_mean={k: float(np.mean([m["term_gap"][i] for m in sel])) for i, k in enumerate(TERMS)}, mean_box_err_end_mm=float(np.mean([m["box_err_end_mm"] for m in sel])))
    rep["matched_returns"] = dict(horizon=H, windows=len(MR), overall=summ(MR), by_regime={r: summ([m for m in MR if m["regime"] == r]) for r in REG + ("mixed",)}, by_data_source={md: summ([m for m in MR if m["mode"] == md]) for md in MODES}, reward_decomposition_consistency_max_abs_err=float(max(m["term_check_err"] for m in MR)),
                              note="gap = imagined - actual return of the SAME action sequence from the SAME start state (positive = optimistic). Reward terms recomputed independently from the recorded simulator states reproduce the simulator reward (consistency error above).")
    print("matched returns done", rep["matched_returns"]["overall"], flush=True)
    # ------------------------------------------------------------------------------------------------------------------------ (4) optimism under optimisation   (5) action-response fidelity
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    def snap(): st = np.empty(mujoco.mj_stateSize(env.m, spec)); mujoco.mj_getState(env.m, env.d, st, spec); return st, (env.prev_ctrl.copy(), env.t, env.held, env.ever_success, env.box_max_rise, env.tcp_prev.copy())
    def restore(sn): mujoco.mj_setState(env.m, env.d, sn[0], spec); env.prev_ctrl, env.t, env.held, env.ever_success, env.box_max_rise, env.tcp_prev = sn[1][0].copy(), sn[1][1], sn[1][2], sn[1][3], sn[1][4], sn[1][5].copy(); mujoco.mj_forward(env.m, env.d)
    def execute(targets): return [env.step(t) for t in targets]
    cand = [(e, t0) for e in va if e["mode"] in ("expert", "poke") for t0 in range(10, len(e["A"]) - H - 4, 12)]; sel = [cand[i] for i in rng.choice(len(cand), size=min(max(a.opt_states, a.fd_states, a.grad_states), len(cand)), replace=False)]; opt_rows = []; fd_rows = []; gnorm = []
    for n_i, (e, t0) in enumerate(sel):
        env.reset(e["seed"]); [env.step(tg) for tg in e["Tg"][:t0]]; sn = snap(); s0 = torch.as_tensor(env.state_vec(), device=dev)[None]; goal = torch.as_tensor(e["goal"], device=dev)[None]; prev0 = e["S"][t0][L.PREV]
        a_rec = np.clip((e["Tg"][t0:t0 + H] - np.vstack([prev0[None], e["A"][t0:t0 + H - 1]])) / MAX_STEP, -1, 1)
        # recorded sequence: imagined vs actual (executed again in the simulator from the snapshot)
        restore(sn); outs = execute(e["Tg"][t0:t0 + H]); real_rec = float(sum(o["reward"] for o in outs)); bp_real = np.stack([o["state"][L.BOX_POS] for o in outs])
        def imag_return(z, grad):
            s = s0; tot = 0.0
            for k in range(H):
                act = torch.tanh(z[k])[None]; a_ex = adapter.act(act, s[:, L.PREV]); s = wm(s, a_ex); tot = tot + task.reward_torch(s, a_ex, goal, fk(s[:, :8]))[0]
            return tot
        z0 = torch.as_tensor(np.arctanh(np.clip(a_rec, -0.97, 0.97)), dtype=torch.float32, device=dev)
        with torch.no_grad(): imag_rec = float(imag_return(z0, False))
        if n_i < a.opt_states:
            z = z0.clone().requires_grad_(True); op = torch.optim.Adam([z], 0.05)
            for _ in range(a.opt_steps): op.zero_grad(); (-imag_return(z, True)).backward(); op.step()
            with torch.no_grad(): imag_opt = float(imag_return(z, False)); act_opt = torch.tanh(z).numpy(); s_ = torch.as_tensor(env.state_vec(), device=dev)
            restore(sn); tgts = []; real_opt = 0.0
            for k in range(H): tg = adapter.target_from_action(torch.as_tensor(act_opt[k], dtype=torch.float32)[None], torch.as_tensor(env.state_vec()[L.PREV])[None])[0].numpy().astype(np.float64); o = env.step(tg); real_opt += o["reward"]
            opt_rows.append(dict(mode=e["mode"], seed=e["seed"], t0=t0, imag_recorded=imag_rec, real_recorded=real_rec, gap_recorded=imag_rec - real_rec, imag_optimised=imag_opt, real_optimised=real_opt, gap_optimised=imag_opt - real_opt, action_change_mean_abs=float(np.abs(act_opt - a_rec).mean())))
        if n_i < a.fd_states:                                                                                                                  # 4-step perturbation response, simulator vs world model
            u = rng.normal(size=7); u /= np.linalg.norm(u); res = {}
            for nm, dl in (("base", 0.0), ("pert", 0.6)):
                restore(sn); ex = []; st = s0.clone()
                for k in range(H):
                    pk = env.state_vec()[L.PREV]; act = a_rec[k].copy()
                    if k < 4: act[:7] = np.clip(act[:7] + dl * u, -1, 1)
                    o = env.step(pk.astype(np.float64) + act * MAX_STEP); ex.append(o["executed"].copy())
                with torch.no_grad():
                    cur = s0.clone()
                    for k in range(H): cur = wm(cur, torch.as_tensor(ex[k], device=dev)[None])
                res[nm] = (env.state_vec()[L.BOX_POS].copy(), cur[0, L.BOX_POS].cpu().numpy().copy(), o["tcp"].copy())
            d_sim = res["pert"][0] - res["base"][0]; d_wm = res["pert"][1] - res["base"][1]; fd_rows.append(dict(d_sim=d_sim.tolist(), d_wm=d_wm.tolist(), tcp_shift_mm=1000 * float(np.linalg.norm(res["pert"][2] - res["base"][2]))))
        if n_i < 8:                                                                                                                            # actor gradient through imagined dynamics (non-trivial actor)
            actor = Actor(task.actor_obs_dim, cfg.actor.hidden, cfg.actor.layers).to(dev)
            for p in actor.net[-1].parameters(): torch.nn.init.normal_(p, std=0.05)
            tri = imagine(wm, lambda o, t: actor(o), task, fk, adapter, s0, goal, H); (-sum(tri["rewards"]).sum()).backward(); gnorm.append(float(torch.sqrt(sum((p.grad ** 2).sum() for p in actor.parameters() if p.grad is not None))))
    O = opt_rows
    if O:
        rep["optimism_under_optimisation"] = dict(n=len(O), horizon=H, mean_gap_recorded=float(np.mean([r["gap_recorded"] for r in O])), mean_gap_optimised=float(np.mean([r["gap_optimised"] for r in O])), mean_gap_optimised_ci95=boot_ci([r["gap_optimised"] for r in O]), mean_real_recorded=float(np.mean([r["real_recorded"] for r in O])), mean_real_optimised=float(np.mean([r["real_optimised"] for r in O])),
            mean_imagined_gain=float(np.mean([r["imag_optimised"] - r["imag_recorded"] for r in O])), mean_real_gain=float(np.mean([r["real_optimised"] - r["real_recorded"] for r in O])), fraction_real_worse_after_optimisation=float(np.mean([r["real_optimised"] < r["real_recorded"] for r in O])), mean_action_change=float(np.mean([r["action_change_mean_abs"] for r in O])), rows=O,
            note="optimised = 40 Adam steps on the imagined return of the 16-step action sequence (warm start = recorded actions); real = the same actions executed in the simulator from the same snapshot")
    if fd_rows:
        ds = np.array([r["d_sim"] for r in fd_rows]); dw = np.array([r["d_wm"] for r in fd_rows]); big = np.linalg.norm(ds, axis=1) > 5e-4                                              # responses > 0.5 mm
        rep["action_response_fidelity"] = dict(n=len(fd_rows), n_with_measurable_sim_response=int(big.sum()), corr_all_components=float(np.corrcoef(ds.ravel(), dw.ravel())[0, 1]) if ds.std() > 0 and dw.std() > 0 else None, sign_agreement_when_measurable=float(np.mean(np.sign(ds[big]) == np.sign(dw[big]))) if big.any() else None,
            mean_sim_response_mm=float(1000 * np.linalg.norm(ds, axis=1).mean()), mean_wm_response_mm=float(1000 * np.linalg.norm(dw, axis=1).mean()), actor_gradient_norms=gnorm, actor_gradients_finite=bool(np.all(np.isfinite(gnorm)) and len(gnorm) > 0), actor_gradients_nonzero=bool(all(g > 0 for g in gnorm)),
            note="4-step perturbation of the arm action (0.6 of the normalised slew step along a random direction); response = change of the box position after 16 steps; the world model is fed the actions the simulator actually executed")
    if a.grad_states > 0:                                                                                                                        # (6) does ascending the WM gradient raise the REAL return?  (gate quantity)
        res6 = []
        for (e, t0) in sel[:a.grad_states]:
            env.reset(e["seed"]); [env.step(tg) for tg in e["Tg"][:t0]]; sn = snap(); s0 = torch.as_tensor(env.state_vec(), device=dev)[None]; goal = torch.as_tensor(e["goal"], device=dev)[None]; prev0 = e["S"][t0][L.PREV]
            a_rec = np.clip((e["Tg"][t0:t0 + H] - np.vstack([prev0[None], e["A"][t0:t0 + H - 1]])) / MAX_STEP, -1, 1).astype(np.float32); ap_ = torch.as_tensor(a_rec.copy(), device=dev).requires_grad_(True); st = s0; tot = 0.0
            for k in range(H): a_ex = adapter.act(ap_[k][None], st[:, L.PREV]); st = wm(st, a_ex); tot = tot + task.reward_torch(st, a_ex, goal, fk(st[:, :8]))[0]
            tot.backward(); g = ap_.grad[:4, :7].cpu().numpy().ravel(); gn = np.linalg.norm(g)
            if gn < 1e-9: continue
            r = {}
            for nm, sg in (("base", 0.0), ("plus", 1.0), ("minus", -1.0)):
                a2 = a_rec.copy(); a2[:4, :7] = np.clip(a2[:4, :7] + sg * 0.5 * (g / gn).reshape(4, 7), -1, 1)
                restore(sn); tt = 0.0
                for k in range(H): o = env.step(env.state_vec()[L.PREV].astype(np.float64) + a2[k] * MAX_STEP); tt += o["reward"]
                r[nm] = tt
            res6.append(dict(real_plus_minus_base=r["plus"] - r["base"], real_minus_minus_base=r["minus"] - r["base"], plus_better_than_minus=bool(r["plus"] > r["minus"]), grad_norm=float(gn)))
        if res6: rep["return_gradient_check"] = dict(n=len(res6), fraction_real_return_higher_along_plus_gradient_than_minus=float(np.mean([x["plus_better_than_minus"] for x in res6])), mean_real_delta_along_plus=float(np.mean([x["real_plus_minus_base"] for x in res6])), mean_real_delta_along_minus=float(np.mean([x["real_minus_minus_base"] for x in res6])), ci95_plus_minus_minus=boot_ci([x["real_plus_minus_base"] - x["real_minus_minus_base"] for x in res6]),
                                                      note="perturbation = 0.5 (normalised slew units, norm over the first 4 steps x 7 joints) along +/- the world-model gradient of the 16-step imagined return; real = same actions executed in the simulator from the same snapshot; 0.5 is chance for the fraction")
    if a.checkpoint and a.actor_episodes > 0:                                                                                                      # (7) matched imagined-vs-real return on the ACTOR'S OWN trajectories
        actor_m, adapter_a = load_actor(cfg, ck, dev, task); rows7 = []; n_term = n_trunc = 0
        for i in range(a.actor_episodes):
            sd = a.actor_seed_base + i; assert_seeds_allowed(sd); s, g_, tcp = env.reset(sd); S = [s.copy()]; A_ = []; R_ = []; TCP_ = [tcp.copy()]; done = False
            while not done:
                with torch.no_grad(): act = actor_m(task.actor_obs(torch.as_tensor(s, device=dev)[None], torch.as_tensor(g_, device=dev)[None], torch.as_tensor(tcp, device=dev)[None])).float(); tgt = adapter_a.target_from_action(act, torch.as_tensor(s[L.PREV], device=dev)[None])[0].cpu().numpy().astype(np.float64)
                o = env.step(tgt); s, tcp = o["state"], o["tcp"]; S.append(s.copy()); A_.append(o["executed"].copy()); R_.append(o["reward"]); TCP_.append(tcp.copy()); done = o["terminated"] or o["truncated"]
            n_term += int(o["terminated"]); n_trunc += int(o["truncated"]); S = np.stack(S); A_ = np.stack(A_); R_ = np.array(R_); TCP_ = np.stack(TCP_)
            with torch.no_grad():
                for t0 in range(0, len(A_) - H + 1, 4):
                    cur = torch.as_tensor(S[t0:t0 + 1], device=dev); gl = torch.as_tensor(g_, device=dev)[None]; tot = 0.0
                    for k in range(H): cur = wm(cur, torch.as_tensor(A_[t0 + k:t0 + k + 1], device=dev)); tot += float(task.reward_torch(cur, torch.as_tensor(A_[t0 + k:t0 + k + 1], device=dev), gl, fk(cur[:, :8]))[0])
                    rows7.append(tot - float(R_[t0:t0 + H].sum()))
        if rows7: rep["matched_returns_actor"] = dict(episodes=a.actor_episodes, episodes_terminated=n_term, episodes_truncated=n_trunc, windows=len(rows7), mean_gap=float(np.mean(rows7)), mean_gap_ci95=boot_ci(rows7), mean_abs_gap=float(np.mean(np.abs(rows7))), optimistic_fraction=float(np.mean(np.array(rows7) > 0)), horizon=H, note="imagined - real return over 16 steps, on trajectories generated by the checkpoint's own actor (held-out seeds, never used for training); only windows of 16 full steps inside an episode are used, so a terminated (failed) episode contributes its steps up to the termination and nothing is bootstrapped past it")
    rep["seconds_total"] = round(time.time() - t_start, 1); rep["metadata"] = repro_metadata(dict(no_training_of_policy=True)); atomic_write_json(out, rep); print("written", out); return 0
if __name__ == "__main__": sys.exit(main())
