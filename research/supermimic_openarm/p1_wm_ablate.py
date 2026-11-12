"""CONTROLLED world-model ablations for P1 pushing (CPU, bounded).  The main training architecture is NOT changed: every variant is a separate arm trained on the SAME dataset (p1_wm_eval.py output) with the SAME number of updates; one factor differs per arm.
Arms (train):  base_s0/base_s1/base_s2 (seed-to-seed spread + ensemble members) | K1, K16 (multi-step window; base K=8) | bal (contact-balanced window sampling, 50% of the mass on windows that contain contact) | gate (contact-gated residual box dynamics) | gate_bal (gate + bal).
Gate: box_next = box + g * (wm_box_next - box), g = max(sigmoid(k (r0 - |tcp - box|)), sigmoid(kv (|v_box| - v0))): the box can change only if the pusher is close or the box is already moving; r0, k, v0, kv are learned.  In free space with a still box the prediction is EXACTLY stationary.
Evaluation (eval): (1) free-space ghost drift and box prediction by regime (h = 1, 16), (2) matched-trajectory imagined-vs-real return gap by regime, (3) optimism under optimisation with mitigations on the same 24 start states: baseline | short horizon | trust region | ensemble-disagreement penalty | combined,
(4) whether optimised actions stay inside the training action distribution (saturation, deviation, Mahalanobis, ensemble disagreement), (5) gradient fidelity: WM gradient vs finite differences of the REAL MuJoCo return (correlation, sign agreement, and 'does ascending the WM gradient raise the real return').
Usage (repo root):
  python -m research.supermimic_openarm.p1_wm_ablate train --arm base_s0 --dataset outputs/p1_validation/wm_eval_dataset_v1.npz --models outputs/p1_validation/ablate_models
  python -m research.supermimic_openarm.p1_wm_ablate eval  --dataset ... --models ... --out outputs/p1_validation/wm_ablation.json
Limit: proxy world models on proxy data (not the pilot's); one training seed per non-base arm, so differences smaller than the base seed spread are NOT evidence."""
import argparse, math, sys, time
from pathlib import Path
import numpy as np
ARMS = dict(base_s0=dict(K=8, bal=False, gate=False, seed=0), base_s1=dict(K=8, bal=False, gate=False, seed=1), base_s2=dict(K=8, bal=False, gate=False, seed=2), K1=dict(K=1, bal=False, gate=False, seed=0), K16=dict(K=16, bal=False, gate=False, seed=0),
            bal=dict(K=8, bal=True, gate=False, seed=0), gate=dict(K=8, bal=False, gate=True, seed=0), gate_bal=dict(K=8, bal=True, gate=True, seed=0))
REG = ("free", "first_contact", "sustained", "contact_loss")
def load(dataset):
    from .p1_wm_eval import labels
    z = np.load(dataset, allow_pickle=True); tr, va = list(z["train"]), list(z["val"])
    for e in tr + va: e["lab"] = labels(e)
    return tr, va
def build(arm, dev, fk):
    import torch, torch.nn as nn
    from .models import WorldModel
    from .envs import layout as L
    wm = WorldModel(2, 256, 3, True, False).to(dev)
    if not ARMS[arm]["gate"]: return wm
    class Gated(nn.Module):
        def __init__(s, base, fk_):
            super().__init__(); s.wm = base; s.fk = fk_; s.r0 = nn.Parameter(torch.tensor(0.10)); s.logk = nn.Parameter(torch.tensor(math.log(60.0))); s.v0 = nn.Parameter(torch.tensor(0.01)); s.logkv = nn.Parameter(torch.tensor(math.log(300.0)))
        def gate(s, st):
            d = (s.fk(st[:, :8]) - st[:, L.BOX_POS]).norm(dim=-1); gp = torch.sigmoid(s.logk.exp() * (s.r0 - d)); gv = torch.sigmoid(s.logkv.exp() * (st[:, L.BOX_VEL].norm(dim=-1) - s.v0)); return torch.maximum(gp, gv)[:, None]
        def forward(s, st, a):
            out = s.wm(st, a); b0 = L.BOX_POS.start; g = s.gate(st); return torch.cat([out[:, :b0], st[:, b0:] + g * (out[:, b0:] - st[:, b0:])], -1)
        def state_loss(s, p, t): return s.wm.state_loss(p, t)
        def update_norms(s, a, b, c): return s.wm.update_norms(a, b, c)
    return Gated(wm, fk).to(dev)
def train_arm(arm, tr, dev, fk, updates, models_dir):
    import torch
    o = ARMS[arm]; K = o["K"]; torch.manual_seed(o["seed"]); rng = np.random.default_rng(o["seed"]); wm = build(arm, dev, fk)
    S_all = torch.as_tensor(np.concatenate([e["S"][:-1] for e in tr]), device=dev); A_all = torch.as_tensor(np.concatenate([e["A"] for e in tr]), device=dev); N_all = torch.as_tensor(np.concatenate([e["S"][1:] for e in tr]), device=dev); wm.update_norms(S_all, A_all, N_all)
    starts = [(i, t) for i, e in enumerate(tr) for t in range(0, len(e["A"]) - K + 1)]; contact = np.array([bool(np.isin(tr[i]["lab"][t:t + K], ["first_contact", "sustained", "contact_loss"]).any()) for i, t in starts])
    p = np.full(len(starts), 1.0 / len(starts))
    if o["bal"]: p = np.where(contact, 0.5 / max(contact.sum(), 1), 0.5 / max((~contact).sum(), 1))
    opt = torch.optim.Adam(wm.parameters(), 3e-4); losses = []; t0 = time.time()
    for u in range(updates):
        idx = rng.choice(len(starts), 256, p=p); Sb = torch.as_tensor(np.stack([tr[starts[i][0]]["S"][starts[i][1]: starts[i][1] + K + 1] for i in idx]), device=dev); Ab = torch.as_tensor(np.stack([tr[starts[i][0]]["A"][starts[i][1]: starts[i][1] + K] for i in idx]), device=dev); pred = Sb[:, 0]; loss = 0
        for j in range(K): pred = wm(pred, Ab[:, j]); loss = loss + wm.state_loss(pred, Sb[:, j + 1])
        loss = loss / K; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0); opt.step(); losses.append(float(loss.detach()))
    Path(models_dir).mkdir(parents=True, exist_ok=True); path = Path(models_dir) / f"{arm}.pt"
    if path.exists(): sys.exit(f"refusing to overwrite {path}")
    torch.save(dict(arm=arm, opts=o, state=wm.state_dict(), updates=updates, first_loss=float(np.mean(losses[:20])), last_loss=float(np.mean(losses[-20:])), seconds=time.time() - t0, contact_window_fraction=float(contact.mean())), path)
    print(f"{arm}: loss {np.mean(losses[:20]):.3f} -> {np.mean(losses[-20:]):.3f} in {time.time() - t0:.0f}s (contact windows {contact.mean():.3f})", flush=True)
def boot(x, n=1000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed); m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(n)] if len(x) > 1 else [float("nan")]; return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["train", "eval"]); ap.add_argument("--arm"); ap.add_argument("--dataset", required=True); ap.add_argument("--models", required=True); ap.add_argument("--out"); ap.add_argument("--updates", type=int, default=6000); ap.add_argument("--states", type=int, default=24)
    ap.add_argument("--opt-steps", type=int, default=40); ap.add_argument("--seed", type=int, default=7); a = ap.parse_args(argv)
    import torch
    from .config import Config
    from .trainer import build_fk_and_adapter
    from .envs import layout as L
    dev = torch.device("cpu"); cfg = Config(); cfg.env.task = "pushing"; fk, adapter, _ = build_fk_and_adapter(cfg, dev); tr, va = load(a.dataset)
    if a.mode == "train":
        torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "1"))); train_arm(a.arm, tr, dev, fk, a.updates, a.models); return 0
    out = Path(a.out)
    if out.exists(): sys.exit(f"refusing to overwrite {out}")
    from .utils import atomic_write_json, repro_metadata
    from .p1_wm_eval import window_label, quat_angle_deg
    from .envs.openarm_env import OpenArmTaskEnv
    from .envs.tasks import make_task
    from .baselines.push_expert import MAX_STEP
    import mujoco
    torch.set_num_threads(4); wms = {}; meta = {}
    for arm in ARMS:
        p = Path(a.models) / f"{arm}.pt"
        if p.exists(): ck = torch.load(p, map_location=dev, weights_only=False); w = build(arm, dev, fk); w.load_state_dict(ck["state"]); w.eval(); [q.requires_grad_(False) for q in w.parameters()]; wms[arm] = w; meta[arm] = dict(first_loss=ck["first_loss"], last_loss=ck["last_loss"], seconds=ck["seconds"], opts=ck["opts"], contact_window_fraction=ck["contact_window_fraction"])
    env = OpenArmTaskEnv("pushing"); task = make_task("pushing"); H = 16; rng = np.random.default_rng(a.seed); rep = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), arms=meta, limit="proxy world models on the proxy dataset; one seed per non-base arm; see base seed spread before reading any difference")
    TERMS = lambda bp, tcp, g: None
    def reward(s, a_ex, goal, tcp): return task.reward_torch(s, a_ex, goal, tcp)
    # ---------------------------------------------------------------------------------------------------------------- (1)+(2) prediction and matched returns for every arm
    def predict_eval(w):
        P = {(h, r): dict(wm=[], st=[], cv=[], ang=[], vel=[]) for h in (1, 16) for r in REG}; MR = []; drift = []
        with torch.no_grad():
            for e in va:
                S = torch.as_tensor(e["S"]); A = torch.as_tensor(e["A"]); T = len(e["A"]); goal = torch.as_tensor(e["goal"])[None]
                for t0 in range(0, T - H + 1, 4):
                    cur = S[t0:t0 + 1]; preds = []
                    for k in range(H): cur = w(cur, A[t0 + k:t0 + k + 1]); preds.append(cur[0])
                    for h in (1, 16):
                        r = window_label(e["lab"], t0, h)
                        if r in ("mixed", "other"): continue
                        p_ = preds[h - 1].numpy(); tg = e["S"][t0 + h]; s0 = e["S"][t0]; x = P[(h, r)]
                        x["wm"].append(1000 * np.linalg.norm(p_[L.BOX_POS] - tg[L.BOX_POS])); x["st"].append(1000 * np.linalg.norm(s0[L.BOX_POS] - tg[L.BOX_POS])); x["cv"].append(1000 * np.linalg.norm(s0[L.BOX_POS] + s0[L.BOX_VEL] * 0.05 * h - tg[L.BOX_POS]))
                        x["ang"].append(float(quat_angle_deg(p_[L.BOX_QUAT] / np.linalg.norm(p_[L.BOX_QUAT]), tg[L.BOX_QUAT]))); x["vel"].append(float(np.linalg.norm(p_[L.BOX_VEL])))
                    pj = torch.stack(preds); tcp_i = fk(pj[:, :8]); a_ex = A[t0:t0 + H]; imag = float(reward(pj, a_ex, goal.expand(H, -1), tcp_i).sum()); real = float(e["R"][t0:t0 + H].sum()); MR.append((window_label(e["lab"], t0, H), imag - real, imag, real))
        tab = {}
        for (h, r), x in P.items():
            if not x["wm"]: continue
            wm_, st_, cv_ = np.array(x["wm"]), np.array(x["st"]), np.array(x["cv"]); tab[f"h{h}_{r}"] = dict(n=len(wm_), wm_mm=float(wm_.mean()), stationary_mm=float(st_.mean()), const_vel_mm=float(cv_.mean()), wm_minus_stationary_ci95=[float((wm_ - st_).mean())] + boot(wm_ - st_), ratio_vs_stationary=float(wm_.mean() / max(st_.mean(), 1e-9)), orientation_deg=float(np.mean(x["ang"])), predicted_box_speed_m_s=float(np.mean(x["vel"])))
        gaps = {r: [m[1] for m in MR if m[0] == r] for r in REG}; return dict(box=tab, matched_gap={r: dict(n=len(v), mean=float(np.mean(v)), ci95=boot(v)) for r, v in gaps.items() if v}, matched_gap_all=dict(mean=float(np.mean([m[1] for m in MR])), ci95=boot([m[1] for m in MR]), corr=float(np.corrcoef([m[2] for m in MR], [m[3] for m in MR])[0, 1])))
    rep["prediction"] = {arm: predict_eval(w) for arm, w in wms.items()}; print("prediction done", flush=True)
    # ---------------------------------------------------------------------------------------------------------------- free-space drift diagnosis (base_s0): where does the ghost motion come from?
    w = wms["base_s0"]; rows = []
    with torch.no_grad():
        for e in va[:60]:
            lab = e["lab"]; S = torch.as_tensor(e["S"]); A = torch.as_tensor(e["A"])
            for t in range(0, len(A) - 1, 2):
                if lab[t] != "free": continue
                n = w(S[t:t + 1], A[t:t + 1])[0]; dbox = (n[L.BOX_POS] - S[t][L.BOX_POS]).numpy(); darm = np.abs(e["S"][t + 1][:7] - e["S"][t][:7]).sum(); d_tcp = float(np.linalg.norm(e["TCP"][t] - e["S"][t][L.BOX_POS])); rows.append((float(np.linalg.norm(dbox)) * 1000, float(darm), d_tcp, float(np.linalg.norm(e["S"][t][L.BOX_VEL])), float(np.linalg.norm((n[L.BOX_VEL]).numpy())), dbox[:2].tolist(), float(np.linalg.norm((S[t + 1][L.BOX_POS] - S[t][L.BOX_POS]).numpy())) * 1000))
    R = np.array([[r[0], r[1], r[2], r[3], r[4], r[6]] for r in rows]); near = R[:, 2] < 0.15
    def c(x, y): return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
    rep["free_space_drift_diagnosis"] = dict(n=len(R), one_step_predicted_box_displacement_mm=dict(mean=float(R[:, 0].mean()), p90=float(np.percentile(R[:, 0], 90))), actual_one_step_displacement_mm=dict(mean=float(R[:, 5].mean())), corr_with_arm_joint_motion=c(R[:, 0], R[:, 1]), corr_with_tcp_box_distance=c(R[:, 0], R[:, 2]),
        mean_pred_disp_mm_when_tcp_within_15cm=float(R[near, 0].mean()) if near.any() else None, mean_pred_disp_mm_when_tcp_beyond_15cm=float(R[~near, 0].mean()) if (~near).any() else None, predicted_box_speed_m_s=float(R[:, 4].mean()), actual_box_speed_m_s=float(R[:, 3].mean()),
        direction_consistency=float(np.linalg.norm(np.mean([r[5] for r in rows], 0)) / max(np.mean([np.linalg.norm(r[5]) for r in rows]), 1e-12)), note="direction_consistency ~1 means a systematic bias (all steps drift the same way); ~0 means zero-mean noise (random walk).  Correlation with arm motion indicates leakage from arm dynamics into the object head.")
    print("drift diagnosis done", flush=True)
    # ---------------------------------------------------------------------------------------------------------------- simulator helpers
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    def snap(): st = np.empty(mujoco.mj_stateSize(env.m, spec)); mujoco.mj_getState(env.m, env.d, st, spec); return st, (env.prev_ctrl.copy(), env.t, env.held, env.ever_success, env.box_max_rise, env.tcp_prev.copy())
    def restore(sn): mujoco.mj_setState(env.m, env.d, sn[0], spec); env.prev_ctrl, env.t, env.held, env.ever_success, env.box_max_rise, env.tcp_prev = sn[1][0].copy(), sn[1][1], sn[1][2], sn[1][3], sn[1][4], sn[1][5].copy(); mujoco.mj_forward(env.m, env.d)
    def real_return(acts):                                                                                       # acts (H,8) normalised delta actions; returns (return, executed list)
        tot = 0.0; ex = []
        for k in range(acts.shape[0]):
            o = env.step(env.state_vec()[L.PREV].astype(np.float64) + acts[k] * MAX_STEP); tot += o["reward"]; ex.append(o["executed"].copy())
        return tot, ex
    cand = [(e, t0) for e in va if e["mode"] in ("expert", "poke") for t0 in range(10, len(e["A"]) - H - 4, 12)]; sel = [cand[i] for i in rng.choice(len(cand), size=min(a.states, len(cand)), replace=False)]
    allA = np.concatenate([(e["Tg"] - e["S"][:-1][:, L.PREV]) / MAX_STEP for e in tr]).clip(-1, 1); mu = allA.mean(0); cov = np.cov(allA[:, :7].T) + 1e-6 * np.eye(7); icov = np.linalg.inv(cov)
    def mahal(x): d = x[:, :7] - mu[:7]; return np.sqrt(np.einsum("ij,jk,ik->i", d, icov, d))
    thr99 = float(np.percentile(mahal(allA[::7]), 99))
    # ---------------------------------------------------------------------------------------------------------------- (3)+(4) optimisation exploitation with mitigations (all on the same states, paired)
    ens_names = [n for n in ("base_s0", "base_s1", "base_s2") if n in wms]; ens = [wms[n] for n in ens_names]
    def imag_ret(ws, s0, goal, acts, h, lam):
        rets = []; dis = 0.0
        S = [s0.clone() for _ in ws]; tot = [0.0 for _ in ws]
        for k in range(h):
            nxt = []
            for mi, w in enumerate(ws):
                a_ex = adapter.act(acts[k][None], S[mi][:, L.PREV]); s1 = w(S[mi], a_ex); tot[mi] = tot[mi] + reward(s1, a_ex, goal, fk(s1[:, :8]))[0]; nxt.append(s1)
            S = nxt
            if lam > 0 and len(ws) > 1: dis = dis + torch.stack([s[0, L.BOX_POS] for s in S]).std(0).norm() * 100.0                               # 1 cm member disagreement = 1 reward unit
        mean = sum(tot) / len(ws); return mean - lam * dis, dict(imag=float(sum(tot) / len(ws)), dis=float(dis))
    def optimise(ws, s0, goal, a0, h, steps, lr, radius, lam):
        a_p = torch.as_tensor(a0[:h].copy(), dtype=torch.float32).requires_grad_(True); base = torch.as_tensor(a0[:h].copy(), dtype=torch.float32); op = torch.optim.Adam([a_p], lr)
        for _ in range(steps):
            op.zero_grad(); J, _ = imag_ret(ws, s0, goal, a_p.clamp(-1, 1), h, lam); (-J).backward(); op.step()
            if radius is not None:
                with torch.no_grad(): a_p.copy_(base + (a_p - base).clamp(-radius, radius))
        return a_p.detach().clamp(-1, 1).numpy()
    VARIANTS = dict(baseline=dict(h=16, steps=a.opt_steps, lr=0.05, radius=None, lam=0.0, ens=False), short_h8=dict(h=8, steps=a.opt_steps, lr=0.05, radius=None, lam=0.0, ens=False), short_h4=dict(h=4, steps=a.opt_steps, lr=0.05, radius=None, lam=0.0, ens=False),
                    trust_0p1=dict(h=16, steps=a.opt_steps, lr=0.05, radius=0.1, lam=0.0, ens=False), conservative_10steps=dict(h=16, steps=10, lr=0.02, radius=0.1, lam=0.0, ens=False), ens_pen_1=dict(h=16, steps=a.opt_steps, lr=0.05, radius=None, lam=1.0, ens=True), ens_pen_3=dict(h=16, steps=a.opt_steps, lr=0.05, radius=None, lam=3.0, ens=True),
                    combined=dict(h=8, steps=20, lr=0.03, radius=0.1, lam=1.0, ens=True))
    opt_arms = [x for x in ("base_s0", "gate", "bal", "gate_bal") if x in wms]; opt = {}; fd = {}
    for n_i, (e, t0) in enumerate(sel):
        env.reset(e["seed"]); [env.step(tg) for tg in e["Tg"][:t0]]; sn = snap(); s0 = torch.as_tensor(env.state_vec())[None]; goal = torch.as_tensor(e["goal"])[None]; prev0 = e["S"][t0][L.PREV]; a_rec = np.clip((e["Tg"][t0:t0 + H] - np.vstack([prev0[None], e["A"][t0:t0 + H - 1]])) / MAX_STEP, -1, 1).astype(np.float32)
        restore(sn); real_rec, _ = real_return(a_rec)
        for arm in opt_arms:
            for vn, v in VARIANTS.items():
                if v["ens"] and (arm != "base_s0" or len(ens) < 3): continue
                ws = ens if v["ens"] else [wms[arm]]; h = v["h"]; a_opt = a_rec.copy(); a_opt[:h] = optimise(ws, s0, goal, a_rec, h, v["steps"], v["lr"], v["radius"], v["lam"])
                with torch.no_grad(): imag_o, info = imag_ret(ws, s0, goal, torch.as_tensor(a_opt), H, 0.0); imag_r, _ = imag_ret(ws, s0, goal, torch.as_tensor(a_rec), H, 0.0)
                restore(sn); real_o, ex = real_return(a_opt)
                with torch.no_grad(): m_o = mahal(a_opt[:h]); m_r = mahal(a_rec[:h]); dis_o = imag_ret(ens, s0, goal, torch.as_tensor(a_opt), H, 1.0)[1]["dis"] if len(ens) > 1 else None; dis_r = imag_ret(ens, s0, goal, torch.as_tensor(a_rec), H, 1.0)[1]["dis"] if len(ens) > 1 else None
                opt.setdefault((arm, vn), []).append(dict(gap_rec=float(imag_r) - real_rec, gap_opt=float(imag_o) - real_o, real_gain=real_o - real_rec, imag_gain=float(imag_o) - float(imag_r), sat=float((np.abs(a_opt[:h, :7]) > 0.95).mean()), dev=float(np.abs(a_opt[:h] - a_rec[:h]).mean()), maha_frac_out=float((m_o > thr99).mean()), maha_frac_out_recorded=float((m_r > thr99).mean()), ens_dis_opt=dis_o, ens_dis_rec=dis_r))
        # ------------------------------------------------------------------------------------------------------------ (5) gradient fidelity vs REAL finite differences
        for arm in [x for x in ("base_s0", "base_s1", "gate", "bal", "gate_bal", "K16") if x in wms]:
            w = wms[arm]; ap_ = torch.as_tensor(a_rec.copy()).requires_grad_(True); J, _ = imag_ret([w], s0, goal, ap_, H, 0.0); J.backward(); g = ap_.grad[:4, :7].numpy().ravel(); gn = np.linalg.norm(g)
            dirs = [rng.normal(size=28) for _ in range(6)]; dirs = [d / np.linalg.norm(d) for d in dirs]
            if gn > 1e-9: dirs += [g / gn, -g / gn]
            else: dirs += [rng.normal(size=28) / 5.3, rng.normal(size=28) / 5.3]                                                              # degenerate gradient: keep 8 rows per state
            eps = 0.5; rows_ = []
            for d_ in dirs:
                ap2 = a_rec.copy(); ap2[:4, :7] = np.clip(ap2[:4, :7] + eps * d_.reshape(4, 7), -1, 1); restore(sn); r2, _ = real_return(ap2)
                with torch.no_grad(): j2 = float(imag_ret([w], s0, goal, torch.as_tensor(ap2), H, 0.0)[0]) - float(J)
                rows_.append((r2 - real_rec, j2, float(eps * (g @ d_))))
            fd.setdefault(arm, []).extend(rows_)
        if n_i % 6 == 0: print(f"state {n_i + 1}/{len(sel)}", flush=True)
    summ = {}
    for (arm, vn), rows_ in opt.items(): summ.setdefault(arm, {})[vn] = dict(n=len(rows_), mean_gap_recorded=float(np.mean([r["gap_rec"] for r in rows_])), mean_gap_optimised=float(np.mean([r["gap_opt"] for r in rows_])), gap_optimised_ci95=boot([r["gap_opt"] for r in rows_]), mean_imagined_gain=float(np.mean([r["imag_gain"] for r in rows_])), mean_real_gain=float(np.mean([r["real_gain"] for r in rows_])), real_gain_ci95=boot([r["real_gain"] for r in rows_]),
        fraction_real_worse=float(np.mean([r["real_gain"] < 0 for r in rows_])), action_saturation_fraction=float(np.mean([r["sat"] for r in rows_])), mean_abs_deviation_from_recorded=float(np.mean([r["dev"] for r in rows_])), frac_steps_outside_training_action_99pct_mahalanobis=float(np.mean([r["maha_frac_out"] for r in rows_])), frac_outside_for_recorded_actions=float(np.mean([r["maha_frac_out_recorded"] for r in rows_])),
        ensemble_disagreement_optimised=(float(np.mean([r["ens_dis_opt"] for r in rows_])) if rows_[0]["ens_dis_opt"] is not None else None), ensemble_disagreement_recorded=(float(np.mean([r["ens_dis_rec"] for r in rows_])) if rows_[0]["ens_dis_rec"] is not None else None))
    rep["optimisation_with_mitigations"] = dict(states=len(sel), horizon_executed=H, mahalanobis_99pct_threshold_training_actions=thr99, summary=summ, note="gap = imagined - real return over 16 steps of the SAME actions from the SAME snapshot; real_gain = real return of optimised actions minus real return of the recorded actions (positive = optimisation helped in the simulator)")
    fds = {}
    for arm, rows_ in fd.items():
        r = np.array(rows_); big = np.abs(r[:, 0]) > 0.05; asc = [];
        n_dirs = 8; k_ = len(r) // n_dirs
        for i in range(k_):
            blk = r[i * n_dirs:(i + 1) * n_dirs]
            if blk.shape[0] == n_dirs: asc.append(float(blk[6, 0] > blk[7, 0]))                                       # real return along +grad vs along -grad
        fds[arm] = dict(n=len(r), n_measurable=int(big.sum()), corr_real_vs_wm_nonlinear=float(np.corrcoef(r[:, 0], r[:, 1])[0, 1]) if r[:, 0].std() > 0 and r[:, 1].std() > 0 else None, corr_real_vs_wm_linear_gradient=float(np.corrcoef(r[:, 0], r[:, 2])[0, 1]) if r[:, 0].std() > 0 and r[:, 2].std() > 0 else None,
                        sign_agreement_nonlinear_when_measurable=float(np.mean(np.sign(r[big, 0]) == np.sign(r[big, 1]))) if big.any() else None, sign_agreement_linear_when_measurable=float(np.mean(np.sign(r[big, 0]) == np.sign(r[big, 2]))) if big.any() else None,
                        frac_states_where_real_return_higher_along_plus_grad_than_minus_grad=float(np.mean(asc)) if asc else None, mean_real_delta_along_plus_grad=float(np.mean([r[i * n_dirs + 6, 0] for i in range(k_)])) if k_ else None, mean_wm_predicted_delta_along_plus_grad=float(np.mean([r[i * n_dirs + 6, 1] for i in range(k_)])) if k_ else None)
    rep["gradient_fidelity_vs_real_finite_difference"] = dict(perturbation="4-step, 7-joint action perturbation of norm 0.5 (normalised slew units): 6 random directions + the WM gradient direction and its negative", per_arm=fds)
    rep["seconds_total"] = None; rep["metadata"] = repro_metadata(dict(no_policy_training=True)); atomic_write_json(out, rep); print("written", out); return 0
if __name__ == "__main__": sys.exit(main())
