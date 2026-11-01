#!/usr/bin/env python3
"""STAGE 0 DESIGN AUDIT (inference only; NO closed loop, NO training, ACT outputs NEVER executed).
Question: do the 6 frozen xsweep ACT checkpoints (C0 x3 seeds, S1 x3 seeds) give semantically valid, *different* action chunks from ONE observation, and do those candidates differ in the frozen WM3-B + frozen J score?
Candidate sources only: C0-only (3), S1-only (3), mixed (6).  No MC-dropout, no noise.  Seed change (within recipe) is separated from recipe change (cross recipe).
Observations: open-loop scripted-expert states (ACT-training-like distribution), one decision point per phase group (approach/grasp/lift/place) per pose.  Future outcome labels are never used to select candidates.
Hard cap: 0.15 process CPU-hour (incl. loading).  On the cap the run is written as PARTIAL and stops; it never raises the cap or retries.  Atomic JSON writes after every stage / decision point.
Run (ACT venv): python eval/act_candidate_audit.py"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import sys, json, time, resource, tempfile, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "eval"), str(ROOT / "data")): sys.path.insert(0, p)
torch.set_num_threads(1)
OUTD = ROOT / "outputs/closed_loop_design"; OUTD.mkdir(parents=True, exist_ok=True); OUT = OUTD / "act_candidate_audit_stage0.json"
def cpu(): a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN); return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime
T0 = time.time(); C0_ = cpu(); CAP = 0.15 * 3600
class CapReached(Exception): pass
def guard(tag, margin=0.0):
    if cpu() - C0_ + margin > CAP: raise CapReached(tag)
res = dict(protocol="stage0-v1", status="RUNNING", cap_cpu_s=CAP, stages=[], decision_points=[], latency_ms=dict(ckpt_load={}, act_inference=[], adapter_slew=[], wm_scoring=[]), config={})
def save():
    res["cpu_s"] = cpu() - C0_; res["wall_s"] = time.time() - T0
    fd, tmp = tempfile.mkstemp(dir=OUTD, suffix=".tmp"); os.close(fd); Path(tmp).write_text(json.dumps(res, indent=1, default=float)); os.replace(tmp, OUT)      # atomic
def stage(name): res["stages"].append(dict(stage=name, cpu_s=round(cpu() - C0_, 2), wall_s=round(time.time() - T0, 2))); print(f"[stage] {name} cpu={cpu()-C0_:.1f}s", flush=True); save()
CK = {"c0_s0": "e6b_c0_seed0", "c0_s1": "e0_c0_seed1", "c0_s2": "e0_c0_seed2", "s1_s0": "e6b_s1_seed0", "s1_s1": "e0_s1_seed1", "s1_s2": "e0_s1_seed2"}
GROUPS = {"c0_only": ["c0_s0", "c0_s1", "c0_s2"], "s1_only": ["s1_s0", "s1_s1", "s1_s2"], "mixed": list(CK)}
DEFAULT = "c0_s0"          # predetermined default / chunk-0 policy (fixed before looking at any result)
POSES = [(0.35, 0.10, 0.0), (0.40, 0.10, 0.0), (0.45, 0.10, 0.0)]
PHASE_GROUP = {"pre_a": "approach", "pregrasp": "approach", "approach": "approach", "close": "grasp", "grip_hold": "grasp", "lift": "lift", "transport": "lift", "lower": "place", "release": "place"}
THR_ARM, THR_GRIP = 0.02, 0.005          # "genuinely different" thresholds (L-inf over the 20 executed steps): rad / m
TIE_TOLS = [1e-9, 1e-3, 5e-3]
res["config"] = dict(checkpoints=CK, groups=GROUPS, default=DEFAULT, poses=POSES, phase_groups=sorted(set(PHASE_GROUP.values())), thr_arm_rad=THR_ARM, thr_grip_m=THR_GRIP, tie_tols=TIE_TOLS, wm_rows="J at row t+20 (frozen); rank-stability diagnostic uses row t+15 (never used for selection)")
try:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import OBS_IMAGES
    class ChunkPolicy:
        def __init__(self, name):
            path = ROOT / "outputs/checkpoints" / CK[name] / "last"; self.policy = ACTPolicy.from_pretrained(str(path)); self.policy.eval()
            self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=str(path)); self.cams = [k.split(".")[-1] for k in self.policy.config.image_features]
        @torch.no_grad()
        def chunk(self, state, images):
            batch = {"observation.state": torch.as_tensor(state, dtype=torch.float32)[None]}
            for c in self.cams: batch[f"observation.images.{c}"] = torch.as_tensor(images[c]).permute(2, 0, 1).float().div(255.0)[None]
            b = dict(self.pre(batch)); b[OBS_IMAGES] = [b[k] for k in self.policy.config.image_features]
            self.policy.model.eval(); a = self.policy.model(b)[0]          # eval mode only: use_vae False, deterministic
            return torch.stack([torch.as_tensor(self.post(a[:, k]).reshape(-1)) for k in range(a.shape[1])]).cpu().numpy()      # (20,16) un-normalised raw output
    policies = {}
    for k in CK:
        guard(f"before load {k}"); t0 = time.perf_counter(); policies[k] = ChunkPolicy(k); res["latency_ms"]["ckpt_load"][k] = (time.perf_counter() - t0) * 1000
    stage("act checkpoints loaded")
    sys.argv = [sys.argv[0], "none"]
    import wm2_run as R, wm2_models as M, wm3b_model as B, decision_utility_lib as L
    M.logistic_fit = lambda X, y, l2=0.01, iters=200: (np.zeros((X.shape[1], y.shape[1]), np.float32), np.zeros(y.shape[1], np.float32))
    t0 = time.perf_counter(); tr = R.load("train"); nrm = M.Norm(tr); ridge = M.Ridge("cond").fit(nrm, tr); w3 = B.WM3B(ridge, nrm); w3.model = B.GRUOC(); w3.model.load_state_dict(torch.load(ROOT / "outputs/wm3/models/wm3b.pt")); w3.model.eval()
    res["latency_ms"]["wm_load_and_ridge_fit"] = (time.perf_counter() - t0) * 1000; stage("wm3b loaded")
    import mujoco, wm1_sim as S
    from lab_env import LabEnv
    from cameras import POLICY_CAMERAS
    from policies.baseline import ScriptedExpert
    from control.safety_limits import SafetyLimits
    env_p = S.get_env()["env"]; lim = SafetyLimits.from_model(env_p.m); expert = ScriptedExpert(env_p.m, control_hz=20); env_i = LabEnv(cameras=POLICY_CAMERAS); stage("sim ready")
    def slew_chain(raw, prev):
        out = []; nr = ns = 0; p = prev
        for r in raw: c, a, b_ = lim.clip_and_slew(r, p, 0.05); out.append(c); p = c; nr += a; ns += b_
        return np.array(out), nr, ns
    def validity(exe, prev):
        e = np.vstack([prev[None], exe]); d = np.abs(np.diff(e, axis=0)) / 0.05; ok_rate = bool((d <= lim.max_speed[None] + 1e-6).all()); ok_rng = bool(((exe >= lim.lo - 1e-9) & (exe <= lim.hi + 1e-9)).all()); return ok_rate and ok_rng and bool(np.isfinite(exe).all()), ok_rate, ok_rng
    def pair_stats(E, idx):         # E: dict name -> (20,16) ; idx: list of names (left arm only: arm joints 0..6, gripper 7)
        out = dict(arm_linf=[], arm_rms=[], grip_linf=[], per_joint_linf=[])
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a, b_ = E[idx[i]][:, :8], E[idx[j]][:, :8]; d = np.abs(a - b_)
                out["arm_linf"].append(float(d[:, :7].max())); out["arm_rms"].append(float(np.sqrt(((a[:, :7] - b_[:, :7]) ** 2).mean()))); out["grip_linf"].append(float(d[:, 7].max())); out["per_joint_linf"].append(d.max(0).tolist())
        return out
    def n_distinct(E, idx):
        reps = []
        for n in idx:
            if not any(np.abs(E[n][:, :7] - E[r][:, :7]).max() < THR_ARM and np.abs(E[n][:, 7] - E[r][:, 7]).max() < THR_GRIP for r in reps): reps.append(n)
        return len(reps)
    def ranks(x): return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    def spearman(a, b_):
        ra, rb = ranks(a), ranks(b_)
        if ra.std() == 0 or rb.std() == 0: return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])
    def summarize_pairs(P):
        s = lambda v: dict(median=float(np.median(v)), max=float(np.max(v))) if len(v) else None
        return dict(arm_linf=s(P["arm_linf"]), arm_rms=s(P["arm_rms"]), grip_linf=s(P["grip_linf"]), n_pairs=len(P["arm_linf"]))
    done_pts = 0
    for (x, y, yaw) in POSES:
        guard(f"pose {x}", margin=20.0)
        t_sim0 = time.perf_counter(); plan = expert.plan((x, y), yaw); A = np.array(plan["ctrl"], float); Ex = np.zeros_like(A)
        env_p.set_perturbation(); env_p.reset((x, y), yaw); prev = env_p.prev_ctrl.copy()
        for t in range(len(A)): Ex[t], _, _ = lim.clip_and_slew(A[t], prev, 0.05); prev = Ex[t]
        Rr = S.simulate(Ex.astype(np.float32), (x, y), yaw); b, c = S.contact_features(Rr); s16 = Rr["state"]; a8 = Rr["action"][:, :8]
        names_t = np.array(plan["phase_names"])[np.array(plan["phase"])]; times = {}
        for g in ("approach", "grasp", "lift", "place"):          # one decision point per phase group: middle of the first segment of that group, constrained to the valid window
            idxs = [t for t in range(4, len(A) - 21) if PHASE_GROUP.get(str(names_t[t])) == g]
            if idxs: times[g] = int(idxs[len(idxs) // 2])
        # render only at the needed times (no per-step rendering)
        env_i.set_perturbation(); o = env_i.reset((x, y), yaw); obs_at = {}
        for t in range(max(times.values()) + 1):
            if t in times.values(): obs_at[t] = env_i.obs()
            env_i.d.ctrl[:] = Ex[t]; env_i.prev_ctrl = Ex[t].copy()
            for _ in range(env_i.substeps): mujoco.mj_step(env_i.m, env_i.d)
            env_i.t += 1.0 / env_i.hz
        res["latency_ms"].setdefault("expert_sim_and_render_per_pose", []).append((time.perf_counter() - t_sim0) * 1000); stage(f"pose {x} simulated")
        for g, t in times.items():
            guard(f"point {x} {g}", margin=15.0)
            o = obs_at[t]; state = o["state"]; imgs = o["images"]; prev_exec = Ex[t - 1]; raw = {}; exe = {}; cnt = {}; val = {}
            for k, p in policies.items(): t0 = time.perf_counter(); raw[k] = p.chunk(state, imgs); res["latency_ms"]["act_inference"].append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            for k in CK: exe[k], nr, ns = slew_chain(raw[k].astype(float), prev_exec); cnt[k] = (nr, ns); val[k] = validity(exe[k], prev_exec)
            res["latency_ms"]["adapter_slew"].append((time.perf_counter() - t0) * 1000)
            names = list(CK); n = len(names)
            W = dict(S_hist=np.repeat(s16[t - 3:t + 1, :8][None], n, 0).astype(np.float32), B_hist=np.repeat(b[t - 3:t + 1][None], n, 0), C_hist=np.repeat(c[t - 3:t + 1][None], n, 0), A_hist=np.repeat(a8[t - 4:t][None], n, 0).astype(np.float32),
                     A_fut=np.stack([exe[k][:, :8] for k in names]).astype(np.float32), t=np.full(n, t), Y_e=np.zeros((n, 4), np.float32), pose=np.repeat(np.array([[x, y, yaw]], np.float32), n, 0))
            t0 = time.perf_counter(); P = w3.predict(W); bb = b[t][None] + np.cumsum(P["Y_b"], 1); J20 = []; J15 = []; palm = []; dxy = []
            for i in range(n):
                rb = np.vstack([b[t][None], bb[i]]); rc = np.vstack([c[t][None], P["Y_c"][i]]); tm = L.terms(rb, rc); J20.append(L.score(tm)); palm.append(bool(tm["palm"])); dxy.append(tm["dxy"]); J15.append(L.score(L.terms(rb[:16], rc[:16])))
            res["latency_ms"]["wm_scoring"].append((time.perf_counter() - t0) * 1000); J20 = np.array(J20); J15 = np.array(J15)
            rec = dict(pose=[x, y, yaw], t=t, phase_name=str(names_t[t]), phase_group=g, groups={}, n_valid=int(sum(v[0] for v in val.values())), rate_ok=int(sum(v[1] for v in val.values())), range_ok=int(sum(v[2] for v in val.values())),
                       clipped_range_values=int(sum(v[0] for v in cnt.values())), slewed_values=int(sum(v[1] for v in cnt.values())), raw_vs_exec_left_max_abs=float(max(np.abs(exe[k][:, :8] - raw[k][:, :8]).max() for k in names)),
                       J20_by_ckpt={k: float(J20[i]) for i, k in enumerate(names)}, palm_pred_by_ckpt={k: palm[i] for i, k in enumerate(names)}, dxy_mm_by_ckpt={k: float(1000 * dxy[i]) for i, k in enumerate(names)})
            for gn, members in GROUPS.items():
                idx = [names.index(m) for m in members]; Jg = J20[idx]; Jg15 = J15[idx]; d_i = members.index(DEFAULT) if DEFAULT in members else None
                dJ = [abs(Jg[i] - Jg[j]) for i in range(len(idx)) for j in range(i + 1, len(idx))]
                rec["groups"][gn] = dict(raw=summarize_pairs(pair_stats(raw, members)), executed=summarize_pairs(pair_stats(exe, members)), n_distinct_executed=n_distinct(exe, members), n_members=len(members),
                                        J20_spread=float(np.ptp(Jg)), J20_ties_frac={str(t_): float(np.mean([d_ <= t_ for d_ in dJ])) for t_ in TIE_TOLS}, rank_spearman_t20_vs_t15=spearman(Jg, Jg15), argmin_same_t20_t15=bool(np.argmin(Jg) == np.argmin(Jg15)),
                                        default_regret=(float(Jg[d_i] - Jg.min()) if d_i is not None else None), best=members[int(np.argmin(Jg))])
            same = [(a_, b_) for a_ in GROUPS["c0_only"] for b_ in GROUPS["c0_only"] if a_ < b_] + [(a_, b_) for a_ in GROUPS["s1_only"] for b_ in GROUPS["s1_only"] if a_ < b_]
            cross = [(a_, b_) for a_ in GROUPS["c0_only"] for b_ in GROUPS["s1_only"]]
            dd = lambda pairs: float(np.mean([np.abs(exe[a_][:, :7] - exe[b_][:, :7]).max() for a_, b_ in pairs])); dJp = lambda pairs: float(np.mean([abs(J20[names.index(a_)] - J20[names.index(b_)]) for a_, b_ in pairs]))
            rec["seed_vs_recipe"] = dict(within_recipe_diff_seed_arm_linf=dd(same), cross_recipe_arm_linf=dd(cross), within_recipe_mean_abs_dJ=dJp(same), cross_recipe_mean_abs_dJ=dJp(cross))
            res["decision_points"].append(rec); done_pts += 1; stage(f"point {x} {g} done")
    res["status"] = "COMPLETE"
except CapReached as e:
    res["status"] = "PARTIAL"; res["stop_reason"] = f"hard CPU cap (0.15 CPU-h) reached at: {e}"
except Exception as e:
    import traceback; res["status"] = "BLOCKED"; res["stop_reason"] = f"{type(e).__name__}: {e}"; res["traceback"] = traceback.format_exc()
finally:
    res["completed_decision_points"] = len(res["decision_points"]); res["expected_decision_points"] = len(POSES) * 4
    lat = res["latency_ms"]; med = lambda v: float(np.median(v)) if v else None; res["latency_summary_ms"] = dict(ckpt_load_median=med(list(lat["ckpt_load"].values())), act_inference_median=med(lat["act_inference"]), act_inference_p95=(float(np.percentile(lat["act_inference"], 95)) if lat["act_inference"] else None),
                                                                           adapter_slew_per_6cands_median=med(lat["adapter_slew"]), wm_scoring_per_6cands_median=med(lat["wm_scoring"]))
    save(); print(json.dumps({k: res[k] for k in ("status", "stop_reason", "cpu_s", "wall_s", "completed_decision_points", "expected_decision_points", "latency_summary_ms") if k in res}, indent=1), flush=True)
