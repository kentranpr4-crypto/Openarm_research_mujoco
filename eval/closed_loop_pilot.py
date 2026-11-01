#!/usr/bin/env python3
"""STAGE 1 closed-loop ACT + WM3-B pilot: runner library + worker + orchestrator.   EXPLORATORY SIMULATION PILOT (DEVELOPMENT evidence; privileged-state oracle observer; NOT Sim2Real).
Arms (same 24 registered poses, same LabEnv/safety limits, horizon 200 steps = 10 chunks of 20 executed actuator targets):
  A0  ACT c0_seed0 alone (one episode per pose).
  A1  chunk 1 = c0_seed0; from chunk 2 on, uniformly random pick among the 3 frozen C0 checkpoints, RNG = default_rng([20261101, 8001, pose_idx, repeat]) (3 repeats per pose).
  A2  chunk 1 = c0_seed0; from chunk 2 on, argmin frozen J (WM3-B, executed sequences, row t+20), deterministic tie-break sha1(pose|vid|tie) at tol 1e-9 (one episode per pose).
J and its constants live in eval/decision_utility_lib.py (frozen).  ACT sees only state + images.  The WM sees the privileged oracle observer (box pose, contacts) - hence an upper bound with an oracle observer.
Subcommands (ACT venv):  freeze | run
No training, no checkpoint/dataset modification.  Hard process-CPU cap for ALL of Stage 1 (tests + pilot): 0.90 CPU-h (outputs/closed_loop_pilot/cpu_ledger.json)."""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import sys, json, time, resource, hashlib, tempfile, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "eval"), str(ROOT / "data")): sys.path.insert(0, p)
OUT = ROOT / "outputs/closed_loop_pilot"; FREEZE = OUT / "protocol_freeze.json"; LEDGER = OUT / "cpu_ledger.json"; XCFG = ROOT / "outputs/xsweep/xsweep_config.json"
CAP_TOTAL_S = 0.90 * 3600; HORIZON = 200; CHUNK = 20; DT = 0.05; TOL_TIE = 1e-9
CK = {"c0_s0": "e6b_c0_seed0", "c0_s1": "e0_c0_seed1", "c0_s2": "e0_c0_seed2"}; CK_NAMES = list(CK); DEFAULT = "c0_s0"
XSWEEP_C0S0 = ROOT / "outputs/xsweep/eval/c0_seed0/act/sweep"
RNG_KEY = [20261101, 8001]; N_REPEATS_A1 = 3
def cpu():
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN); return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime
class CapReached(Exception): pass
def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp"); os.close(fd); Path(tmp).write_text(json.dumps(obj, indent=1, default=float)); os.replace(tmp, path)
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ledger_used():
    return float(sum(e["cpu_s"] for e in json.loads(LEDGER.read_text()))) if LEDGER.exists() else 0.0
def ledger_add(tag, cpu_s):
    L = json.loads(LEDGER.read_text()) if LEDGER.exists() else []; L.append(dict(tag=tag, cpu_s=float(cpu_s), at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))); atomic_json(LEDGER, L)
def pilot_poses():
    P = [p for p in json.loads(XCFG.read_text())["poses"] if p["x"] <= 0.39 + 1e-9]; P.sort(key=lambda p: p["idx"]); assert len(P) == 24 and [p["x"] for p in P if p["block"] == 0] == [0.34, 0.35, 0.36, 0.37, 0.38, 0.39]; return P
def historical_a0():
    R = {}
    for ln in open(XSWEEP_C0S0 / "episodes.jsonl"):
        r = json.loads(ln); R[r["episode_index"]] = r
    return R
# ---------------------------------------------------------------------------------------------------------------------------------- ACT chunk policy
def load_chunk_policies(torch, names=CK_NAMES):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import OBS_IMAGES
    class ChunkPolicy:
        def __init__(self, name):
            path = ROOT / "outputs/checkpoints" / CK[name] / "last"; self.name = name; self.policy = ACTPolicy.from_pretrained(str(path)); self.policy.eval()
            self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=str(path)); self.cams = [k.split(".")[-1] for k in self.policy.config.image_features]
            assert self.policy.config.chunk_size == CHUNK and self.policy.config.n_action_steps == CHUNK and not self.policy.config.use_vae
        @torch.no_grad()
        def chunk(self, state, images):
            batch = {"observation.state": torch.as_tensor(state, dtype=torch.float32)[None]}
            for c in self.cams: batch[f"observation.images.{c}"] = torch.as_tensor(images[c]).permute(2, 0, 1).float().div(255.0)[None]
            b = dict(self.pre(batch)); b[OBS_IMAGES] = [b[k] for k in self.policy.config.image_features]
            self.policy.model.eval(); a = self.policy.model(b)[0]                       # (1,20,16) normalised, deterministic (use_vae False)
            out = torch.stack([torch.as_tensor(self.post(a[:, k]).reshape(-1)) for k in range(a.shape[1])]).cpu().numpy().astype(np.float64)
            assert out.shape == (CHUNK, 16) and np.isfinite(out).all(), out.shape
            return out
    return {k: ChunkPolicy(k) for k in names}
def load_wm(torch):
    sys.argv = [sys.argv[0], "none"]
    import wm2_run as R, wm2_models as M, wm3b_model as B
    M.logistic_fit = lambda X, y, l2=0.01, iters=200: (np.zeros((X.shape[1], y.shape[1]), np.float32), np.zeros(y.shape[1], np.float32))      # event heads unused by J
    tr = R.load("train"); nrm = M.Norm(tr); ridge = M.Ridge("cond").fit(nrm, tr); w3 = B.WM3B(ridge, nrm); w3.model = B.GRUOC(); w3.model.load_state_dict(torch.load(ROOT / "outputs/wm3/models/wm3b.pt")); w3.model.eval(); return w3
# ---------------------------------------------------------------------------------------------------------------------------------- live privileged observer
class LiveObserver:
    """Per-step privileged rows identical to data/wm1_sim.simulate (row t = observation BEFORE action t) -> contract-v1 b (7) and c (7) rows via wm1_sim.contact_features (row-wise, causal)."""
    def __init__(self, env):
        import mujoco; import wm1_sim as S
        self.mj, self.S, self.env = mujoco, S, env; m = env.m
        self.gb, self.gL, self.gR, self.gH = m.geom("blue_box_collision").id, m.geom("openarm_left_left_finger_collision").id, m.geom("openarm_left_right_finger_collision").id, m.geom("openarm_left_hand_collision").id
    def row(self):
        mujoco, S, env = self.mj, self.S, self.env; m, d = env.m, env.d; box = env.box; Rt = d.xmat[env.tcp_id].reshape(3, 3); Tp = d.xpos[env.tcp_id].copy(); Bp = d.xpos[box].copy(); Rb = d.xmat[box].reshape(3, 3); f6 = np.zeros(6)
        ff, nf, ca, hf, tf, rf = 0.0, 0, [], 0.0, 0.0, 0.0
        for i in range(d.ncon):
            c = d.contact[i]; g1, g2 = c.geom1, c.geom2
            if self.gb not in (g1, g2): continue
            o = g2 if g1 == self.gb else g1; mujoco.mj_contactForce(m, d, i, f6); fn = abs(f6[0])
            if o in (self.gL, self.gR): ff += fn; nf += 1; ca.append((Rt.T @ (c.pos - Tp))[2])
            elif o == self.gH: hf += fn
            elif m.body(m.geom_bodyid[o]).name == "tray": rf += fn
            else: tf += fn
        R = dict(box_pos=np.array([Bp], np.float32), box_yaw=np.array([S.yaw_of(d.xquat[box])], np.float32), box_speed=np.array([float(np.linalg.norm(d.cvel[box][3:]))], np.float32), box_tilt=np.array([float(np.arccos(np.clip(Rb[2, 2], -1, 1)))], np.float32),
                 box_in_tcp=np.array([Rt.T @ (Bp - Tp)], np.float32), finger_force=np.array([ff], np.float32), finger_contacts=np.array([nf], np.float32), finger_contact_a=np.array([np.mean(ca) if ca else np.nan], np.float32),
                 hand_force=np.array([hf], np.float32), table_force=np.array([tf], np.float32), tray_force=np.array([rf], np.float32), action=np.zeros((1, 16), np.float32))
        b, c = S.contact_features(R); return b[0], c[0]
# ---------------------------------------------------------------------------------------------------------------------------------- windows, scoring, phase
def build_window(S_rows, B_rows, C_rows, A_exec, t, cand_exe, pose):
    """Everything strictly at or before row t / action t-1; cand_exe (n,20,16) executed candidates.  Raises on malformed shapes."""
    n = cand_exe.shape[0]
    if t < 4 or cand_exe.shape[1:] != (CHUNK, 16): raise ValueError(f"bad window request t={t} cand={cand_exe.shape}")
    S_ = np.asarray(S_rows[t - 3:t + 1], np.float32)[:, :8]; B_ = np.asarray(B_rows[t - 3:t + 1], np.float32); C_ = np.asarray(C_rows[t - 3:t + 1], np.float32); A_ = np.asarray(A_exec[t - 4:t], np.float32)[:, :8]
    if S_.shape != (4, 8) or B_.shape != (4, 7) or C_.shape != (4, 7) or A_.shape != (4, 8): raise ValueError(f"history shapes {S_.shape} {B_.shape} {C_.shape} {A_.shape}")
    return dict(S_hist=np.repeat(S_[None], n, 0), B_hist=np.repeat(B_[None], n, 0), C_hist=np.repeat(C_[None], n, 0), A_hist=np.repeat(A_[None], n, 0), A_fut=np.asarray(cand_exe[:, :, :8], np.float32), t=np.full(n, t), Y_e=np.zeros((n, 4), np.float32), pose=np.repeat(np.array([pose], np.float32), n, 0))
def score_candidates(w3, L, W, b_t, c_t):
    P = w3.predict(W); bb = b_t[None] + np.cumsum(P["Y_b"], 1); terms = []
    for i in range(W["A_fut"].shape[0]): terms.append(L.terms(np.vstack([b_t[None], bb[i]]), np.vstack([c_t[None], P["Y_c"][i]])))
    return terms, np.array([L.score(tm) for tm in terms])
def phase_of(cum_contacts, max_rise, dxy_tray):
    """Pre-registered deterministic phase labels from privileged state so far."""
    if cum_contacts == 0 and max_rise < 0.01: return "approach"
    if max_rise < 0.05: return "grasp"
    return "place" if dxy_tray < 0.03 else "carry"
def slew_chain(lim, raw, prev):
    out = []; nr = ns = 0; p = np.asarray(prev, float)
    for r in raw: c, a, b = lim.clip_and_slew(r, p, DT); out.append(c); p = c; nr += a; ns += b
    return np.array(out), nr, ns
def check_legal(lim, exe, prev):
    e = np.vstack([np.asarray(prev, float)[None], exe]); rate_ok = bool((np.abs(np.diff(e, axis=0)) / DT <= lim.max_speed[None] * (1 + 1e-9) + 1e-9).all()); rng_ok = bool(((exe >= lim.lo - 1e-9) & (exe <= lim.hi + 1e-9)).all())
    return rate_ok and rng_ok and bool(np.isfinite(exe).all()) and exe.shape[1] == 16
# ---------------------------------------------------------------------------------------------------------------------------------- episode
def lean_step(env, ctrl, need_images):
    """Identical physics/bookkeeping to LabEnv.step, but cameras are rendered only when the next observation is fed to ACT (every 20th step): ACT inputs are unchanged, rendering has no effect on the simulation state."""
    import mujoco
    env.d.ctrl[:] = ctrl; env.prev_ctrl = np.asarray(ctrl, float).copy()
    for _ in range(env.substeps): mujoco.mj_step(env.m, env.d)
    env.t += 1.0 / env.hz
    o = dict(state=env.adapter.state(env.d), velocity=env.adapter.velocity(env.d), t=env.t)
    if need_images: o["images"] = {c: env.rig.rgb(env.d, c) for c in env.cams}
    return o
def run_episode(arm, pose, repeat, env, obs_, policies, w3, lim, guard=None, save_dir=None, horizon=HORIZON, keep_arrays=False):
    import decision_utility_lib as L, lab_scene
    from eval.metrics import task_labels, tracking_error
    x, y, yaw, pid = pose["x"], pose["y"], pose["yaw"], pose["idx"]; rng = np.random.default_rng(RNG_KEY + [pid, repeat]) if arm == "A1" else None
    env.set_perturbation(); o = env.reset((x, y), yaw); prev = env.prev_ctrl.copy(); S_rows, B_rows, C_rows, A_exec, Bp, TCP, BS = [], [], [], [], [], [], []; decisions = []; n_clip = n_slew = 0; invalid = 0; exe = None; src = None; cum_contacts = 0.0; max_rise = 0.0; z0 = None
    t_ep0 = time.perf_counter(); t_act = t_wm = 0.0
    assert horizon % CHUNK == 0
    for t in range(horizon):
        if guard: guard()
        i = env.info(); b, c = obs_.row(); S_rows.append(np.asarray(o["state"], np.float32)); B_rows.append(b); C_rows.append(c); Bp.append(i["box_pos"]); TCP.append(i["tcp_pos"]); BS.append(i["box_speed"])
        if z0 is None: z0 = float(i["box_pos"][2])
        cum_contacts += float(c[1]); max_rise = max(max_rise, float(i["box_pos"][2] - z0))
        if t % CHUNK == 0:
            ci = t // CHUNK; ph = phase_of(cum_contacts, max_rise, float(np.linalg.norm(i["box_pos"][:2] - np.asarray(lab_scene.TRAY_C))))
            if ci == 0 or arm == "A0": names_cand = [DEFAULT]; sel = DEFAULT; raws = None
            elif arm == "A1": sel = CK_NAMES[int(rng.integers(len(CK_NAMES)))]; names_cand = [sel]
            else: names_cand = list(CK_NAMES)
            t0 = time.perf_counter(); raws = {k: policies[k].chunk(o["state"], o["images"]) for k in names_cand}; t_act += time.perf_counter() - t0
            exes = {}; counts = {}
            for k in names_cand: exes[k], nr, ns = slew_chain(lim, raws[k], prev); counts[k] = (nr, ns)
            dec = dict(t=t, chunk=ci, phase=ph, arm=arm, candidates=names_cand)
            if arm == "A2" and ci > 0:
                t0 = time.perf_counter(); W = build_window(S_rows, B_rows, C_rows, A_exec, t, np.stack([exes[k] for k in names_cand]), [x, y, yaw]); terms, J = score_candidates(w3, L, W, B_rows[t], C_rows[t]); t_wm += time.perf_counter() - t0
                vids = list(range(len(names_cand))); k_sel = L.select(J, pid, vids, tol=TOL_TIE); sel = names_cand[k_sel]
                dec.update(J=[float(v) for v in J], terms=[dict(dxy=float(tm["dxy"]), dz=float(tm["dz"]), palm=bool(tm["palm"]), lifted=bool(tm["lifted"])) for tm in terms], selected=sel, n_within_tie_tol=int(np.sum(J - J.min() < TOL_TIE)), margin_to_second=float(np.sort(J)[1] - np.sort(J)[0]), default_J=float(J[0]), spread=float(np.ptp(J)))
            elif arm == "A1" and ci > 0: dec.update(selected=sel)
            else: dec.update(selected=sel)
            exe = exes[sel]; n_clip += counts[sel][0]; n_slew += counts[sel][1]; src = sel
            if not check_legal(lim, exe, prev): invalid += 1
            decisions.append(dec)
        a = exe[t % CHUNK]; A_exec.append(np.asarray(a, np.float32)); prev = a; o = lean_step(env, a, (t + 1) % CHUNK == 0 and t + 1 < horizon)
    i = env.info(); Bp.append(i["box_pos"]); TCP.append(i["tcp_pos"]); BS.append(i["box_speed"])
    lab = task_labels(Bp, TCP, BS, lab_scene.TRAY_C, lab_scene.TABLE_TOP_Z + lab_scene.TRAY_BASE_T); S_arr, A_arr = np.array(S_rows), np.array(A_exec); C_arr = np.array(C_rows)
    outcome = "success" if lab["task_success"] else ("no_pick" if not lab["pick_success"] else "dropped_or_misplaced"); tray = np.asarray(lab_scene.TRAY_C); fb = np.asarray(Bp[-1])
    palm = bool((C_arr[:, 4] > L.PALM_THR).any()); cluster = bool((not lab["task_success"]) and lab["pick_success"] and (fb[0] - tray[0] < -0.025) and abs(fb[1] - tray[1]) < 0.01)
    rec = dict(arm=arm, pose_idx=pid, repeat=repeat, x=x, y=y, yaw=yaw, block=pose["block"], outcome=outcome, palm_contact=palm, cluster_flag=cluster, action_invalid_chunks=invalid, n_clipped=int(n_clip), n_slewed=int(n_slew), **lab, tracking=tracking_error(S_arr, A_arr),
               decisions=decisions, act_ms=t_act * 1000, wm_ms=t_wm * 1000, wall_s=time.perf_counter() - t_ep0, finite=bool(np.isfinite(A_arr).all() and np.isfinite(S_arr).all()))
    if keep_arrays: rec["_arrays"] = dict(state=S_arr, action=A_arr, b=np.array(B_rows), c=C_arr)
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True); np.savez_compressed(Path(save_dir) / f"{arm}_p{pid:02d}_r{repeat}.npz", state=S_arr, action=A_arr, box_pos=np.array(Bp), tcp_pos=np.array(TCP), c_rows=C_arr)
    return rec
def make_env(cams):
    from lab_env import LabEnv
    return LabEnv(image_hw=(120, 160), control_hz=20, render_profile="fast", cameras=tuple(cams))
# ---------------------------------------------------------------------------------------------------------------------------------- worker / orchestrator
def make_guard(c0, cap_s, wid):
    def guard(margin=0.0):
        if cpu() - c0 + margin > cap_s: raise CapReached(f"worker {wid}")
    return guard
G = {}          # per-process globals (policies, WM, env) - initialised once per pool worker
def init_worker(cap_s, objs=None):
    import torch; torch.set_num_threads(1); os.environ["LP_NUM_THREADS"] = "1"
    from control.safety_limits import SafetyLimits
    G["c0"] = cpu(); G["cap_s"] = cap_s
    if objs is None:
        policies = load_chunk_policies(torch); cams = policies[DEFAULT].cams; assert all(p.cams == cams for p in policies.values()); w3 = load_wm(torch); env = make_env(cams)
    else: policies, w3, env = objs
    G.update(policies=policies, w3=w3, env=env, obs=LiveObserver(env), lim=SafetyLimits.from_model(env.m), poses={p["idx"]: p for p in pilot_poses()}, init_cpu=cpu() - G["c0"])
def run_task(task):
    """task = (pose_idx, arm, repeat[, horizon]).  Returns a JSON-serialisable dict; never raises across the pool (status OK | CAP | ERROR)."""
    pid, arm, rep = task[:3]; horizon = task[3] if len(task) > 3 else HORIZON; guard = make_guard(G["c0"], G["cap_s"], os.getpid())
    try:
        guard(margin=45.0)
        r = run_episode(arm, G["poses"][pid], rep, G["env"], G["obs"], G["policies"], G["w3"], G["lim"], guard=lambda: guard(), save_dir=(OUT / "results/traj") if horizon == HORIZON else None, horizon=horizon)
        r.update(status="OK", pid_worker=os.getpid(), cpu_s_cum=cpu() - G["c0"]); return r
    except CapReached as e: return dict(status="CAP", task=list(task), reason=str(e), cpu_s_cum=cpu() - G["c0"])
    except Exception as e:
        import traceback; return dict(status="ERROR", task=list(task), reason=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:], cpu_s_cum=cpu() - G["c0"])
def verify_freeze():
    F = json.loads(FREEZE.read_text()); bad = [f for f, h in F["files_sha256"].items() if sha256(ROOT / f) != h]
    if bad: raise SystemExit(f"frozen files changed: {bad}")
    return F
def main():
    if len(sys.argv) < 2 or sys.argv[1] != "run": raise SystemExit("usage: closed_loop_pilot.py run")
    F = verify_freeze(); rep = json.loads((OUT / "integration_report.json").read_text())
    if rep["verdict"] != "PASS": raise SystemExit("integration gates not PASS")
    if rep["code_sha256"]["eval/closed_loop_pilot.py"] != sha256(__file__): raise SystemExit("runner changed after the integration gates")
    resf = OUT / "results/pilot_results.json"
    if resf.exists(): raise SystemExit("results exist; refusing to overwrite")
    nw = 4; used = ledger_used(); remaining = CAP_TOTAL_S - used - 60.0; assert remaining > 900, remaining; cap_w = remaining / nw; poses = pilot_poses(); t0 = time.time(); c_par0 = cpu()
    state = dict(protocol_freeze_sha256=sha256(FREEZE), status="RUNNING", phase=1, cap_per_worker_s=cap_w, ledger_used_before_s=used, episodes=[], events=[])
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(nw, initializer=init_worker, initargs=(cap_w,)) as pool:
        def run_phase(tasks):
            n_cap = 0
            for r in pool.imap_unordered(run_task, tasks, chunksize=1):
                state["episodes"].append(r); n_cap += r["status"] != "OK"; atomic_json(resf, state)
                if r["status"] != "OK": state["events"].append(dict(stop=r["status"], reason=r.get("reason")))
                print(f"[{r.get('arm','?')} p{r.get('pose_idx','?')} r{r.get('repeat','?')}] {r['status']} {r.get('outcome','')} cpu_w={r['cpu_s_cum']:.0f}s", flush=True)
            return n_cap
        n1 = run_phase([(p["idx"], "A0", 0) for p in poses]); A0 = [r for r in state["episodes"] if r["status"] == "OK" and r["arm"] == "A0"]; hist = historical_a0()
        diff = [r["pose_idx"] for r in A0 if r["outcome"] != hist[r["pose_idx"]]["outcome"]]; succ = sum(r["task_success"] for r in A0) / max(len(A0), 1)
        state["phase1"] = dict(n_a0=len(A0), a0_success_rate=succ, differs_from_historical=diff, n_differs=len(diff))
        stop = None
        if n1 or len(A0) < len(poses): stop = "phase 1 incomplete (cap/error)"
        elif len(diff) > 2: stop = f"A0 differs from historical xsweep on {len(diff)} > 2 poses"
        elif succ >= 0.90: stop = f"A0 success {succ:.3f} >= 0.90 (no headroom)"
        if stop: state["status"] = "STOPPED_PHASE1"; state["stop_reason"] = stop
        else:
            state["phase"] = 2; tasks = [(p["idx"], a, r) for p in poses for a, r in (("A2", 0), ("A1", 0), ("A1", 1), ("A1", 2))]; n2 = run_phase(tasks); state["status"] = "COMPLETE" if n2 == 0 else "PARTIAL"
            if n2: state["stop_reason"] = "cap or error in phase 2"
        pool.close(); pool.join()
    cpu_s = cpu() - c_par0          # parent + reaped pool workers (RUSAGE_CHILDREN)
    state["wall_s"] = time.time() - t0; state["cpu_s_estimate"] = cpu_s; atomic_json(resf, state); ledger_add("pilot_run_estimate", cpu_s); print(json.dumps({k: state[k] for k in ("status", "phase1", "cpu_s_estimate", "wall_s") if k in state}), flush=True)
if __name__ == "__main__": main()
