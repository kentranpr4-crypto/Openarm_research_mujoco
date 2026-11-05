#!/usr/bin/env python3
"""MOCK context for the full-harness dry-run of eval/g2b_integration_v2.py (NO ACT, NO torch, NO MuJoCo, NO real policy).  A deterministic toy environment and policy let the REAL gate code (I2, I3, I4), the REAL S0 runner, the REAL labeller / recorder-record code and the REAL orchestrator run end to end;
scenarios (env var G2B_INT_DRY) inject failures.  Used only by eval/test_g2b_integration_dryrun.py."""
import os, sys, copy, time, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import g2b_trace as TR_REAL, g2b_labeller as LB, g2b_s0 as S0
class MockD:
    def __init__(s): s.qpos = np.zeros(16); s.qvel = np.zeros(6); s.qacc_warmstart = np.zeros(16); s.ctrl = np.zeros(16); s.time = 0.0; s.xpos = np.zeros((3, 3)); s.xmat = np.tile(np.eye(3).ravel(), (3, 1)); s.xquat = np.tile([1.0, 0, 0, 0], (3, 1)); s.xpos[1] = [0.4, 0.1, 0.38]
class MockAdapter:
    def state(self, d): return d.qpos.astype(np.float32).copy()
class MockEnv:
    cams = ["cam"]; m = None
    def __init__(s, mode): s.mode = mode; s.d = MockD(); s.prev_ctrl = np.zeros(16); s.t = 0.0; s.adapter = MockAdapter()
    def set_perturbation(s): pass
    def reset(s, xy, yaw): s.d = MockD(); s.d.qpos[:3] = [xy[0], xy[1], yaw]; s.prev_ctrl = np.zeros(16); s.prev_ctrl[7] = 0.044; s.t = 0.0; return None
class MockLim:
    max_speed = np.full(16, 2.0); lo = np.full(16, -1e3); hi = np.full(16, 1e3)
    def clip_and_slew(s, raw, prev, dt):
        raw = np.clip(np.asarray(raw, float), s.lo, s.hi); step = np.clip(raw - prev, -s.max_speed * dt, s.max_speed * dt); return prev + step, 0, 0
def burn(sec):
    t = time.process_time()
    while time.process_time() - t < sec: sum(i * i for i in range(2000))
class MockPolicy:
    cams = ["cam"]
    def __init__(s, mode): s.mode = mode; s.n = 0; s.last = None; s.policy = SimpleNamespace(reset=lambda: None, _action_queue=[]) if mode != "no_reset_attr" else SimpleNamespace(_action_queue=[])
    def chunk(s, state, images):
        s.n += 1; key = np.asarray(state, np.float32).tobytes()
        if s.mode == "raise_in_i3" and s.n == 14: raise RuntimeError("mock policy exception")
        if s.mode == "hard_spin" and s.n == 12: burn(1000.0)
        if s.mode == "i4_between" and s.n == 16: burn(1.2)                                                       # last query of pose 2 -> the guard at the start of pose 3 must stop BETWEEN poses
        if s.mode == "i4_mid" and s.n == 9: burn(1.2)                                                            # first query of pose 2 -> the guard inside pose 2 must stop MID-pose
        if s.mode == "i4_slow": time.sleep(0.05)
        if s.mode == "coop_cap": burn(0.4)
        out = 0.3 * np.sin(np.asarray(state, float)[None, :] + np.arange(20)[:, None] * 0.1)
        if s.mode == "noisy_policy": out = out + 1e-3 * s.n
        if s.mode == "stateful_policy" and s.last == key: out = out + 1e-3                                       # only a REPEATED identical query differs (the statelessness check)
        s.last = key; return out
def lean_step(env, u, need_images):
    env.d.ctrl[:] = u; env.prev_ctrl = np.asarray(u, float).copy(); env.d.qpos = 0.8 * env.d.qpos + 0.2 * np.asarray(u, float); env.d.qacc_warmstart = np.asarray(u, float) * 0.01; env.d.qvel[:3] = [0.0, 0.0, 0.01]; env.d.xpos[1][2] += 0.0005; env.d.time += 0.05; env.t += 0.05
def slew_chain(lim, raw, prev):
    out = []; p = np.asarray(prev, float)
    for r in raw: c, _, _ = lim.clip_and_slew(r, p, 0.05); out.append(c); p = c
    return np.array(out), 0, 0
def check_legal(lim, exe, prev):
    e = np.vstack([np.asarray(prev, float)[None], exe]); return bool((np.abs(np.diff(e, axis=0)) / 0.05 <= lim.max_speed[None] * (1 + 1e-9) + 1e-9).all() and np.isfinite(exe).all() and exe.shape[1] == 16)
def make_ctx(mode):
    env = MockEnv(mode)
    def snapshot(e): return dict(d=copy.deepcopy(e.d), prev=np.asarray(e.prev_ctrl, float).copy(), t=float(e.t))
    def restore(e, snap):
        keep = e.d.qacc_warmstart.copy(); e.d = copy.deepcopy(snap["d"]); e.prev_ctrl = snap["prev"].copy(); e.t = snap["t"]
        if mode == "bad_restore": e.d.qacc_warmstart = keep                                                      # solver warm-start NOT restored -> I2 must FAIL
    def obs_full(e): return dict(state=e.adapter.state(e.d), velocity=e.d.qvel.copy(), t=e.t, images={"cam": (np.abs(e.d.qpos[:12].reshape(3, 4)) * 255).astype(np.uint8)})
    BR = SimpleNamespace(snapshot=snapshot, restore=restore, obs_full=obs_full); P = SimpleNamespace(slew_chain=slew_chain, lean_step=lean_step, check_legal=check_legal)
    class FakeCmap:
        tcp, box_body, finger_qadr, box_dof = 0, 1, 2, 0
        def read_contacts(s, d): return dict(nL=0, nR=0, nH=0, nO=0, FL=0.0, FR=0.0, Fh=0.0, Fo=0.0, neg_normal=0, F_on_box_world=np.zeros(3), pen_obst=0.0)
    def recorder(m, meta, refresh=True):
        r = TR_REAL.TraceRecorder.__new__(TR_REAL.TraceRecorder); r.cmap = FakeCmap(); r.m = None; r.rows = {k: [] for k in TR_REAL.FIELDS}; r.meta = dict(meta); r._last = None; r.refresh = False; r._scr = None; return r
    TR = SimpleNamespace(TraceRecorder=recorder, to_labeller_trace=TR_REAL.to_labeller_trace, save_trace=TR_REAL.save_trace)
    poses = [dict(idx=3000 + i, x=0.36 + 0.01 * i, y=0.1, yaw=0.0) for i in range(3)]
    if mode == "two_poses": poses = poses[:2]
    return SimpleNamespace(np=np, mujoco=None, P=P, BR=BR, S0=S0, TR=TR, LB=LB, ScriptedExpert=None, env=env, m=None, lim=MockLim(), pol=MockPolicy(mode), POSES=poses)
def dry_workload(rep, guard, args):
    import g2b_integration_v2 as I
    mode = os.environ["G2B_INT_DRY"]; rep.milestone("import_numpy"); rep.milestone("import_project_modules"); ctx = make_ctx(mode); rep.milestone("act_setup_done")
    cp = Path(rep.path)
    if mode == "no_report": cp.unlink(); os._exit(0)
    if mode == "truncated_report": cp.write_text('{"status": "COMPLE'); os._exit(0)
    if mode == "child_crash_hard": os._exit(7)
    wanted, required, reuse = I.parse_env(); I.run_gates(ctx, rep, guard, wanted, required, reuse)
