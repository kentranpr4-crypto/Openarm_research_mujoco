#!/usr/bin/env python3
"""G2b limited INTEGRATION gates I1-I4 (NOT an evaluation; no treatment effect; no 7-arm branching; no training; no new dataset).  Approved by the 2026-11-05 review decision with a HARD cap of 0.05 aggregate CPU-h = 180 CPU-s for the whole integration task
(the ledger outputs/g2b_integration/cpu_ledger.json already holds the static/synthetic test cost; this process adds its own).  Sequential, stop on first failure, no automatic retry, cap never raised.
Poses: the first three KNOWN development poses of the G2a freeze (idx 3000-3002); sealed reserve poses, the 40 prospective G2b poses and ACT test seeds 31-33 are not touched.
  I1  scripted expert, one known pose, <=160 steps: full per-finger trace -> labels; truncation and invalid traces; state-time consistency of the trace (raw vs refreshed vs qpos, historical observer, live-state non-interference)
  I2  snapshot restore determinism + c0 continuation reproduction (ACT c0_s0 checkpoint)
  I3  S0 / ACT integration: k=0 equivalence with the nominal loop; k=8 hold + resume, observation/query timing, policy reset, statelessness
  I4  CPU benchmark: nominal c0, <=3 known poses, horizon 160, with trace logging and labelling (only if I1-I3 PASS)
Run (ACT venv): /home/user/.venvs/act/bin/python eval/g2b_integration.py"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import sys, json, time, resource, traceback, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "eval"), str(ROOT / "data")): sys.path.insert(0, p)
import closed_loop_pilot as P
OUT = ROOT / "outputs/g2b_integration"; LEDGER = OUT / "cpu_ledger.json"; REPORT = OUT / "integration_report.json"; CAP_S = 180.0; GUARD_S = 172.0; DT = 0.05; HORIZON = 160; T_DEC = 40; T_END_I23 = 100
def ledger_prev(): return float(sum(e["cpu_s"] for e in json.loads(LEDGER.read_text()))) if LEDGER.exists() else 0.0
PREV = ledger_prev()
class Cap(Exception): pass
def total_cpu(): return PREV + P.cpu()
def guard(margin=0.0):
    if total_cpu() + margin > GUARD_S: raise Cap(f"aggregate CPU {total_cpu():.1f} s + margin {margin:.1f} s would exceed the guard {GUARD_S} s (cap {CAP_S} s)")
resource.setrlimit(resource.RLIMIT_CPU, (int(max(5, CAP_S - PREV)), int(max(6, CAP_S - PREV + 2))))                                  # backstop only
POSES = [p for p in json.loads((ROOT / "outputs/g2a_act_corrections/protocol_freeze.json").read_text())["poses"] if p["idx"] in (3000, 3001, 3002)]
assert [p["idx"] for p in POSES] == [3000, 3001, 3002]
GATES = []; CTX = {}
def save(status="RUNNING"):
    rep = dict(task="G2b integration gates I1-I4", status=status, cap_s=CAP_S, guard_s=GUARD_S, ledger_before_s=PREV, process_cpu_s=round(P.cpu(), 2), aggregate_cpu_s=round(total_cpu(), 2), gates=GATES, poses=[p["idx"] for p in POSES])
    P.atomic_json(REPORT, rep); return rep
def run_gate(name, fn, need=0.0):
    c0 = P.cpu(); ok = False; ev = {}; partial = False
    try: guard(margin=need); ok, ev = fn()
    except Cap as e: ok, partial, ev = False, True, dict(stopped="CPU budget guard", detail=str(e))
    except Exception as e: ok, ev = False, dict(exception=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-1500:])
    GATES.append(dict(gate=name, result="PASS" if ok else ("PARTIAL" if partial else "FAIL"), cpu_s=round(P.cpu() - c0, 2), aggregate_cpu_after_s=round(total_cpu(), 2), evidence=ev)); save(); print(f"{name}: {'PASS' if ok else 'FAIL'}  gate CPU {GATES[-1]['cpu_s']} s  aggregate {GATES[-1]['aggregate_cpu_after_s']} s", flush=True)
    if not ok: print(json.dumps(ev, default=float)[:1200], flush=True)
    return ok
# ----------------------------------------------------------------------------------------------------------------------------- I1
def I1():
    import mujoco, g2b_trace as TR, g2b_labeller as LB
    from policies.baseline import ScriptedExpert
    from control.safety_limits import SafetyLimits
    env = P.make_env(()); CTX["env_noimg"] = env; m = env.m; ex = ScriptedExpert(m, control_hz=20); lim = SafetyLimits.from_model(m); obs = P.LiveObserver(env); pose = POSES[0]; xy = (pose["x"], pose["y"]); pl = ex.plan(xy, pose["yaw"]); T = int(min(len(pl["ctrl"]), HORIZON))
    env.set_perturbation(); env.reset(xy, pose["yaw"]); prev = env.prev_ctrl.copy(); g_prev0 = float(prev[7]); rec = TR.TraceRecorder(m, dict(id="I1", pose=pose, t0=0), refresh=True); cm = rec.cmap; box_z0 = float(env.d.xpos[cm.box_body][2]); adr = int(m.jnt_qposadr[m.joint("blue_box_free").id])
    d_raw_qpos, d_ref_qpos, rel_lag, hist_d, noninterf, speeds, sub_checks, scr = [], [], [], [], [], [], [], mujoco.MjData(m)
    for t in range(T):
        guard(margin=6.0); c, _, _ = lim.clip_and_slew(pl["ctrl"][t], prev, DT); d = env.d
        before = (d.qpos.copy(), d.qvel.copy(), d.qacc_warmstart.copy(), d.ctrl.copy(), float(d.time))
        rec.record(d, c[7], c); same = all(np.array_equal(a, b) for a, b in zip(before[:4], (d.qpos, d.qvel, d.qacc_warmstart, d.ctrl))) and before[4] == float(d.time); noninterf.append(bool(same))
        qb = d.qpos[adr:adr + 3].copy(); raw_x = d.xpos[cm.box_body].copy(); ref_x = rec.rows["box_pos"][-1]; v = float(np.linalg.norm(d.qvel[cm.box_dof:cm.box_dof + 3])); speeds.append(v)
        d_ref_qpos.append(float(np.abs(ref_x - qb).max())); d_raw_qpos.append(float(np.abs(raw_x - qb).max())); rel_lag.append(float(np.linalg.norm(raw_x - qb)) / (v * 0.001) if v > 0.05 else np.nan)
        rr = cm.read_contacts(d); b_, c_ = obs.row(); hist_d.append(max(abs((rr["FL"] + rr["FR"]) - c_[0] * 50.0), abs((rr["nL"] + rr["nR"]) - c_[1]), abs(rr["Fh"] - c_[4])))
        if v > 0.10 and len(sub_checks) < 3:                                                                                      # one-substep qvel identity on a scratch copy (live data untouched)
            mujoco.mj_copyData(scr, m, d); x0 = scr.qpos[adr:adr + 3].copy(); v0 = scr.qvel[cm.box_dof:cm.box_dof + 3].copy(); mujoco.mj_step(m, scr); x1 = scr.qpos[adr:adr + 3].copy(); v1 = scr.qvel[cm.box_dof:cm.box_dof + 3].copy(); h = m.opt.timestep
            fd = (x1 - x0) / h; mean = 0.5 * (v0 + v1); sub_checks.append(float(np.linalg.norm(fd - mean) / np.linalg.norm(mean)))
        P.lean_step(env, c, False); prev = c
    rec.record_final(env.d); A = rec.arrays(); meta = dict(t0=0, box_z0=box_z0, g_prev0=g_prev0); lt = TR.to_labeller_trace(A, meta); lab = LB.label(lt); ev = lab["events"]; N = len(A["g_exec"])
    TR.save_trace(OUT / "traces/I1_expert_p3000", A, dict(id="I1", pose=pose, t0=0, g_prev0=g_prev0, note="integration gate; scripted expert; not an evaluation"), label=lab, code_files=[ROOT / "eval/g2b_trace.py", ROOT / "eval/g2b_labeller.py"])
    # truncation: every cut gives either the full-trace value or CENSORED (never a contradicting FALSE/TRUE); timestamps of events never move
    full = {e: lab["endpoints"][e]["value"] for e in ("E1", "E2", "E3")}; bad_cut = []
    for n in list(range(6, N, 4)) + [N - 1]:
        cut = dict(lt); cut["meta"] = dict(lt["meta"])
        for k in ("g_exec", "g_meas", "nL", "nR", "FL", "FR", "box_pos", "box_vel", "Fh"): cut[k] = lt[k][:n]
        o = LB.label(cut)
        for e in ("E1", "E2", "E3"):
            if o["endpoints"][e]["value"] not in (full[e], "CENSORED"): bad_cut.append((n, e, o["endpoints"][e]["value"], full[e]))
        for k, tt in o["events"].items():
            if k in ev and tt != ev[k] and k != "retention_window_end": bad_cut.append((n, k, tt, ev[k]))
    nan = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in lt.items()}; nan["FL"][5] = np.nan; inv_nan = LB.label(nan)["terminal"]; inv_meta = LB.label(dict(lt, meta=dict(lt["meta"], valid=False)))["terminal"]
    cc, ga, lc, ro = ev.get("close_command"), ev.get("grasp_acquired"), ev.get("lift_confirmed"), ev.get("release_command"); phys = ev.get("release_physical_onset"); gm = A["g_meas"]; g0 = float(gm[0]); closing_phys = int(np.argmax(gm < g0 - 0.001)) if (gm < g0 - 0.001).any() else None; ge_ = A["g_exec"]; first_cmd = int(np.argmax(ge_ < ge_[0] - 0.001)) if (ge_ < ge_[0] - 0.001).any() else None       # FIRST commanded departure from open (not the 0.040 threshold crossing)
    lagfin = np.array(rel_lag, float); lagfin = lagfin[np.isfinite(lagfin)]
    checks = dict(ordered_events=bool(cc is not None and ga is not None and lc is not None and cc < ga < lc and (ro is None or lc <= ro)), E1_TRUE_E2_not_FALSE=bool(full["E1"] == "TRUE" and full["E2"] != "FALSE"), no_neg_normal=bool(int(A["neg_normal"].sum()) == 0),
                  live_state_untouched_by_refresh=bool(all(noninterf)), refreshed_box_equals_qpos_1e9=bool(max(d_ref_qpos) < 1e-9), historical_observer_equal_1e5=bool(max(hist_d) < 1e-5), substep_qvel_identity_median_2pct=bool(len(sub_checks) > 0 and float(np.median(sub_checks)) < 0.02),
                  truncation_monotone=bool(not bad_cut), nan_invalid=bool(inv_nan == "INVALID"), meta_invalid=bool(inv_meta == "INVALID"), command_precedes_physical_closing=bool(first_cmd is not None and closing_phys is not None and closing_phys >= first_cmd + 1),
                  physical_open_after_command=bool(ro is None or phys is None or phys > ro))
    evd = dict(pose=pose["idx"], rows=N, steps=T, plan_len=int(len(pl["ctrl"])), terminal=lab["terminal"], subtype=lab["subtype"], endpoints=full, events=ev, margins=lab["margins"], ambiguous={e: lab["endpoints"][e]["ambiguous"] for e in ("E1", "E2", "E3")}, checks=checks, truncation_violations=bad_cut[:5],
               sample_timing=dict(raw_xpos_minus_qpos_max_m=max(d_raw_qpos), refreshed_minus_qpos_max_m=max(d_ref_qpos), raw_offset_over_v_dt_substep_median=float(np.median(lagfin)) if len(lagfin) else None, hist_observer_max_diff=max(hist_d), substep_qvel_rel_diff=sub_checks, max_box_speed_ms=max(speeds)),
               physical_vs_command=dict(first_commanded_departure_row=first_cmd, close_command_row=cc, first_physical_closing_row=closing_phys, release_command_row=ro, release_physical_onset_row=phys), invalid_tests=dict(nan=inv_nan, meta=inv_meta))
    return all(checks.values()), evd
# ----------------------------------------------------------------------------------------------------------------------------- shared ACT helpers
def setup_act():
    import torch; torch.set_num_threads(1); from control.safety_limits import SafetyLimits
    import branching_replay as BR
    CTX.update(BR=BR, torch=torch); pol = P.load_chunk_policies(torch, ["c0_s0"])["c0_s0"]; env = P.make_env(pol.cams); CTX.update(policy=pol, env=env, lim=SafetyLimits.from_model(env.m))
    class Spy:
        def __init__(s, p): s.p = p; s.calls = []; s.resets = 0; s.policy = p.policy; o = p.policy.reset; s._o = o
        def attach(s):
            def r(): s.resets += 1; s._o()
            s.p.policy.reset = r
        def chunk(s, state, images): s.calls.append(np.array(state, float).copy()); return s.p.chunk(state, images)
    CTX["Spy"] = Spy
def nominal(env, policy, lim, prev, t0, t1, rec=None, snap_at=None, timers=None):
    """independent reference loop (G2a-style): ACT chunk at every 20th step from the live observation, P.slew_chain, P.lean_step."""
    BR = CTX["BR"]; exe_all, Q, snap, sprev = [], [], None, None; exe = None; prev = np.asarray(prev, float).copy()
    for t in range(t0, t1):
        if (t - t0) % 20 == 0:
            if snap_at is not None and t == snap_at: snap = BR.snapshot(env); sprev = prev.copy()
            guard(margin=4.0); c0 = P.cpu(); o = BR.obs_full(env); raw = policy.chunk(o["state"], o["images"]); exe, _, _ = P.slew_chain(lim, raw, prev)
            if timers is not None: timers["act_s"] += P.cpu() - c0; timers["n_query"] += 1
        u = exe[(t - t0) % 20]
        if snap_at is not None and t == snap_at and snap is None: snap = BR.snapshot(env); sprev = prev.copy()
        if rec is not None:
            c1 = P.cpu(); rec.record(env.d, u[7], u)
            if timers is not None: timers["rec_s"] += P.cpu() - c1
        c2 = P.cpu(); P.lean_step(env, u, False)
        if timers is not None: timers["step_s"] += P.cpu() - c2
        exe_all.append(u.copy()); Q.append(env.d.qpos.copy()); prev = u
    return np.array(exe_all), np.array(Q), snap, sprev
def episode_to(env, policy, lim, pose, t_end, rec=None, snap_at=None, timers=None):
    env.set_perturbation(); env.reset((pose["x"], pose["y"]), pose["yaw"]); prev = env.prev_ctrl.copy(); return nominal(env, policy, lim, prev, 0, t_end, rec=rec, snap_at=snap_at, timers=timers), prev
# ----------------------------------------------------------------------------------------------------------------------------- I2
def I2():
    env, policy, lim, BR = CTX["env"], CTX["policy"], CTX["lim"], CTX["BR"]; pose = POSES[0]
    import g2b_trace as TR; rec = TR.TraceRecorder(env.m, dict(id="I2", pose=pose, t0=0), refresh=True)                                  # recorder (with scratch mj_forward refresh) attached ONLY to the continuous run; the restored branches have none -> bit equality proves it does not mutate the live simulation
    (exeA, QA, snap, sprev), _ = episode_to(env, policy, lim, pose, T_END_I23, rec=rec, snap_at=T_DEC); CTX.update(snap=snap, sprev=sprev, exeA=exeA, QA=QA); snap_q0 = snap["d"].qpos.copy(); snap_t = snap["t"]; res = []
    for rep in range(2):
        BR.restore(env, snap); t_after = env.t; exeB, QB, _, _ = nominal(env, policy, lim, sprev, T_DEC, T_END_I23); res.append(dict(rep=rep, restored_t_equals_snapshot=bool(abs(t_after - snap_t) < 1e-12), exe_equal=bool(np.array_equal(exeB, exeA[T_DEC:])), qpos_equal=bool(np.array_equal(QB, QA[T_DEC:])), max_abs_exe_diff=float(np.abs(exeB - exeA[T_DEC:]).max()), max_abs_qpos_diff=float(np.abs(QB - QA[T_DEC:]).max())))
        CTX[f"exeB{rep}"] = exeB; CTX[f"QB{rep}"] = QB
    snap_intact = bool(np.array_equal(snap["d"].qpos, snap_q0)); ok = all(r["restored_t_equals_snapshot"] and r["exe_equal"] and r["qpos_equal"] for r in res) and snap_intact and bool(np.array_equal(CTX["exeB0"], CTX["exeB1"]))
    return ok, dict(recorder_attached_run_equals_recorder_free_restores=bool(ok), pose=pose["idx"], t_dec=T_DEC, t_end=T_END_I23, restores=res, snapshot_not_mutated=snap_intact, restore0_equals_restore1=bool(np.array_equal(CTX["exeB0"], CTX["exeB1"])), n_steps_branch=int(len(CTX["exeB0"])))
# ----------------------------------------------------------------------------------------------------------------------------- I3
def I3():
    import g2b_s0 as S0
    env, policy, lim, BR, snap, sprev, Spy = CTX["env"], CTX["policy"], CTX["lim"], CTX["BR"], CTX["snap"], CTX["sprev"], CTX["Spy"]
    class Ad:
        def __init__(s, e): s.e = e
        def step(s, u): P.lean_step(s.e, u, False)
        def observe(s): return BR.obs_full(s.e)
    def branch(k):
        BR.restore(env, snap); spy = Spy(policy); spy.attach(); log = []; Q = []
        class AdQ(Ad):
            def step(s, u): super().step(u); Q.append(s.e.d.qpos.copy())
        ex, lg = S0.run_branch(AdQ(env), spy, lim, sprev, T_DEC, T_END_I23, k=k, on_step=lambda t, u, role: log.append((t, role, np.asarray(u, float).copy(), float(env.t))))
        policy.policy.reset = spy._o; return ex, lg, spy, log, np.array(Q)
    exe0, lg0, spy0, log0, Q0 = branch(0); eq_exe = bool(np.array_equal(exe0, CTX["exeB0"])); eq_q = bool(np.array_equal(Q0, CTX["QB0"])); q0 = [t for t, r in lg0 if r == "query"]
    exe8, lg8, spy8, log8, Q8 = branch(8); holds = [u for t, r, u, _ in log8 if r == "hold"]; acts = [(t, u, tt) for t, r, u, tt in log8 if r == "act"]; queries = [t for t, r in lg8 if r == "query"]
    hold_ok = len(holds) == 8 and all(np.array_equal(h, sprev) for h in holds) and all(r == "hold" for t, r, _, _ in log8[:8]); times_ok = all(abs(tt - (snap["t"] + (t - T_DEC) * DT)) < 1e-9 for t, _, _, tt in [(a, 0, 0, c) for a, _, c in acts])
    exp_q = S0.s0_query_steps(T_DEC, T_END_I23); q_state_ok = len(spy8.calls) == len(exp_q); legal = bool(P.check_legal(lim, exe8, sprev)); n8 = len(exe8)
    # statelessness: the same observation twice gives the same chunk (no hidden queue / action history); restore to the resume state of S0 is not needed: use the current (end) state
    o = BR.obs_full(env); a1 = policy.chunk(o["state"], o["images"]); a2 = policy.chunk(o["state"], o["images"]); stateless = bool(np.array_equal(a1, a2))
    diff_phys = float(np.abs(Q8[-1] - Q0[-1]).max()); checks = dict(k0_exe_equals_nominal=eq_exe, k0_qpos_equals_nominal=eq_q, k0_query_steps=bool(q0 == [40, 60, 80]), s0_hold_8_identical_to_last_target=bool(hold_ok), s0_query_steps_48_68_88=bool(queries == exp_q == [48, 68, 88]), s0_n_queries_match=bool(q_state_ok),
              s0_policy_reset_called_once=bool(spy8.resets == 1), k0_no_reset=bool(spy0.resets == 0), s0_commands_legal_clip_slew=legal, s0_steps_total_60=bool(n8 == 60), policy_stateless_same_obs_same_chunk=stateless, step_time_axis_consistent=bool(times_ok))
    return all(checks.values()), dict(checks=checks, k0_query_steps=q0, s0_query_steps=queries, expected_s0_query_steps=exp_q, s0_final_qpos_linf_vs_nominal=diff_phys, note="S0 vs c0 final-state difference is informational (S0 may genuinely change physics); no outcome label is used", s0_reset_count=spy8.resets, k0_reset_count=spy0.resets)
# ----------------------------------------------------------------------------------------------------------------------------- I4
def I4():
    import g2b_trace as TR, g2b_labeller as LB
    env, policy, lim = CTX["env"], CTX["policy"], CTX["lim"]; rows = []
    for pose in POSES:
        if rows: guard(margin=1.35 * max(r["total_cpu_s"] for r in rows))
        else: guard(margin=14.0)
        timers = dict(act_s=0.0, rec_s=0.0, step_s=0.0, n_query=0); c_start = P.cpu(); env.set_perturbation(); env.reset((pose["x"], pose["y"]), pose["yaw"]); prev = env.prev_ctrl.copy(); rec = TR.TraceRecorder(env.m, dict(id="I4", pose=pose, t0=0), refresh=True)
        box_z0 = float(env.d.xpos[rec.cmap.box_body][2]); g_prev0 = float(prev[7]); c_prefix = None
        exe1, Q1, _, _ = nominal(env, policy, lim, prev, 0, T_DEC, rec=rec, timers=timers); c_prefix = P.cpu() - c_start
        exe2, Q2, _, _ = nominal(env, policy, lim, exe1[-1], T_DEC, HORIZON, rec=rec, timers=timers); rec.record_final(env.d); c_branch = P.cpu() - c_start - c_prefix; c1 = P.cpu(); A = rec.arrays(); lab = LB.label(TR.to_labeller_trace(A, dict(t0=0, box_z0=box_z0, g_prev0=g_prev0))); c_lab = P.cpu() - c1; total = P.cpu() - c_start
        rows.append(dict(pose=pose["idx"], total_cpu_s=round(total, 3), prefix_0_40_cpu_s=round(c_prefix, 3), branch_40_160_cpu_s=round(c_branch, 3), labelling_cpu_s=round(c_lab, 3), act_cpu_s=round(timers["act_s"], 3), n_act_queries=timers["n_query"], recorder_cpu_s=round(timers["rec_s"], 3), mj_steps_cpu_s=round(timers["step_s"], 3), rows=int(len(A["g_exec"])),
                         terminal=lab["terminal"], subtype=lab["subtype"], events=lab["events"], release_phys_lag_rows=(lab["events"]["release_physical_onset"] - lab["events"]["release_command"] if "release_physical_onset" in lab["events"] and "release_command" in lab["events"] else None), grasp_lost_row=lab["events"].get("grasp_lost"), E={e: lab["endpoints"][e]["value"] for e in ("E1", "E2", "E3")}, trace_finite=bool(all(np.isfinite(v).all() for v in A.values())), neg_normal=int(A["neg_normal"].sum())))
    mean_b = float(np.mean([r["branch_40_160_cpu_s"] for r in rows])); mean_t = float(np.mean([r["total_cpu_s"] for r in rows])); ok = len(rows) == 3 and all(r["rows"] == HORIZON + 1 and r["trace_finite"] and r["neg_normal"] == 0 for r in rows)
    return ok, dict(per_pose=rows, mean_branch_40_160_cpu_s=mean_b, mean_full_episode_cpu_s=mean_t, labels_are_informational_only=True)
def main():
    OUT.mkdir(parents=True, exist_ok=True); print(f"ledger before: {PREV:.1f} s; cap {CAP_S}; guard {GUARD_S}", flush=True)
    ok = run_gate("I1", I1, 20.0)
    if ok:
        try: guard(margin=30.0); c_s = P.cpu(); CTX.pop("env_noimg", None); setup_act(); GATES.append(dict(gate="setup_act (torch import + checkpoint load + camera env)", result="DONE", cpu_s=round(P.cpu() - c_s, 2), aggregate_cpu_after_s=round(total_cpu(), 2)))
        except Exception as e: GATES.append(dict(gate="setup_act", result="FAIL", evidence=dict(exception=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-800:]))); ok = False
    for name, fn, need in (("I2", I2, 15.0), ("I3", I3, 15.0), ("I4", I4, 30.0)):
        if not ok: GATES.append(dict(gate=name, result="NOT_RUN", cpu_s=0.0, evidence="a previous gate failed or the cap guard stopped the run")); continue
        ok = run_gate(name, fn, need)
    save("ALL_PASS" if ok else "STOPPED"); print(json.dumps(dict(status="ALL_PASS" if ok else "STOPPED", aggregate_cpu_s=round(total_cpu(), 1), cap_s=CAP_S)), flush=True)
if __name__ == "__main__": main()
