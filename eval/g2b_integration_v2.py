#!/usr/bin/env python3
"""G2b INTEGRATION GATES I1-I4, v2 (X4 labeller, hardened CPU-accounted launcher).  NOT an evaluation: no treatment effect, no 7-arm branching, no training, no dataset generation.
Run through the launcher of eval/g2b_intervention_benchmark.py (aggregate CPU of the whole process tree, process-group kill, incremental atomic reports, start-up sub-limit, cooperative guard at 92 %):
  /home/user/.venvs/act/bin/python eval/g2b_integration_v2.py --cap-s <cap> --report outputs/g2b_integration/integration_v2.json --startup-milestone act_setup_done --startup-limit-s 45
Poses: the first three KNOWN G2a development poses (idx 3000-3002); no prospective G2b pose, no sealed reserve pose, no ACT seed 31-33.  Strictly sequential; stop at the first FAIL; no retry; thresholds, labels and tests are never adjusted to pass.
  I1  scripted expert on a real MuJoCo scene (<=160 steps): full per-finger trace -> X4 labeller; s[t], a[t], s[t+1] alignment proved by re-simulating from a scratch copy; labeller events compared with INDEPENDENT physical observations (raw qpos, mj_geomDistance finger-box gaps);
      truncated and invalid traces
  I2  snapshot/restore bit-exactness, c0 continuation reproduction, control history / solver state / actuator state preserved, no future-state leakage
  I3  ACT + S0: k=0 equals nominal exactly; k=8 hold, resume observation and action history, policy reset, clip->slew re-derivation
  I4  nominal c0 cost, <=3 known poses, horizon 160, component CPU (ACT, physics, trace, label, prefix) """
import os, sys, json, time, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import g2b_intervention_benchmark as BENCH
T_DEC, T_END, HORIZON, CHUNK, DT = 40, 100, 160, 20, 0.05
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
INTEGRITY_FILES = ("eval/g2b_labeller.py", "eval/g2b_trace.py", "eval/g2b_s0.py", "eval/g2b_intervention_benchmark.py", "sim/lab_env.py", "control/safety_limits.py", "policies/baseline.py", "eval/closed_loop_pilot.py", "eval/branching_replay.py", "eval/g2a_act_corrections.py")
CKPT_DIR = "outputs/checkpoints/e6b_c0_seed0/last"; CKPT_FILES = ("model.safetensors", "config.json", "norm_contract.json", "policy_preprocessor.json", "policy_postprocessor.json", "policy_preprocessor_step_3_normalizer_processor.safetensors", "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
def make_manifest(report_path, trace_base, out_path, i1_run_time_utc):
    """baseline for reusing I1: hashes of the I1 report, the saved I1 trace, every code file that defines the labeller / recorder / action semantics and the ACT checkpoint"""
    import subprocess
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    man = dict(created_for="reuse of the I1 PASS evidence", i1_run_time_utc=i1_run_time_utc, git_head=git("rev-parse", "HEAD"), i1_report=str(report_path), i1_report_sha256=sha(report_path), i1_trace_npz_sha256=sha(str(trace_base) + ".npz"), i1_trace_json_sha256=sha(str(trace_base) + ".json"),
               code={f: dict(sha256=sha(ROOT / f), last_commit_utc=git("log", "-1", "--format=%cI", "--", f)) for f in INTEGRITY_FILES}, checkpoint={f: sha(ROOT / CKPT_DIR / f) for f in CKPT_FILES}, g2a_freeze_sha256=sha(ROOT / "outputs/g2a_act_corrections/protocol_freeze.json"))
    BENCH.atomic_json(out_path, man); return man
def verify_i1_reuse(manifest_path, report_path):
    """-> (ok, reason).  I1 evidence may be reused only if the saved report is PASS (20/20 checks), the saved trace and every labeller / recorder / action-semantics / checkpoint input are byte-identical to the manifest, and the I1 trace's own sidecar hashes agree with the current labeller and recorder."""
    try:
        man = json.loads(Path(manifest_path).read_text()); rep = json.loads(Path(report_path).read_text()); g = [x for x in (rep.get("child_report") or rep)["gates"] if x["gate"] == "I1"]
        if not g or g[0]["result"] != "PASS" or not all(g[0]["evidence"]["checks"].values()): return False, "saved I1 report is not a full PASS"
        if sha(report_path) != man["i1_report_sha256"]: return False, "I1 report hash changed"
        base = ROOT / "outputs/g2b_integration/traces/I1v2_expert_p3000"
        if sha(str(base) + ".npz") != man["i1_trace_npz_sha256"] or sha(str(base) + ".json") != man["i1_trace_json_sha256"]: return False, "saved I1 trace hash changed"
        for f, v in man["code"].items():
            if sha(ROOT / f) != v["sha256"]: return False, f"code file changed: {f}"
        for f, h in man["checkpoint"].items():
            if sha(ROOT / CKPT_DIR / f) != h: return False, f"checkpoint file changed: {f}"
        if sha(ROOT / "outputs/g2a_act_corrections/protocol_freeze.json") != man["g2a_freeze_sha256"]: return False, "G2a freeze file changed"
        side = json.loads(Path(str(base) + ".json").read_text()).get("code_sha256", {})
        for f, h in side.items():
            if sha(f) != h: return False, f"I1 trace sidecar hash differs from the current {f}"
        for f, v in man["code"].items():
            if v["last_commit_utc"] and v["last_commit_utc"] > man["i1_run_time_utc"]: return False, f"{f} was committed after the I1 run"
    except Exception as e: return False, f"{type(e).__name__}: {e}"
    return True, "all inputs identical to the manifest"
def code_regions(text):
    """text of the shared helpers and of the I2 / I3 gate code (between the `# ---- <name>` marker comments); used to prove that the I2 / I3 gate code is byte-identical to the code that produced the saved evidence"""
    import re
    lines = text.split("\n"); idx = {}
    for i, l in enumerate(lines):
        m = re.match(r"    # -{20,} (helpers|I1|I2|I3|I4)\s*$", l)
        if m: idx[m.group(1)] = i
    order = ["helpers", "I1", "I2", "I3", "I4"]; return {a: "\n".join(lines[idx[a] + 1:idx[b]]) for a, b in zip(order, order[1:])}
def verify_i23_reuse(manifest_path, report_path):
    try:
        man = json.loads(Path(manifest_path).read_text()); rep = json.loads(Path(report_path).read_text()); g = {x["gate"]: x for x in (rep.get("child_report") or rep)["gates"]}
        if g.get("I2", {}).get("result") != "PASS" or g.get("I3", {}).get("result") != "PASS" or not all(g["I3"]["evidence"]["checks"].values()): return False, "saved I2/I3 evidence is not a full PASS"
        if sha(report_path) != man["v4_report_sha256"]: return False, "saved I2/I3 report hash changed"
        reg = code_regions((ROOT / "eval/g2b_integration_v2.py").read_text())
        for k, h in man["harness_regions_sha256"].items():
            if hashlib.sha256(reg[k].encode()).hexdigest() != h: return False, f"harness code region `{k}` differs from the code that produced the saved evidence"
        for f, v in man["code"].items():
            if sha(ROOT / f) != v["sha256"]: return False, f"code file changed: {f}"
        for f, h in man["checkpoint"].items():
            if sha(ROOT / CKPT_DIR / f) != h: return False, f"checkpoint file changed: {f}"
        if sha(ROOT / "outputs/g2a_act_corrections/protocol_freeze.json") != man["g2a_freeze_sha256"]: return False, "G2a freeze file changed"
    except Exception as e: return False, f"{type(e).__name__}: {e}"
    return True, "all inputs identical to the manifest"
def real_context(rep, guard):
    from types import SimpleNamespace
    sys.path[:0] = [str(ROOT), str(ROOT / "sim"), str(ROOT / "data")]; os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    import numpy as np; rep.milestone("import_numpy"); guard.check(10.0)
    import torch; torch.set_num_threads(1); rep.milestone("import_torch"); guard.check(10.0)
    import mujoco, closed_loop_pilot as P, branching_replay as BR, g2b_s0 as S0, g2b_trace as TR, g2b_labeller as LB
    from policies.baseline import ScriptedExpert
    from control.safety_limits import SafetyLimits
    rep.milestone("import_project_modules"); F = json.loads((ROOT / "outputs/g2a_act_corrections/protocol_freeze.json").read_text()); POSES = [p for p in F["poses"] if p["idx"] in (3000, 3001, 3002)]; assert [p["idx"] for p in POSES] == [3000, 3001, 3002]
    pol = P.load_chunk_policies(torch, ["c0_s0"])["c0_s0"]; rep.milestone("act_checkpoint_loaded"); guard.check(10.0)
    env = P.make_env(pol.cams); m = env.m; lim = SafetyLimits.from_model(m); rep.milestone("mujoco_env_and_camera_renderer_initialized"); rep.milestone("act_setup_done"); guard.check(10.0)
    return SimpleNamespace(np=np, mujoco=mujoco, P=P, BR=BR, S0=S0, TR=TR, LB=LB, ScriptedExpert=ScriptedExpert, env=env, m=m, lim=lim, pol=pol, POSES=POSES)
def run_gates(c, rep, guard, wanted, required, reuse):
    np, mujoco, P, BR, S0, TR, LB, ScriptedExpert = c.np, c.mujoco, c.P, c.BR, c.S0, c.TR, c.LB, c.ScriptedExpert; env, m, lim, pol, POSES = c.env, c.m, c.lim, c.pol, c.POSES; cpu = BENCH.cpu_total; d = rep.d; MS = float(os.environ.get("G2B_INT_NEED_SCALE", "1.0"))           # MS scales every guard margin; 1.0 in real runs, 0 only in the mock dry-run of the hard-cap path
    d.update(task="G2b integration gates I1-I4 v2", poses=[p["idx"] for p in POSES], contract=LB.PARAMS["CONTRACT"], labeller_params_hash=LB.params_hash(), labeller_spec_hash=LB.spec_hash(), gates=[], code_sha256={f: sha(ROOT / f) for f in ("eval/g2b_integration_v2.py", "eval/g2b_labeller.py", "eval/g2b_trace.py", "eval/g2b_s0.py", "eval/g2b_intervention_benchmark.py", "eval/test_g2b_trace_alignment.py")})
    G = {}
    def gate(name, fn, need):
        guard.check(need * MS); c0, w0 = cpu(), time.time(); ok, ev = fn(); row = dict(gate=name, source="THIS_RUN", result="PASS" if ok else "FAIL", cpu_s=round(cpu() - c0, 3), wall_s=round(time.time() - w0, 3), aggregate_cpu_after_s=round(cpu(), 3), evidence=ev); d["gates"].append(row); rep.milestone(f"gate_{name}_end", result=row["result"], gate_cpu_s=row["cpu_s"])
        if not ok: d["status"] = "STOPPED_ON_FAIL"; d["stop_reason"] = f"{name} FAIL"
        return ok
    # ------------------------------------------------------------------------------------------------------------------------- helpers
    def obs_hash(o):
        h = hashlib.sha256(np.ascontiguousarray(o["state"], dtype=np.float64).tobytes())
        for c in sorted(o["images"]): h.update(np.ascontiguousarray(o["images"][c]).tobytes())
        return h.hexdigest()
    def nominal(prev, t0, t1, rec=None, snap_at=None, timers=None, log=None):
        exe_all, Q, snap, sprev, exe = [], [], None, None, None; prev = np.asarray(prev, float).copy()
        for t in range(t0, t1):
            if (t - t0) % CHUNK == 0:
                if snap_at is not None and t == snap_at: snap = BR.snapshot(env); sprev = prev.copy()
                guard.check(4.0 * MS); c0 = cpu(); o = BR.obs_full(env); raw = pol.chunk(o["state"], o["images"]); exe, _, _ = P.slew_chain(lim, raw, prev)
                if timers is not None: timers["act_s"] += cpu() - c0; timers["n_query"] += 1
                if log is not None: log.append(dict(t=t, obs_sha=obs_hash(o), raw=raw.copy(), prev=prev.copy(), exe=exe.copy()))
            u = exe[(t - t0) % CHUNK]
            if rec is not None: c1 = cpu(); rec.record(env.d, u[7], u); timers is not None and timers.__setitem__("rec_s", timers["rec_s"] + cpu() - c1)
            c2 = cpu(); P.lean_step(env, u, False)
            if timers is not None: timers["step_s"] += cpu() - c2
            exe_all.append(u.copy()); Q.append(env.d.qpos.copy()); prev = u
        return np.array(exe_all), np.array(Q), snap, sprev
    def episode(pose, t_end, rec=None, snap_at=None, timers=None, log=None):
        env.set_perturbation(); env.reset((pose["x"], pose["y"]), pose["yaw"]); prev = env.prev_ctrl.copy(); return nominal(prev, 0, t_end, rec=rec, snap_at=snap_at, timers=timers, log=log), prev
    # ------------------------------------------------------------------------------------------------------------------------- I1
    def I1():
        ex = ScriptedExpert(m, control_hz=20); obs = P.LiveObserver(env); pose = POSES[0]; xy = (pose["x"], pose["y"]); pl = ex.plan(xy, pose["yaw"]); T = int(min(len(pl["ctrl"]), HORIZON)); env.set_perturbation(); env.reset(xy, pose["yaw"]); prev = env.prev_ctrl.copy(); g_prev0 = float(prev[7])
        rec = TR.TraceRecorder(m, dict(id="I1v2", pose=pose, t0=0), refresh=True); cm = rec.cmap; box_z0 = float(env.d.xpos[cm.box_body][2]); adr = int(m.jnt_qposadr[m.joint("blue_box_free").id]); scr, scr2, scr3 = mujoco.MjData(m), mujoco.MjData(m), mujoco.MjData(m); fromto = np.zeros(6)
        noninterf, d_ref, hist_d, sub_checks, align, gapL, gapR = [], [], [], [], [], [], []
        for t in range(T):
            guard.check(6.0 * MS); c, _, _ = lim.clip_and_slew(pl["ctrl"][t], prev, DT); dd = env.d; before = (dd.qpos.copy(), dd.qvel.copy(), dd.qacc_warmstart.copy(), dd.ctrl.copy(), float(dd.time))
            rec.record(dd, c[7], c); noninterf.append(bool(all(np.array_equal(a, b) for a, b in zip(before[:4], (dd.qpos, dd.qvel, dd.qacc_warmstart, dd.ctrl))) and before[4] == float(dd.time))); d_ref.append(float(np.abs(rec.rows["box_pos"][-1] - dd.qpos[adr:adr + 3]).max()))
            mujoco.mj_copyData(scr3, m, dd); mujoco.mj_forward(m, scr3)                                          # end-of-step state, scratch copy: independent finger-box gaps
            gapL.append(float(mujoco.mj_geomDistance(m, scr3, cm.fL, cm.box, 0.05, fromto))); gapR.append(float(mujoco.mj_geomDistance(m, scr3, cm.fR, cm.box, 0.05, fromto)))
            if t % 2 == 0: rr = cm.read_contacts(dd); b_, c_ = obs.row(); hist_d.append(max(abs((rr["FL"] + rr["FR"]) - c_[0] * 50.0), abs((rr["nL"] + rr["nR"]) - c_[1]), abs(rr["Fh"] - c_[4])))
            chk = (t % 5 == 0)
            if chk:                                                                                                                                         # s[t], a[t] -> s[t+1]: re-simulate from a scratch copy with the CORRECT command a[t] and with the previous command a[t-1]
                mujoco.mj_copyData(scr, m, dd); scr.ctrl[:] = c
                for _ in range(env.substeps): mujoco.mj_step(m, scr)
                mujoco.mj_copyData(scr2, m, dd); scr2.ctrl[:] = prev
                for _ in range(env.substeps): mujoco.mj_step(m, scr2)
                exp_q, wrong_q, exp_fin = scr.qpos.copy(), scr2.qpos.copy(), float(scr.qpos[cm.finger_qadr])
            v = float(np.linalg.norm(dd.qvel[cm.box_dof:cm.box_dof + 3]))
            if v > 0.10 and len(sub_checks) < 3:
                mujoco.mj_copyData(scr, m, dd); x0 = scr.qpos[adr:adr + 3].copy(); v0 = scr.qvel[cm.box_dof:cm.box_dof + 3].copy(); mujoco.mj_step(m, scr); x1 = scr.qpos[adr:adr + 3].copy(); v1 = scr.qvel[cm.box_dof:cm.box_dof + 3].copy(); fd = (x1 - x0) / m.opt.timestep; mean = 0.5 * (v0 + v1); sub_checks.append(float(np.linalg.norm(fd - mean) / np.linalg.norm(mean)))
            P.lean_step(env, c, False)
            if chk: align.append(dict(t=t, correct_equal=bool(np.array_equal(env.d.qpos, exp_q)), wrong_differs=bool(np.abs(wrong_q - env.d.qpos).max() > 1e-9) if not np.array_equal(c, prev) else None, finger_next=exp_fin, row=t + 1))
            prev = c
        mujoco.mj_copyData(scr3, m, env.d); mujoco.mj_forward(m, scr3); gapL.append(float(mujoco.mj_geomDistance(m, scr3, cm.fL, cm.box, 0.05, fromto))); gapR.append(float(mujoco.mj_geomDistance(m, scr3, cm.fR, cm.box, 0.05, fromto)))      # final row (record_final adds row T): the gap lists must have N = T + 1 entries like the trace
        guard.check(0.0); rec.record_final(env.d); A = rec.arrays(); meta = dict(t0=0, box_z0=box_z0, g_prev0=g_prev0); lt = TR.to_labeller_trace(A, meta); lab = LB.label(lt); ev = lab["events"]; N = len(A["g_exec"]); assert N == T + 1 and len(gapL) == len(gapR) == N and all(len(v) == N for v in A.values()), f"trace schema: T={T} actions must give T+1={T + 1} observation rows in every field and in the independent gap series; got N={N}, gaps={len(gapL)}/{len(gapR)}"; full = {e: lab["endpoints"][e]["value"] for e in ("E1", "E2", "E3")}
        TR.save_trace(ROOT / "outputs/g2b_integration/traces/I1v2_expert_p3000", A, dict(id="I1v2", pose=pose, t0=0, g_prev0=g_prev0, box_z0=box_z0, note="integration gate; scripted expert; not an evaluation"), label=lab, code_files=[ROOT / "eval/g2b_trace.py", ROOT / "eval/g2b_labeller.py"])
        # ---- independent physical observations (raw qpos / geometry), compared with the labeller's events
        gm, ge, box = A["g_meas"], A["g_exec"], A["box_pos"]; rise = box[:, 2] - box_z0; first_cmd = int(np.argmax(ge < ge[0] - 0.001)); phys_close = int(np.argmax(gm < gm[0] - 0.001))
        touch = (np.array(gapL) <= 0.0005) & (np.array(gapR) <= 0.0005); gap_contact = None
        for s in range(0, N - 3):
            if touch[s:s + 3].all(): gap_contact = s; break
        lift_row = None
        for s in range(0, N - 2):
            if (rise[s:s + 3] >= LB.PARAMS["R_LIFT"]).all(): lift_row = s; break
        cmd = ev.get("release_command"); sep = None
        if cmd is not None:
            free = (np.array(gapL) > 0.0005) & (np.array(gapR) > 0.0005)
            for s in range(cmd + 1, N - 2):
                if free[s:s + 3].all(): sep = s; break
        pm = np.minimum.accumulate(gm[ev["grasp_acquired"]:]) if "grasp_acquired" in ev else None; po = None
        if pm is not None:
            rs = gm[ev["grasp_acquired"]:] - pm
            for s in range(0, len(rs) - 1):
                if (rs[s:s + 2] >= LB.PARAMS["OPEN_RISE"]).all(): po = ev["grasp_acquired"] + s; break
        # truncation / invalid behaviour including the X4 invariants
        bad_cut = []
        for n in list(range(6, N, 4)) + [N - 1]:
            cut = dict(lt); cut["meta"] = dict(lt["meta"])
            for k in ("g_exec", "g_meas", "nL", "nR", "FL", "FR", "box_pos", "box_vel", "Fh"): cut[k] = lt[k][:n]
            o = LB.label(cut); e = o["events"]
            for k in ("E1", "E2", "E3"):
                if o["endpoints"][k]["value"] not in (full[k], "CENSORED"): bad_cut.append((n, k, o["endpoints"][k]["value"], full[k]))
            if "release_confirmed" in e and e["release_confirmed"] > n - 1: bad_cut.append((n, "verified beyond observed", e["release_confirmed"]))
            if o["e3_decision_row"] is not None and o["e3_decision_row"] > n - 1: bad_cut.append((n, "decision beyond observed", o["e3_decision_row"]))
            if (o["endpoints"]["E3"]["value"] == "TRUE") != (o["terminal"] == "PLACEMENT_SUCCESS"): bad_cut.append((n, "E3 vs terminal"))
        nan = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in lt.items()}; nan["FL"][5] = np.nan; nan2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in lt.items()}; nan2["g_meas"][7] = np.nan; nog = dict(lt); nog.pop("g_meas")
        inv = [LB.label(nan)["terminal"], LB.label(dict(lt, meta=dict(lt["meta"], valid=False)))["terminal"], LB.label(nan2)["terminal"], LB.label(nog)["terminal"]]
        fin_ok = None
        if lab["terminal"] == "PLACEMENT_SUCCESS":                                                                                                            # observable final state must be consistent with a success label
            dxy = float(np.linalg.norm(box[-1, :2] - np.array([LB.PARAMS["TRAY_X"], LB.PARAMS["TRAY_Y"]]))); held_last = bool(A["nL"][-1] >= 1 and A["nR"][-1] >= 1 and A["FL"][-1] >= 2.5 and A["FR"][-1] >= 2.5)
            fin_ok = bool(dxy < 0.03 and abs(box[-1, 2] - LB.PARAMS["REST_Z"]) < 0.006 and float(np.linalg.norm(A["box_vel"][-1])) < 0.02 and not held_last)
        free_arr = (np.array(gapL) > 0.0005) & (np.array(gapR) > 0.0005); dxy_all = np.linalg.norm(box[:, :2] - np.array([LB.PARAMS["TRAY_X"], LB.PARAMS["TRAY_Y"]]), axis=1); spd = np.linalg.norm(A["box_vel"], axis=1)
        quiet = (dxy_all < LB.PARAMS["PLACE_XY"]) & (np.abs(box[:, 2] - LB.PARAMS["REST_Z"]) < LB.PARAMS["PLACE_Z"]) & (spd < LB.PARAMS["REST_V"]) & free_arr; rv_ = ev.get("release_confirmed"); ind_window = bool(rv_ is not None and any(quiet[s_:s_ + 10].all() for s_ in range(rv_, N - 9)))
        label_window_ok = bool("placement_settled" not in ev or quiet[ev["placement_settled"]:ev["placement_settled"] + 10].all()); consistent = bool((lab["terminal"] == "PLACEMENT_SUCCESS") == ind_window and label_window_ok)
        wr = [a["wrong_differs"] for a in align if a["wrong_differs"] is not None]; fin_match = all(abs(A["g_meas"][a["row"]] - a["finger_next"]) == 0.0 for a in align)
        checks = dict(events_present_ordered=bool(all(k in ev for k in ("close_command", "grasp_acquired", "lift_confirmed", "release_command", "release_physical_onset", "release_confirmed_onset", "release_confirmed", "placement_settled")) and ev["close_command"] < ev["grasp_acquired"] < ev["lift_confirmed"] <= ev["release_command"] < ev["release_physical_onset"] <= ev["release_confirmed_onset"] < ev["release_confirmed"] <= ev["placement_settled"] and ev["target_arrival"] <= ev["release_command"]),
                      E1_TRUE_E2_not_FALSE=bool(full["E1"] == "TRUE" and full["E2"] != "FALSE"), labeller_consistent_with_independent_quiet_window_after_verified_release=bool(consistent), verification_is_onset_plus_K_minus_1=bool(ev["release_confirmed"] == ev["release_confirmed_onset"] + LB.PARAMS["K_REL"] - 1), decision_row_ok=bool(lab["e3_decision_row"] == ev["placement_settled"] + LB.PARAMS["K_SETTLE"] - 1 <= N - 1),
                      alignment_next_state_equals_resimulation_with_a_t=bool(all(a["correct_equal"] for a in align) and len(align) >= 10), alignment_wrong_command_a_tminus1_differs=bool(len(wr) >= 3 and all(wr)), alignment_recorded_finger_row_tplus1_equals_resimulated=bool(fin_match),
                      command_precedes_physical_closing=bool(phys_close >= first_cmd + 1), labeller_acquisition_after_independent_finger_contact=bool(gap_contact is not None and ev["grasp_acquired"] >= gap_contact and ev["grasp_acquired"] - gap_contact <= 12), labeller_lift_equals_independent_box_rise=bool(lift_row == ev["lift_confirmed"]),
                      labeller_physical_onset_equals_independent=bool(po == ev["release_physical_onset"]), independent_finger_separation_between_command_and_verification=bool(sep is not None and ev["release_command"] < sep <= ev["release_confirmed_onset"] and ev["release_confirmed_onset"] - sep <= 10), final_observable_state_consistent_with_success=bool(fin_ok) if fin_ok is not None else True,
                      refresh_does_not_touch_live_state=bool(all(noninterf)), refreshed_box_equals_qpos_1e9=bool(max(d_ref) < 1e-9), historical_observer_equal_1e5=bool(max(hist_d) < 1e-5), substep_qvel_identity_median_2pct=bool(len(sub_checks) > 0 and float(np.median(sub_checks)) < 0.02), no_neg_normal=bool(int(A["neg_normal"].sum()) == 0),
                      truncation_and_X4_invariants=bool(not bad_cut), invalid_traces_nan_meta_gmeas=bool(inv == ["INVALID"] * 4))
        sep_info = dict(first_commanded_departure_row=first_cmd, close_command_row=ev.get("close_command"), first_physical_closing_row=phys_close, independent_finger_contact_row=gap_contact, grasp_acquired_row=ev.get("grasp_acquired"), independent_lift_row=lift_row, lift_confirmed_row=ev.get("lift_confirmed"), release_command_row=cmd, independent_finger_separation_row=sep,
                        release_physical_onset_row=ev.get("release_physical_onset"), independent_physical_onset_row=po, release_confirmed_onset_row=ev.get("release_confirmed_onset"), release_confirmed_row=ev.get("release_confirmed"), placement_settled_row=ev.get("placement_settled"), e3_decision_row=lab["e3_decision_row"], rows=N)
        return all(checks.values()), dict(pose=pose["idx"], steps=T, plan_len=int(len(pl["ctrl"])), terminal=lab["terminal"], endpoints=full, events=ev, row_comparison=sep_info, checks=checks, truncation_violations=bad_cut[:5], invalid_terminals=inv,
                                           alignment=dict(n_checks=len(align), lag1_vs_lag0_note="a[t-1] instead of a[t] changes s[t+1] in %d of %d command-change checks" % (sum(wr), len(wr))), sample_timing=dict(refreshed_minus_qpos_max_m=max(d_ref), hist_observer_max_diff=max(hist_d), substep_qvel_rel_diff=sub_checks, max_gap_L_m=max(gapL), min_gap_L_m=min(gapL), max_box_speed_ms=float(np.linalg.norm(A["box_vel"], axis=1).max())))
    # ------------------------------------------------------------------------------------------------------------------------- I2
    def I2():
        pose = POSES[0]; rec = TR.TraceRecorder(m, dict(id="I2v2", pose=pose, t0=0), refresh=True); logA = []
        (exeA, QA, snap, sprev), _ = episode(pose, T_END, rec=rec, snap_at=T_DEC, log=logA); G.update(snap=snap, sprev=sprev, exeA=exeA, QA=QA, logA=logA, pose=pose); sq = snap["d"]; snap_copy = (sq.qpos.copy(), sq.qvel.copy(), sq.qacc_warmstart.copy(), sq.ctrl.copy(), float(sq.time), snap["prev"].copy(), snap["t"]); res = []
        q40_cont = QA[T_DEC - 1]; leak_free = bool(np.array_equal(snap["d"].qpos, q40_cont))                                                                     # snapshot state == continuous-run state after step 39 (taken BEFORE step 40 was applied)
        for rep_i in range(2):
            BR.restore(env, snap); e = env.d; restored = dict(qpos=bool(np.array_equal(e.qpos, snap_copy[0])), qvel=bool(np.array_equal(e.qvel, snap_copy[1])), qacc_warmstart=bool(np.array_equal(e.qacc_warmstart, snap_copy[2])), ctrl=bool(np.array_equal(e.ctrl, snap_copy[3])), time=bool(float(e.time) == snap_copy[4]), prev_ctrl=bool(np.array_equal(env.prev_ctrl, snap_copy[5])), env_t=bool(env.t == snap_copy[6]))
            logB = []; exeB, QB, _, _ = nominal(sprev, T_DEC, T_END, log=logB); res.append(dict(rep=rep_i, restored_state=restored, exe_equal=bool(np.array_equal(exeB, exeA[T_DEC:])), qpos_equal=bool(np.array_equal(QB, QA[T_DEC:])), max_abs_exe_diff=float(np.abs(exeB - exeA[T_DEC:]).max()), max_abs_qpos_diff=float(np.abs(QB - QA[T_DEC:]).max()),
                                                                                  query_inputs_equal=bool([x["obs_sha"] for x in logB] == [x["obs_sha"] for x in logA if x["t"] >= T_DEC]), raw_chunks_equal=bool(all(np.array_equal(x["raw"], y["raw"]) for x, y in zip(logB, [x for x in logA if x["t"] >= T_DEC])))))
            G[f"exeB{rep_i}"], G[f"QB{rep_i}"] = exeB, QB
        intact = bool(np.array_equal(sq.qpos, snap_copy[0]) and np.array_equal(sq.qvel, snap_copy[1]) and np.array_equal(sq.qacc_warmstart, snap_copy[2]) and np.array_equal(sq.ctrl, snap_copy[3]) and float(sq.time) == snap_copy[4])
        ok = leak_free and intact and bool(np.array_equal(G["exeB0"], G["exeB1"])) and all(all(r["restored_state"].values()) and r["exe_equal"] and r["qpos_equal"] and r["query_inputs_equal"] and r["raw_chunks_equal"] for r in res)
        return ok, dict(pose=pose["idx"], restores=res, snapshot_equals_continuous_state_before_step_40=leak_free, snapshot_not_mutated=intact, restore0_equals_restore1=bool(np.array_equal(G["exeB0"], G["exeB1"])), recorder_attached_run_equals_recorder_free_restores=bool(ok), n_steps_branch=int(len(G["exeB0"])), first_query_after_restore_input_sha_equals_continuous=bool(res[0]["query_inputs_equal"]))
    # ------------------------------------------------------------------------------------------------------------------------- I3
    def I3():
        snap, sprev = G["snap"], G["sprev"]; actq = []
        class Spy:
            def __init__(s, p): s.p = p; s.calls = []; s.resets = 0; s.policy = p.policy; s._o = p.policy.reset; p.policy.reset = s._r
            def _r(s): s.resets += 1; s._o()
            def chunk(s, st, im): r = s.p.chunk(st, im); s.calls.append(dict(state=np.array(st, float).copy(), raw=r.copy())); return r
        def branch(k):
            BR.restore(env, snap); spy = Spy(pol); log, Q = [], []
            class Ad:
                def step(s, u): P.lean_step(env, u, False); Q.append(env.d.qpos.copy())
                def observe(s): return BR.obs_full(env)
            states = {}
            def on_step(t, u, role): log.append((t, role, np.asarray(u, float).copy(), float(env.t))); states[t] = env.adapter.state(env.d).astype(float).copy()
            ex_s, lg = S0.run_branch(Ad(), spy, lim, sprev, T_DEC, T_END, k=k, on_step=on_step); pol.policy.reset = spy._o; return ex_s, lg, spy, log, np.array(Q), states
        exe0, lg0, spy0, log0, Q0, st0 = branch(0); exe8, lg8, spy8, log8, Q8, st8 = branch(8); inner = getattr(pol.policy, "_action_queue", None)
        holds = [u for t, r, u, _ in log8 if r == "hold"]; q8 = [t for t, r in lg8 if r == "query"]; exp_q = S0.s0_query_steps(T_DEC, T_END)
        # clip->slew re-derivation: executed chunk after each query == slew_chain(raw, previous executed command)
        def rederive(log, spy, steps):
            ok = True; calls = iter(spy.calls)
            for q in steps:
                c = next(calls); prev_u = [u for t, r, u, _ in log if t == q - 1]; prev_u = prev_u[0] if prev_u else sprev; exp_exe, _, _ = P.slew_chain(lim, c["raw"], prev_u); got = np.array([u for t, r, u, _ in log if q <= t < q + CHUNK and r == "act"]); ok &= bool(np.array_equal(got, exp_exe[:len(got)]))
            return ok
        slew_ok0 = rederive(log0, spy0, [40, 60, 80]); slew_ok8 = rederive(log8, spy8, exp_q); q_state_ok = bool(all(np.array_equal(c["state"], st8[q]) for c, q in zip(spy8.calls, exp_q))); q_state_ok0 = bool(all(np.array_equal(c["state"], st0[q]) for c, q in zip(spy0.calls, [40, 60, 80])))
        first_resumed = [u for t, r, u, _ in log8 if t == T_DEC + 8][0]; held_last = holds[-1]; step_lim = np.abs(first_resumed - held_last)
        o = BR.obs_full(env); a1 = pol.chunk(o["state"], o["images"]); a2 = pol.chunk(o["state"], o["images"]); times_ok = all(abs(tt - (snap["t"] + (t - T_DEC) * DT)) < 1e-9 for t, _, _, tt in log8)
        checks = dict(k0_exe_equals_nominal_exactly=bool(np.array_equal(exe0, G["exeB0"])), k0_qpos_equals_nominal_exactly=bool(np.array_equal(Q0, G["QB0"])), k0_query_steps_40_60_80=bool([t for t, r in lg0 if r == "query"] == [40, 60, 80]), k0_no_policy_reset=bool(spy0.resets == 0), s0_hold_8_identical_to_last_target=bool(len(holds) == 8 and all(np.array_equal(h, sprev) for h in holds)),
                      s0_query_steps_48_68_88=bool(q8 == exp_q == [48, 68, 88]), s0_query_observation_equals_live_state_at_query=q_state_ok, k0_query_observation_equals_live_state=q_state_ok0, s0_policy_reset_called_once=bool(spy8.resets == 1), s0_executed_commands_equal_rederived_clip_slew=bool(slew_ok8), k0_executed_commands_equal_rederived_clip_slew=bool(slew_ok0), s0_commands_legal=bool(P.check_legal(lim, exe8, sprev)),
                      s0_first_resumed_command_continues_from_held_command_within_slew_limit=bool(np.all(step_lim[:7] <= lim.max_speed[:7] * DT * (1 + 1e-9) + 1e-9)), policy_stateless_same_obs_same_chunk=bool(np.array_equal(a1, a2)), step_time_axis_consistent=bool(times_ok), policy_action_queue_empty_after_reset=bool(inner is None or len(inner) == 0))
        return all(checks.values()), dict(checks=checks, s0_query_steps=q8, k0_query_steps=[t for t, r in lg0 if r == "query"], s0_reset_count=spy8.resets, s0_final_qpos_linf_vs_nominal=float(np.abs(Q8[-1] - Q0[-1]).max()), note="the S0-vs-c0 final-state difference is informational (S0 may genuinely change the physics); no outcome label is used", action_queue_attr_present=inner is not None)
    # ------------------------------------------------------------------------------------------------------------------------- I4
    def nominal_i4(prev, t0, t1, rec, T, prog):
        """I4 reference loop (a copy of `nominal` with per-chunk progress reporting; I2/I3 keep using `nominal` unchanged)"""
        exe_all, exe = [], None; prev = np.asarray(prev, float).copy()
        for t in range(t0, t1):
            if (t - t0) % CHUNK == 0:
                guard.check(4.0 * MS); c0 = cpu(); o = BR.obs_full(env); raw = pol.chunk(o["state"], o["images"]); exe, _, _ = P.slew_chain(lim, raw, prev); T["act_s"] += cpu() - c0; T["n_query"] += 1
            u = exe[(t - t0) % CHUNK]; c1 = cpu(); rec.record(env.d, u[7], u); T["rec_s"] += cpu() - c1; c2 = cpu(); P.lean_step(env, u, False); T["step_s"] += cpu() - c2; exe_all.append(u.copy()); T["steps"] += 1; prev = u
            if (t - t0) % CHUNK == CHUNK - 1: prog.update(completed_physics_steps=T["steps"], act_queries=T["n_query"], act_cpu_s=round(T["act_s"], 3), physics_cpu_s=round(T["step_s"], 3), recorder_cpu_s=round(T["rec_s"], 3), aggregate_cpu_s=round(cpu() - T["c_start"], 3), wall_s=round(time.time() - T["w_start"], 3)); rep.write()
        return np.array(exe_all)
    def I4():
        rows = d.setdefault("i4_poses", []); d["i4_planned_poses"] = [p["idx"] for p in POSES]; d["i4_in_progress"] = None; c_gate, w_gate = cpu(), time.time(); chash = {k: d["code_sha256"][k] for k in ("eval/g2b_integration_v2.py", "eval/g2b_labeller.py")}
        try:
            for pose in POSES:
                done = [r["aggregate_cpu_s"] for r in rows if r["status"] == "COMPLETE"]; guard.check((1.4 * max(done) if done else 14.0) * MS)
                T = dict(act_s=0.0, rec_s=0.0, step_s=0.0, n_query=0, steps=0, c_start=cpu(), w_start=time.time()); prog = dict(pose=pose["idx"], status="IN_PROGRESS", completed_physics_steps=0, act_queries=0, setup_cpu_s=0.0, act_cpu_s=0.0, physics_cpu_s=0.0, recorder_cpu_s=0.0, labeller_cpu_s=0.0, aggregate_cpu_s=0.0, wall_s=0.0, validity=None, code_sha256=chash)
                d["i4_in_progress"] = prog; rep.write(); c_set = cpu(); env.set_perturbation(); env.reset((pose["x"], pose["y"]), pose["yaw"]); prev = env.prev_ctrl.copy(); rec = TR.TraceRecorder(m, dict(id="I4v5", pose=pose, t0=0), refresh=True); box_z0 = float(env.d.xpos[rec.cmap.box_body][2]); g_prev0 = float(prev[7]); prog["setup_cpu_s"] = round(cpu() - c_set, 3)
                c0_ = cpu(); exe1 = nominal_i4(prev, 0, T_DEC, rec, T, prog); c_prefix = cpu() - c0_; exe2 = nominal_i4(exe1[-1], T_DEC, HORIZON, rec, T, prog); c_branch = cpu() - c0_ - c_prefix; rec.record_final(env.d)
                c1 = cpu(); A = rec.arrays(); lab = LB.label(TR.to_labeller_trace(A, dict(t0=0, box_z0=box_z0, g_prev0=g_prev0))); c_lab = cpu() - c1; total = cpu() - T["c_start"]
                valid = dict(rows_161=bool(len(A["g_exec"]) == HORIZON + 1), steps_160=bool(T["steps"] == HORIZON), trace_finite=bool(all(np.isfinite(v).all() for v in A.values())), no_neg_normal=bool(int(A["neg_normal"].sum()) == 0), label_not_invalid=bool(lab["terminal"] != "INVALID")); valid["all_valid"] = bool(all(valid.values()))
                rec_row = dict(pose=pose["idx"], status="COMPLETE" if valid["all_valid"] else "INVALID", completed_physics_steps=int(T["steps"]), act_queries=int(T["n_query"]), setup_cpu_s=prog["setup_cpu_s"], prefix_0_40_cpu_s=round(c_prefix, 3), branch_40_160_cpu_s=round(c_branch, 3), act_cpu_s=round(T["act_s"], 3), physics_cpu_s=round(T["step_s"], 3), recorder_cpu_s=round(T["rec_s"], 3), labeller_cpu_s=round(c_lab, 3),
                               aggregate_cpu_s=round(total, 3), wall_s=round(time.time() - T["w_start"], 3), validity=valid, code_sha256=chash, terminal=lab["terminal"], events=lab["events"], informational_only="labels are not used for any decision")
                rows.append(rec_row); d["i4_in_progress"] = None; rep.write()                                                                                 # written immediately when the pose completes
        except BaseException as e:
            part = d.get("i4_in_progress"); d["gates"].append(dict(gate="I4", source="THIS_RUN", result="PARTIAL", cpu_s=round(cpu() - c_gate, 3), wall_s=round(time.time() - w_gate, 3), aggregate_cpu_after_s=round(cpu(), 3), evidence=dict(planned_poses=d["i4_planned_poses"], completed_poses=len(rows), per_pose=list(rows), interrupted_pose=part, reason=f"{type(e).__name__}: {e}")))
            rep.milestone("gate_I4_interrupted", gate_cpu_s=round(cpu() - c_gate, 3), completed_poses=len(rows)); raise
        ok = len(rows) == len(POSES) == 3 and all(r["status"] == "COMPLETE" for r in rows)
        return ok, dict(planned_poses=d["i4_planned_poses"], completed_poses=len(rows), per_pose=rows, mean_aggregate_cpu_s=float(np.mean([r["aggregate_cpu_s"] for r in rows])), mean_branch_40_160_cpu_s=float(np.mean([r["branch_40_160_cpu_s"] for r in rows])), mean_prefix_cpu_s=float(np.mean([r["prefix_0_40_cpu_s"] for r in rows])), labels_informational_only=True)
    specs = dict(I1=(I1, 14.0), I2=(I2, 10.0), I3=(I3, 8.0), I4=(I4, 14.0)); d["required_gates"] = list(required); d["gates_requested_this_run"] = list(wanted)
    for ru in (reuse or []):
        ok_r, why = (verify_i1_reuse if ru["kind"] == "I1" else verify_i23_reuse)(ru["manifest"], ru["report"])
        if not ok_r: d["status"] = "BLOCKED"; d["stop_reason"] = f"{ru['kind']} reuse is no longer justified: {why}"; rep.milestone(f"reuse_{ru['kind']}_rejected", reason=why); return
        rr = json.loads(Path(ru["report"]).read_text()); allg = (rr.get("child_report") or rr)["gates"]
        for gname in ru["gates"]:
            row = [g for g in allg if g["gate"] == gname]
            if not row or row[0]["result"] != "PASS": d["status"] = "BLOCKED"; d["stop_reason"] = f"saved {gname} evidence is not a PASS"; return
            d["gates"].append(dict(row[0], source="REUSED_FROM_SAVED_REPORT", reuse_manifest=str(ru["manifest"]), reuse_integrity="verified"))
        rep.milestone(f"reuse_{ru['kind']}_verified", reused_gates=",".join(ru["gates"]))
    for name in wanted:
        fn, need = specs[name]
        if not gate(name, fn, need): return
    passed = {g["gate"] for g in d["gates"] if g["result"] == "PASS"}; missing = [r for r in required if r not in passed]
    if missing: d["status"] = "PARTIAL"; d["stop_reason"] = f"required gate(s) without PASS evidence: {missing}"; return
    d["status"] = "COMPLETE"
def parse_env():
    wanted = [g for g in os.environ.get("G2B_INT_GATES", "I1,I2,I3,I4").split(",") if g]; assert all(g in ("I1", "I2", "I3", "I4") for g in wanted), wanted
    reuse = json.loads(os.environ["G2B_INT_REUSE"]) if os.environ.get("G2B_INT_REUSE") else []
    return wanted, ["I1", "I2", "I3", "I4"], reuse
def workload(rep, guard, args):
    wanted, required, reuse = parse_env(); ctx = real_context(rep, guard); run_gates(ctx, rep, guard, wanted, required, reuse)
if __name__ == "__main__":
    if "--child" in sys.argv:
        if os.environ.get("G2B_INT_DRY"):
            import g2b_integration_dryrun as DRY; BENCH.real = DRY.dry_workload
        else: BENCH.real = workload
        sys.exit(BENCH.child_main(BENCH.parse()))
    BENCH.__file__ = os.path.abspath(__file__); sys.exit(BENCH.launcher_main(BENCH.parse()))
