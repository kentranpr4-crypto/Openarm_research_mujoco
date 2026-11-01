#!/usr/bin/env python3
"""OFFLINE BRANCHING REPLAY (DEVELOPMENT; privileged oracle observer; simulation only; NOT a closed-loop evaluation and NOT Sim2Real).
For each of the 24 registered Stage-1 poses the default ACT (c0_seed0) is replayed (A0); at each decision state t = 20,40,...,180 the full MuJoCo state (mj_copyData) and the control history are snapshotted, the three frozen C0 candidates (c0_s0 = default, c0_s1, c0_s2) are generated from the SAME
observation, executed as clip->slew chains from the last executed target, scored by the frozen WM3-B + frozen J (J20 = row t+20) BEFORE any branch is executed, then each candidate is executed from the restored snapshot to obtain its REALISED J20.  The realised outcomes are labels for evaluating the ranking only;
nothing is selected or executed on their basis (the replay continues with the default candidate).  J2 (optional, t<=160): prediction = WM autoregressive 2-block rollout of [candidate, hold-last-executed-target for 20 steps]; label = realised J over the same 40 steps (hold continuation); J2 uses no future observation.
Budget: whole experiment <= 0.40 process CPU-h (outputs/branching_replay/cpu_ledger.json).  Run (ACT venv):  python eval/branching_replay.py run"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import sys, json, time, hashlib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "eval"), str(ROOT / "data")): sys.path.insert(0, p)
import closed_loop_pilot as P            # Stage-1 helpers (frozen, unmodified): policies, WM, observer, windows, slew chain, lean step
import decision_utility_lib as L
OUT = ROOT / "outputs/branching_replay"; FREEZE = OUT / "protocol_freeze.json"; LEDGER = OUT / "cpu_ledger.json"
CAP_TOTAL_S = 0.40 * 3600; STATES = list(range(20, 181, 20)); J2_STATES = [t for t in STATES if t <= 160]; CK = P.CK_NAMES; NB = len(CK)
def ledger_used(): return float(sum(e["cpu_s"] for e in json.loads(LEDGER.read_text()))) if LEDGER.exists() else 0.0
def ledger_add(tag, cpu_s):
    Lg = json.loads(LEDGER.read_text()) if LEDGER.exists() else []; Lg.append(dict(tag=tag, cpu_s=float(cpu_s), at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))); P.atomic_json(LEDGER, Lg)
# ------------------------------------------------------------------------------------------------------------------------------------------ snapshot / restore / execution
def snapshot(env):
    import mujoco; d2 = mujoco.MjData(env.m); mujoco.mj_copyData(d2, env.m, env.d); return dict(d=d2, prev=np.asarray(env.prev_ctrl, float).copy(), t=float(env.t))
def restore(env, snap):
    import mujoco; mujoco.mj_copyData(env.d, env.m, snap["d"]); env.prev_ctrl = snap["prev"].copy(); env.t = snap["t"]
def obs_full(env): return dict(state=env.adapter.state(env.d), velocity=env.adapter.velocity(env.d), t=env.t, images={c: env.rig.rgb(env.d, c) for c in env.cams})
def exec_chunk(env, obs_, ctrl_seq, final_image=False):
    """Execute len(ctrl_seq) steps from the current env state; returns rows at the START of each step plus one final row (n+1 rows each) and the applied-ctrl audit.  Row definition identical to the Stage-1 runner / wm1_sim."""
    n = len(ctrl_seq); S, B, C, BX, TCP, applied = [], [], [], [], [], []
    for j in range(n):
        i = env.info(); b, c = obs_.row(); S.append(env.adapter.state(env.d).astype(np.float32)); B.append(b); C.append(c); BX.append(i["box_pos"]); TCP.append(i["tcp_pos"])
        P.lean_step(env, ctrl_seq[j], False); applied.append(np.abs(np.asarray(env.d.ctrl, float) - np.asarray(ctrl_seq[j], float)).max())
    i = env.info(); b, c = obs_.row(); S.append(env.adapter.state(env.d).astype(np.float32)); B.append(b); C.append(c); BX.append(i["box_pos"]); TCP.append(i["tcp_pos"])
    o = obs_full(env) if final_image else None
    return dict(S=np.array(S), B=np.array(B), C=np.array(C), box=np.array(BX), tcp=np.array(TCP), applied_ctrl_maxdiff=float(max(applied)) if applied else 0.0), o
def realised(box, c_rows):
    """J on EXECUTED rows (same formula/terms as the frozen J): box (R,3), c (R,7)."""
    b = np.zeros((len(box), 7)); b[:, :3] = box; c = np.zeros((len(c_rows), 7)); c[:, 4] = np.asarray(c_rows)[:, 4]; tm = L.terms(b, c); return tm, float(L.score(tm))
def terms_dict(tm): return dict(dxy=float(tm["dxy"]), dz=float(tm["dz"]), palm=bool(tm["palm"]), lifted=bool(tm["lifted"]))
def score_state(w3, hist, t, exe, pose):
    """Predicted J20 terms/values and the J2 rollout, from rows <= t and actions <= t-1 plus the candidate sequences only."""
    S_rows, B_rows, C_rows, A_exec = hist["S"], hist["B"], hist["C"], hist["A"]
    W = P.build_window(S_rows, B_rows, C_rows, A_exec, t, exe, pose); terms, J = P.score_candidates(w3, L, W, np.asarray(B_rows[t]), np.asarray(C_rows[t])); out = dict(J20=[float(v) for v in J], terms=[terms_dict(q) for q in terms])
    if t in J2_STATES:
        hold = np.repeat(exe[:, -1:, :8], 20, axis=1); A = np.concatenate([W["A_hist"], np.asarray(exe[:, :, :8], np.float32), hold.astype(np.float32)], axis=1)           # (n, 4+20+20, 8): history, candidate, hold
        hs = dict(S=W["S_hist"].copy(), B=W["B_hist"].copy(), C=W["C_hist"].copy()); R = L.rollout(w3.predict, hs, A, np.full(len(exe), A.shape[1] + 1), pose, t0=4)
        j2 = []; t2 = []
        for q in R: tm = L.terms(q["b"], q["c"]); j2.append(float(L.score(tm))); t2.append(terms_dict(tm))
        out.update(J2=j2, terms2=t2, rollout_rows=[len(q["b"]) for q in R])
    return out, W
def phase_label(hist, t):
    B = np.array(hist["B"][:t + 1]); C = np.array(hist["C"][:t + 1]); import lab_scene
    return P.phase_of(float(C[:, 1].sum()), float((B[:, 2] - B[0, 2]).max()), float(np.linalg.norm(B[t, :2] - np.asarray(lab_scene.TRAY_C))))
# ------------------------------------------------------------------------------------------------------------------------------------------ one pose
def process_pose(pid, G, guard, states=STATES, do_j2=False, keep_clones=None, out_arrays=None, on_record=None):
    """Replay A0 for pose pid, branching at every decision state in `states`.  Returns the list of state records (+ clones for the optional J2 phase)."""
    env, obs_, policies, w3, lim = G["env"], G["obs"], G["policies"], G["w3"], G["lim"]; pose = G["poses"][pid]; x, y, yaw = pose["x"], pose["y"], pose["yaw"]
    env.set_perturbation(); o = env.reset((x, y), yaw); hist = dict(S=[], B=[], C=[], A=[]); records = []; step_cpu = 0.0; n_steps = 0
    raw0 = policies[P.DEFAULT].chunk(o["state"], o["images"]); exe0, _, _ = P.slew_chain(lim, raw0, env.prev_ctrl); rows, o = exec_chunk(env, obs_, exe0, final_image=True)         # chunk 0 = default c0_s0 (all arms)
    hist["S"] += list(rows["S"]); hist["B"] += list(rows["B"]); hist["C"] += list(rows["C"]); hist["A"] += list(exe0.astype(np.float32))
    for t in range(20, max(states) + 1, 20):
        if t not in states:
            # replay-only: continue A0 with the default chunk (no branching at this state)
            raw = policies[P.DEFAULT].chunk(o["state"], o["images"]); ex, _, _ = P.slew_chain(lim, raw, np.asarray(env.prev_ctrl, float)); rows, o = exec_chunk(env, obs_, ex, final_image=True)
            hist["S"] += list(rows["S"][1:]); hist["B"] += list(rows["B"][1:]); hist["C"] += list(rows["C"][1:]); hist["A"] += list(ex.astype(np.float32)); continue
        guard(margin=15.0)
        assert len(hist["B"]) == t + 1 and len(hist["A"]) == t, (len(hist["B"]), t)
        prev = np.asarray(env.prev_ctrl, float).copy(); assert np.allclose(prev, np.asarray(hist["A"][-1], float), atol=1e-6)
        raw = np.stack([policies[k].chunk(o["state"], o["images"]) for k in CK]); exe = np.stack([P.slew_chain(lim, raw[i], prev)[0] for i in range(NB)]); legal = [bool(P.check_legal(lim, exe[i], prev)) for i in range(NB)]
        counts = [P.slew_chain(lim, raw[i], prev)[1:] for i in range(NB)]
        ph = phase_label(hist, t); pre = hist_fingerprint(hist)
        pred, W = score_state(w3, hist, t, exe, [x, y, yaw]); assert hist_fingerprint(hist) == pre            # predictions computed BEFORE any branch is executed; inputs untouched
        snap = snapshot(env); ends = []; brs = []; o_next = None; c0 = time.process_time()
        for k in range(NB):
            restore(env, snap); assert np.array_equal(env.prev_ctrl, prev)
            rw, o_end = exec_chunk(env, obs_, exe[k], final_image=(k == 0)); n_steps += 20
            if k == 0: o_next = o_end
            tm, Jr = realised(rw["box"], rw["C"]); ends.append(snapshot(env) if ((do_j2 and t in J2_STATES) or k == 0) else None)
            brs.append(dict(branch_id=f"p{pid:02d}_t{t:03d}_{CK[k]}", ckpt=CK[k], k=k, J20_real=Jr, terms_real=terms_dict(tm), box_xy_end=[float(v) for v in rw["box"][-1][:2]], box_z=[float(v) for v in rw["box"][:, 2]], hand_force_max=float(np.max(rw["C"][:, 4])), finger_contacts_max=float(np.max(rw["C"][:, 1])),
                            finger_force_max_N=float(np.max(rw["C"][:, 0]) * 50.0), legal=legal[k], clipped=int(counts[k][0]), slewed=int(counts[k][1]), applied_ctrl_maxdiff=rw["applied_ctrl_maxdiff"], finite=bool(np.isfinite(rw["B"]).all() and np.isfinite(rw["C"]).all()), c_rows=rw["C"].tolist() if out_arrays is None else None, _rows=rw))
        step_cpu += time.process_time() - c0
        Lin = [float(np.abs(exe[k][:, :7] - exe[0][:, :7]).max()) for k in range(NB)]
        rec = dict(state_id=f"p{pid:02d}_t{t:03d}", pose_idx=pid, t=t, phase=ph, x=x, y=y, yaw=yaw, pred=pred, branches=[{k_: v for k_, v in b.items() if k_ != "_rows"} for b in brs], exe_linf_vs_default_left_arm=Lin, input_fingerprint=pre, j2_states=bool(t in J2_STATES))
        if out_arrays is not None:
            Path(out_arrays).mkdir(parents=True, exist_ok=True); np.savez_compressed(Path(out_arrays) / f"{rec['state_id']}.npz", raw=raw.astype(np.float32), exe=exe.astype(np.float32), prev=prev, box=np.stack([b["_rows"]["box"] for b in brs]), tcp=np.stack([b["_rows"]["tcp"] for b in brs]), c=np.stack([b["_rows"]["C"] for b in brs]))
        if keep_clones is not None and do_j2 and t in J2_STATES: keep_clones[rec["state_id"]] = dict(ends=ends, exe=exe, rows_box=[b["_rows"]["box"] for b in brs], rows_c=[b["_rows"]["C"] for b in brs], rec=rec)
        records.append(rec)
        if on_record: on_record(rec)
        # continue A0 along the default branch (branch 0 is the A0 continuation): restore its end state, extend the histories with its rows
        rw0 = brs[0]["_rows"]; restore(env, ends[0]); o = o_next
        hist["S"] += list(rw0["S"][1:]); hist["B"] += list(rw0["B"][1:]); hist["C"] += list(rw0["C"][1:]); hist["A"] += list(exe[0].astype(np.float32))
        assert len(hist["B"]) == t + 21
    return records, dict(step_cpu_s=step_cpu, n_steps=n_steps)
def hist_fingerprint(hist):
    h = hashlib.sha256()
    for k in ("S", "B", "C", "A"): h.update(np.ascontiguousarray(np.array(hist[k], np.float32)).tobytes())
    return h.hexdigest()[:16]
def run_j2(keep, G, guard):
    """Phase 2 (optional): hold-last-executed-target continuation for 20 steps from each stored end-of-branch state; label = realised J over the 40 steps."""
    env, obs_ = G["env"], G["obs"]; n = 0
    for sid, K in keep.items():
        guard(margin=10.0)
        for k in range(NB):
            restore(env, K["ends"][k]); ctrl = K["exe"][k][-1]; env.prev_ctrl = np.asarray(ctrl, float).copy(); rw, _ = exec_chunk(env, obs_, [ctrl] * 20)
            box = np.concatenate([K["rows_box"][k], rw["box"][1:]]); c = np.concatenate([K["rows_c"][k], rw["C"][1:]]); tm, J2 = realised(box, c); b = K["rec"]["branches"][k]; b["J2_real"] = J2; b["terms2_real"] = terms_dict(tm); b["hold_applied_ctrl_maxdiff"] = rw["applied_ctrl_maxdiff"]; n += 1
    return n
# ------------------------------------------------------------------------------------------------------------------------------------------ worker / orchestrator
G = {}
def init_G(objs=None):
    import torch; torch.set_num_threads(1); os.environ["LP_NUM_THREADS"] = "1"
    from control.safety_limits import SafetyLimits
    if objs is None:
        policies = P.load_chunk_policies(torch); cams = policies[P.DEFAULT].cams; assert all(p.cams == cams for p in policies.values()); w3 = P.load_wm(torch); env = P.make_env(cams)
    else: policies, w3, env = objs
    G.update(policies=policies, w3=w3, env=env, obs=P.LiveObserver(env), lim=SafetyLimits.from_model(env.m), poses={p["idx"]: p for p in P.pilot_poses()})
def worker(job):
    wid, pose_ids, cap_s = job; c0 = P.cpu(); guard = P.make_guard(c0, cap_s, wid); resf = OUT / "results" / f"worker{wid}.json"; res = dict(worker=wid, cap_s=cap_s, phase1="RUNNING", phase2="NOT_STARTED", states=[], stop_reason=None); keep = {}; stats = dict(step_cpu_s=0.0, n_steps=0)
    try:
        init_G(); res["init_cpu_s"] = P.cpu() - c0
        for pid in pose_ids:
            guard(margin=40.0)
            def sink(r): res["states"].append(r); res["cpu_s"] = P.cpu() - c0; P.atomic_json(resf, res)
            recs, st = process_pose(pid, G, guard, do_j2=True, keep_clones=keep, out_arrays=OUT / "results/arrays", on_record=sink); stats["step_cpu_s"] += st["step_cpu_s"]; stats["n_steps"] += st["n_steps"]
        res["phase1"] = "COMPLETE"
    except P.CapReached as e: res["phase1"] = "PARTIAL"; res["stop_reason"] = f"cap in phase 1: {e}"
    except Exception as e:
        import traceback; res["phase1"] = "BLOCKED"; res["stop_reason"] = f"{type(e).__name__}: {e}"; res["traceback"] = traceback.format_exc()[-1500:]
    res["phase1_cpu_s"] = P.cpu() - c0; P.atomic_json(resf, res)
    if res["phase1"] == "COMPLETE" and keep:
        per_step = stats["step_cpu_s"] / max(stats["n_steps"], 1); need = per_step * 20 * NB * len(keep) * 1.35 + 10.0; left = cap_s - 20.0 - (P.cpu() - c0); res["phase2_need_s"] = need; res["phase2_left_s"] = left
        if left < need: res["phase2"] = "SKIPPED_BUDGET"
        else:
            try: res["phase2_branches"] = run_j2(keep, G, guard); res["phase2"] = "COMPLETE"
            except P.CapReached as e: res["phase2"] = "PARTIAL"; res["stop_reason"] = f"cap in phase 2: {e}"
            except Exception as e: res["phase2"] = "BLOCKED"; res["stop_reason"] = f"{type(e).__name__}: {e}"
    res["cpu_s"] = P.cpu() - c0; P.atomic_json(resf, res); return wid, res["phase1"], res["phase2"], res["cpu_s"]
def verify_freeze():
    F = json.loads(FREEZE.read_text()); bad = [f for f, h in F["files_sha256"].items() if P.sha256(ROOT / f) != h]
    if bad: raise SystemExit(f"frozen files changed: {bad}")
    return F
def main():
    if len(sys.argv) < 2 or sys.argv[1] != "run": raise SystemExit("usage: branching_replay.py run")
    F = verify_freeze(); rep = json.loads((OUT / "integration_report.json").read_text())
    if rep["verdict"] != "PASS": raise SystemExit("integration gates not PASS")
    if rep["code_sha256"]["eval/branching_replay.py"] != P.sha256(__file__): raise SystemExit("runner changed after the integration gates")
    if (OUT / "results").exists() and any((OUT / "results").glob("worker*.json")): raise SystemExit("results exist; refusing to overwrite")
    used = ledger_used(); nw = 4; cap_w = (CAP_TOTAL_S - used - 30.0) / nw; assert cap_w > 150, cap_w; ids = [p["idx"] for p in P.pilot_poses()]; chunks = [ids[i::nw] for i in range(nw)]; t0 = time.time(); c_par0 = P.cpu()
    (OUT / "results").mkdir(parents=True, exist_ok=True); import multiprocessing as mp
    with mp.get_context("spawn").Pool(nw) as pool: outs = pool.map(worker, [(w, c, cap_w) for w, c in enumerate(chunks)]); pool.close(); pool.join()
    cpu_s = P.cpu() - c_par0; ledger_add("branching_run", cpu_s); print(json.dumps(dict(outs=outs, cpu_s=cpu_s, wall_s=time.time() - t0, ledger_total_s=ledger_used(), cap_total_s=CAP_TOTAL_S)))
if __name__ == "__main__": main()
