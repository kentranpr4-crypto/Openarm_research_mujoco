#!/usr/bin/env python3
"""B1 STATIC/PLUMBING tests of the per-finger instrumentation (needs MuJoCo; ACT venv).  NOT an evaluation: one scripted-expert plumbing run (~95 control steps, no policy, no candidate) at pose (0.40, 0.10, 0.0) plus constructed static contact states.
Verifies body/geom mapping, contact-frame conventions, force sign and physical units against first principles (box weight balance) and against the historical observer; nothing is inferred from the actuator servo force.
Run: /home/user/.venvs/act/bin/python eval/test_g2b_trace_contacts.py  -> outputs/g2b_impl/trace_contacts_report.json"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, time, tempfile, traceback, resource, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "eval"), str(ROOT / "data")): sys.path.insert(0, p)
import mujoco, closed_loop_pilot as P, g2b_trace as TR, g2b_labeller as LB
from policies.baseline import ScriptedExpert
from control.safety_limits import SafetyLimits
def cpu(): a = resource.getrusage(resource.RUSAGE_SELF); return a.ru_utime + a.ru_stime
C0 = cpu(); GATES = []
def gate(n, name):
    def deco(fn):
        c0 = cpu()
        try: ok, ev = fn()
        except Exception as e: ok, ev = False, f"EXCEPTION {type(e).__name__}: {e}\n{traceback.format_exc()[-700:]}"
        GATES.append(dict(gate=n, name=name, result="PASS" if ok else "FAIL", evidence=ev, cpu_s=round(cpu() - c0, 2))); print(f"T{n:02d} {'PASS' if ok else 'FAIL'}  {name}  {json.dumps(ev, default=float)[:400] if not ok else ''}", flush=True); return fn
    return deco
env = P.make_env(()); m = env.m; ex = ScriptedExpert(m, control_hz=20); lim = SafetyLimits.from_model(m); obs = P.LiveObserver(env); POSE = (0.40, 0.10, 0.0)
pl = ex.plan(POSE[:2], POSE[2]); names = np.array(pl["phase_names"])[np.array(pl["phase"])]; T = int(np.where(names == "transport")[0][0]) + 8
env.set_perturbation(); env.reset(POSE[:2], POSE[2]); prev = env.prev_ctrl.copy(); rec = TR.TraceRecorder(m, dict(pose=POSE, note="plumbing run, not a result", t0=0)); hist = []; box_z0 = float(env.d.xpos[rec.cmap.box_body][2])
for t in range(T):
    c, _, _ = lim.clip_and_slew(pl["ctrl"][t], prev, 0.05); rec.record(env.d, c[7], c); b_, c_ = obs.row(); hist.append((b_.copy(), c_.copy())); P.lean_step(env, c, False); prev = c
rec.record_final(env.d); A = rec.arrays(); cm = rec.cmap; mass = float(m.body_mass[cm.box_body]); gz = float(m.opt.gravity[2]); i_hold = np.where(names[:T] == "grip_hold")[0]
speed = np.linalg.norm(A["box_vel"], axis=1); hold_rows = [int(i) for i in i_hold if speed[i] < 2e-3]
@gate(1, "name-based geom/body mapping (left arm only), distinct ids, finger joint and free-joint indices")
def _(): return True, dict(bodies=cm.body, finger_qadr=cm.finger_qadr, box_dof=cm.box_dof)
@gate(2, "contact-frame convention and force sign: normal components >= 0 on every row; net world force on the resting box == +m*|g| (box weight)")
def _():
    fz = [float(A["F_on_box_world"][i, 2]) for i in hold_rows]; w = mass * abs(gz); err = [abs(f - w) for f in fz]; neg = int(A["neg_normal"].sum())
    return neg == 0 and len(hold_rows) >= 3 and max(err) <= 0.05 * w + 0.02, dict(n_hold_rows=len(hold_rows), box_mass_kg=mass, weight_N=w, Fz_on_box_N=[round(f, 4) for f in fz[:4]], max_abs_err_N=max(err), neg_normal_points=neg)
@gate(3, "physical units: newtons (weight balance), metres, seconds; per-finger forces balanced (action-reaction) and in the expected range")
def _():
    FL, FR = A["FL"][hold_rows], A["FR"][hold_rows]; mean = (FL + FR) / 2; rel = np.abs(FL - FR) / mean; tot = FL + FR
    return bool(rel.max() < 0.10 and 43.0 <= tot.min() and tot.max() <= 53.0), dict(FL_N=float(FL.mean()), FR_N=float(FR.mean()), rel_imbalance_max=float(rel.max()), total_N=[float(tot.min()), float(tot.max())], note="range 43-53 N is an observational consistency check against the G1 median 47.9 N, not the source of the logged values")
@gate(4, "per-finger counts/forces sum to the historical observer values (finger_force, contacts, palm) on every row; box-in-TCP matches the observer a_offset")
def _():
    n = len(hist); ff = np.array([h[1][0] * 50.0 for h in hist]); nf = np.array([h[1][1] for h in hist]); hf = np.array([h[1][4] for h in hist]); a_off = np.array([h[0][6] for h in hist])
    d1 = float(np.abs((A["FL"][:n] + A["FR"][:n]) - ff).max()); d2 = float(np.abs((A["nL"][:n] + A["nR"][:n]) - nf).max()); d3 = float(np.abs(A["Fh"][:n] - hf).max()); d4 = float(np.abs(A["box_in_tcp"][:n, 2] - a_off).max())
    return max(d1, d2, d3, d4) < 1e-5, dict(finger_force_diff=d1, contact_count_diff=d2, palm_diff=d3, a_offset_diff=d4)
@gate(5, "left/right assignment: during the hold both fingers carry load and each finger has several contact points; palm silent in the nominal hold")
def _():
    nl, nr = A["nL"][hold_rows], A["nR"][hold_rows]
    return bool(nl.min() >= 1 and nr.min() >= 1 and (A["Fh"][hold_rows] == 0).all()), dict(nL=[int(nl.min()), int(nl.max())], nR=[int(nr.min()), int(nr.max())], contact_points_total=[int((nl + nr).min()), int((nl + nr).max())])
@gate(7, "box velocity (free-joint qvel linear part, world frame, m/s) matches the position change over ONE physics substep (definition/frame/units check on the final moving state; positions from qpos because xpos is not refreshed until the next forward pass)")
def _():
    d = env.d; adr = int(m.jnt_qposadr[m.joint('blue_box_free').id]); x0 = d.qpos[adr:adr + 3].copy(); v0 = d.qvel[cm.box_dof:cm.box_dof + 3].copy(); mujoco.mj_step(m, d); x1 = d.qpos[adr:adr + 3].copy(); v1 = d.qvel[cm.box_dof:cm.box_dof + 3].copy(); h = float(m.opt.timestep); fd = (x1 - x0) / h; mean = 0.5 * (v0 + v1); rel = float(np.linalg.norm(fd - mean) / max(np.linalg.norm(mean), 1e-9))
    return bool(np.linalg.norm(mean) > 0.02 and rel < 0.02), dict(substep_s=h, speed_ms=float(np.linalg.norm(mean)), rel_diff=rel, v_qvel=[float(x) for x in mean], v_fd=[float(x) for x in fd], note="runs 1-2 compared control-step differences (central 15 % tolerance: median 20 %; trapezoid 5 %: median 23.5 %, max 88 %) and failed: over 0.05 s the scripted lift accelerates strongly and changes contacts, so control-step differences cannot validate an instantaneous velocity; the per-substep check isolates the definition")
@gate(6, "palm force is logged separately: constructed overlap of the box with the hand mesh (static, no stepping)")
def _():
    env.set_perturbation(); env.reset(POSE[:2], POSE[2]); d = env.d; hc = d.geom_xpos[cm.hand].copy(); adr = int(m.jnt_qposadr[m.joint("blue_box_free").id]); d.qpos[adr:adr + 3] = hc; mujoco.mj_forward(m, d); r = cm.read_contacts(d); f6 = np.zeros(6); raw = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if cm.box in (c.geom1, c.geom2) and cm.hand in (c.geom1, c.geom2): mujoco.mj_contactForce(m, d, i, f6); raw += abs(f6[0])
    return bool(r["nH"] > 0 and r["Fh"] > 0 and r["FL"] == 0 and r["FR"] == 0 and abs(r["Fh"] - raw) < 1e-9 and r["neg_normal"] == 0), dict(nH=r["nH"], Fh_N=r["Fh"], independent_recompute_N=raw, FL=r["FL"], FR=r["FR"])
@gate(8, "end-to-end: recorder arrays -> labeller trace -> labels (grasp acquired and lift confirmed on the scripted hold; trace valid)")
def _():
    lt = TR.to_labeller_trace(A, dict(t0=0, box_z0=box_z0)); o = LB.label(lt); ev = o["events"]
    return bool(lt["meta"]["valid"] and o["endpoints"]["E1"]["value"] == "TRUE" and "grasp_acquired" in ev and "lift_confirmed" in ev), dict(terminal=o["terminal"], events=ev, E=[o["endpoints"][e]["value"] for e in ("E1", "E2", "E3")], margins=o["margins"])
@gate(9, "schema, units metadata, atomic save/load round trip, invalid-data handling")
def _():
    with tempfile.TemporaryDirectory() as td:
        TR.save_trace(Path(td) / "b0", A, dict(id="plumbing", t0=0), label=LB.label(TR.to_labeller_trace(A, dict(t0=0, box_z0=box_z0))), code_files=[ROOT / "eval/g2b_trace.py", ROOT / "eval/g2b_labeller.py"]); B, side = TR.load_trace(Path(td) / "b0"); left = sorted(q.name for q in Path(td).iterdir())
    same = all(np.array_equal(A[k], B[k]) for k in A); bad = {k: v.copy() for k, v in A.items()}; bad["FL"][3] = np.nan; v = TR.to_labeller_trace(bad, dict(t0=0, box_z0=box_z0))["meta"]["valid"]
    return bool(same and left == ["b0.json", "b0.npz"] and all("unit" in f for f in side["fields"].values()) and set(side["fields"]) == set(TR.FIELDS) and v is False and len(side["code_sha256"]) == 2 and side["label"]["events"]), dict(files=left, n_fields=len(side["fields"]), nan_marks_trace_invalid=(v is False))
rep = dict(verdict="PASS" if all(g["result"] == "PASS" for g in GATES) and len(GATES) == 9 else "FAIL", gates=GATES, rows=int(len(A["g_exec"])), cpu_s=cpu() - C0, note="plumbing/static checks; not an evaluation; no candidate, no policy inference")
(ROOT / "outputs/g2b_impl/trace_contacts_report.json").write_text(json.dumps(rep, indent=1, default=float)); print(rep["verdict"], f"cpu={rep['cpu_s']:.1f}s")
for g in GATES: print(g["gate"], g["result"], json.dumps(g["evidence"], default=float)[:420])
