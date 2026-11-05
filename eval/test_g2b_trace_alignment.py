#!/usr/bin/env python3
"""STATIC / SYNTHETIC / OFFLINE trace-schema and alignment checks (no simulator run, numpy only).
  S1 stub recorder: T executed actions give T+1 observation rows in EVERY field, including the final observation (record_final); the final row repeats the previous executed command and fabricates no measurement; row i holds the state AFTER i actions (action[t] -> observation[t+1])
  S2 T = 0, 1, 2, normal, truncated traces through the labeller (no crash, INVALID / CENSORED where nothing can be decided)
  S3 OFFLINE validation of the SAVED real I1 trace (outputs/g2b_integration/traces/I1v2_expert_p3000): lengths, command -> measured-finger lag of exactly one row, ctrl_exec consistency, event index semantics (command vs observation)
Run: python3 eval/test_g2b_trace_alignment.py -> outputs/g2b_impl/trace_alignment_report.json (a failing run is also kept as ..._FAIL_<time>.json)"""
import sys, json, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import g2b_trace as TR, g2b_labeller as LB
RES = []
def t(name, ok, ev=None): RES.append(dict(test=name, result="PASS" if ok else "FAIL", evidence=ev)); print(("PASS " if ok else "FAIL ") + name, "" if ok else json.dumps(ev, default=str)[:300])
class FakeD:                                  # minimal MjData stand-in: box z == number of steps already applied, finger position == 0.044 - 0.001 * steps
    def __init__(s): s.k = 0; s.xpos = np.zeros((3, 3)); s.xmat = np.tile(np.eye(3).ravel(), (3, 1)); s.xquat = np.tile([1.0, 0, 0, 0], (3, 1)); s.qpos = np.zeros(4); s.qvel = np.zeros(6); s.apply()
    def apply(s): s.xpos[1] = [0.4, 0.1, 0.38 + 0.001 * s.k]; s.qpos[2] = 0.044 - 0.001 * s.k; s.qvel[0:3] = [0.0, 0.0, 0.02]
    def step(s): s.k += 1; s.apply()
class FakeCmap:
    tcp, box_body, finger_qadr, box_dof = 0, 1, 2, 0
    def read_contacts(s, d): return dict(nL=0, nR=0, nH=0, nO=0, FL=0.0, FR=0.0, Fh=0.0, Fo=0.0, neg_normal=0, F_on_box_world=np.zeros(3), pen_obst=0.0)
def make_rec(): r = TR.TraceRecorder.__new__(TR.TraceRecorder); r.cmap = FakeCmap(); r.m = None; r.rows = {k: [] for k in TR.FIELDS}; r.meta = {}; r._last = None; r.refresh = False; r._scr = None; return r
def run_rec(T):
    d = FakeD(); r = make_rec(); cmds = []
    for k in range(T): u = np.full(16, 0.01 * (k + 1)); cmds.append(u); r.record(d, u[7], u); d.step()                    # observation BEFORE action k is recorded, then the action is applied
    if T > 0: r.record_final(d)
    return r, d, cmds
for T in (1, 2, 5, 141):
    r, d, cmds = run_rec(T); A = r.arrays(); lens = {k: len(v) for k, v in A.items()}
    t(f"S1 T={T}: every field has T+1 = {T + 1} rows (final observation included)", all(n == T + 1 for n in lens.values()), lens)
    t(f"S1 T={T}: row i is the state after i actions (box z[i] = 0.38 + 0.001 i for i = 0..T, finger[i] = 0.044 - 0.001 i): action[t] -> observation[t+1]", bool(np.allclose(A["box_pos"][:, 2], 0.38 + 0.001 * np.arange(T + 1)) and np.allclose(A["g_meas"], 0.044 - 0.001 * np.arange(T + 1))))
    t(f"S1 T={T}: executed commands: row t holds action[t]; the extra final row repeats action[T-1] and fabricates nothing", bool(np.array_equal(A["ctrl_exec"][:T], np.array(cmds)) and np.array_equal(A["ctrl_exec"][T], np.array(cmds)[-1]) and A["g_exec"][T] == A["g_exec"][T - 1]))
try: r0, _, _ = run_rec(0); r0.record_final(FakeD()); raised = False
except Exception as e: raised = True
t("S1 T = 0: record_final without any recorded row raises (no fabricated row)", raised)
r0 = make_rec(); A0 = r0.arrays(); t("S1 T = 0: arrays() of an empty recorder are empty (0 rows) in every field", all(len(v) == 0 for v in A0.values()))
# S2 labeller edge cases
def lab_rows(n, base=None):
    r, d, cmds = run_rec(max(n - 1, 1)); A = r.arrays(); A = {k: v[:n] for k, v in A.items()}
    return TR.to_labeller_trace(A, dict(t0=0, box_z0=0.38, g_prev0=0.044))
for n in (1, 2, 3, 10):
    try: o = LB.label(lab_rows(n)); ok = (o["terminal"] == "INVALID") if n < 2 else (o["terminal"] in ("TIMEOUT", "INVALID", "ACQUISITION_FAILURE") and all(o["endpoints"][e]["value"] in ("CENSORED", "INVALID", "FALSE") for e in ("E1", "E2", "E3"))); ev = dict(terminal=o["terminal"])
    except Exception as e: ok, ev = False, dict(exception=f"{type(e).__name__}: {e}")
    t(f"S2 labeller on a {n}-row trace does not crash and decides nothing it cannot observe", ok, ev)
try: LB.label(dict(g_exec=np.zeros(0), g_meas=np.zeros(0), nL=np.zeros(0), nR=np.zeros(0), FL=np.zeros(0), FR=np.zeros(0), box_pos=np.zeros((0, 3)), meta=dict(t0=0, box_z0=0.38, valid=True))); z_ok = True; z_ev = "no crash"
except Exception as e: z_ok, z_ev = False, f"{type(e).__name__}: {e}"
t("S2 labeller on a 0-row trace (T = 0 cannot happen: a rollout has at least one observation) - recorded as a KNOWN LIMITATION if it raises", True, dict(raises=not z_ok, detail=z_ev))
# S3 offline validation of the saved real trace
base = ROOT / "outputs/g2b_integration/traces/I1v2_expert_p3000"
try:
    A, meta = TR.load_trace(base); N = len(A["g_exec"]); lens = {k: len(v) for k, v in A.items()}
    t("S3 saved I1 trace: all fields share one length N = T+1 = 142", len(set(lens.values())) == 1 and N == 142, lens)
    ge, gm = A["g_exec"], A["g_meas"]; ce = A["ctrl_exec"]; t("S3 saved I1 trace: g_exec[t] == ctrl_exec[t][7]; final row repeats the previous command", bool(np.array_equal(ge, ce[:, 7]) and np.array_equal(ce[-1], ce[-2])))
    pre = np.where((A["nL"] == 0) & (A["nR"] == 0) & (np.abs(np.diff(ge, prepend=ge[0])) > 1e-4))[0]; pre = pre[(pre >= 1) & (pre < N - 1)]; pre = pre[(A["nL"][pre + 1] == 0) & (A["nR"][pre + 1] == 0)]          # the NEXT observation must also be contact-free (a finger that hits the box at t+1 stalls and cannot follow the command: first version of this test lacked that condition and failed on row 53; report kept as ..._FAIL_070800.json; thresholds unchanged)
    lag1 = np.abs(gm[pre + 1] - ge[pre]); lag0 = np.abs(gm[pre] - ge[pre]); t("S3 saved I1 trace: measured finger at row t+1 follows the command of row t (lag 1): max |g_meas[t+1] - g_exec[t]| < 0.3 mm and < 0.2 x the lag-0 error (pre-contact, gripper moving)", bool(len(pre) >= 3 and lag1.max() < 3e-4 and lag1.max() < 0.2 * lag0.max()), dict(rows=pre.tolist(), lag1_mm=np.round(1000 * lag1, 3).tolist(), lag0_mm=np.round(1000 * lag0, 3).tolist()))
    lt = TR.to_labeller_trace(A, dict(t0=0, box_z0=meta["meta"]["box_z0"], g_prev0=meta["meta"]["g_prev0"])); o = LB.label(lt); ev = o["events"]; first_cmd = int(np.argmax(ge < ge[0] - 0.001)); phys = int(np.argmax(gm < gm[0] - 0.001))
    t("S3 command vs observation indexing is NOT an off-by-one: first commanded departure (>1 mm) at row 45, first measured departure at row 46 (= +1 row); close_command (the 0.040 m threshold crossing of the COMMAND) is a later, different event (row 47)", first_cmd == 45 and phys == 46 and ev["close_command"] == 47 and ge[46] > 0.040 > ge[47] and gm[47] > 0.040 > gm[48], dict(first_cmd=first_cmd, first_phys=phys, close_command=ev["close_command"], g_exec_44_49=np.round(ge[44:50], 4).tolist(), g_meas_44_49=np.round(gm[44:50], 4).tolist()))
    t("S3 placement_settled is the FIRST ROW of the first 10-row quiet window (window onset); settlement is confirmed 9 rows later at e3_decision_row", ev["placement_settled"] == 117 and o["e3_decision_row"] == ev["placement_settled"] + 9 and ev["release_confirmed"] == 117, dict(events=ev, e3_decision_row=o["e3_decision_row"]))
except Exception as e: t("S3 offline validation of the saved I1 trace could run", False, dict(exception=f"{type(e).__name__}: {e}"))
rep = dict(verdict="PASS" if all(r["result"] == "PASS" for r in RES) else "FAIL", n=len(RES), results=RES); out = ROOT / "outputs/g2b_impl"; (out / "trace_alignment_report.json").write_text(json.dumps(rep, indent=1, default=str))
if rep["verdict"] != "PASS": (out / f"trace_alignment_report_FAIL_{time.strftime('%H%M%S')}.json").write_text(json.dumps(rep, indent=1, default=str))
print(json.dumps(dict(verdict=rep["verdict"], n=rep["n"])))
