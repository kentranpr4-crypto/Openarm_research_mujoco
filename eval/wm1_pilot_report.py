#!/usr/bin/env python3
"""WM-1 intervention pilot report: quality, intervention consistency, safety, contact/drop prevalence (checks defined BEFORE the pilot was analysed; see PILOT_CRITERIA).
Run (ACT venv): python eval/wm1_pilot_report.py   ->  outputs/wm1/pilot/pilot_report.json, pilot_tables.md, pilot_plots.png"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, tempfile, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "data")):
    if p not in sys.path: sys.path.insert(0, p)
import wm1_intervene as WI, wm1_sim as S
OUT = ROOT / "outputs/wm1/pilot"; idx = json.loads((OUT / "pilot_index.json").read_text()); EP = idx["episodes"]; cfg = idx["config"]
CRIT = dict(P1="250/250 finite episodes, no forbidden seed, one split per pose", P2=">= 9/10 control episodes succeed (expert parity)", P3="8 re-runs bitwise identical", P4="25 distinct executed-action hashes per pose; every non-control variant differs from control by RMS > 1e-3 rad (left arm)",
            P5="executed actions inside ctrlrange and slew limits (arm 2.0 rad/s, gripper 0.2 m/s), no NaN", P6="dose-response: depth a_close monotone (Spearman <= -0.8) in >= 8/10 poses; timing close-step shift = level +-1 for >= 90 % ; noise RMS in [0.5,1.5] x sigma for >= 80 %; speed duration change within 15 % of expected; release TCP z shift = level +-3 mm for >= 80 %",
            P7="report palm/drop prevalence; goal: palm-contact positives inside the TRAIN region from deep-grasp variants (>= 10 % of depth>=+0.03 variants in TRAIN poses)", P8="IK/plan failures and range/slew clip counts reported")
D = {e["name"]: dict(np.load(OUT / "episodes" / f"{e['name']}.npz")) | e for e in EP}
def ev(R):
    g = R["gripper_cmd"]; cl = np.nonzero(g < 0.028)[0]; tc = int(cl[0]) if len(cl) else None; tr = None
    if tc is not None:
        r = np.nonzero(g[tc:] > 0.035)[0]; tr = tc + int(r[0]) if len(r) else None
    return tc, tr
rows = []
for n, R in D.items():
    tc, tr = ev(R); fin = R["box_pos"][-1, :2] - np.array([0.44, 0.30]); post = bool(R["pick_success"]) and not R["task_success"]
    rows.append(dict(name=n, pid=R["pid"], vid=R["vid"], family=R["family"], level=R["level"], seed=R["seed"], split=R["split"], region=R["region"], T=R["T"], tc=tc, tr=tr, a_close=float(R["box_in_tcp"][tc, 2] * 100) if tc is not None else np.nan,
                     palm=bool((R["hand_force"][tc:(tr or len(R["hand_force"]))] > 0.01).any()) if tc is not None else False, finger=bool((R["finger_contacts"] > 0).any()), success=bool(R["task_success"]), pick=bool(R["pick_success"]), place=bool(R["place_success"]), drop=post and fin[0] > -1,
                     cluster=bool(post and fin[0] < -0.025 and abs(fin[1]) < 0.01), post_pick_fail=post, final_err_cm=100 * float(np.linalg.norm(fin)), tcp_z_release=float(R["tcp_pos"][tr, 2]) if tr is not None else np.nan, plan_ok=R["plan_ok"], ik_fail=R["ik_fail_steps"], clipped=R["clipped_range"], slewed=R["slewed"], finite=R["finite"]))
by = lambda **kw: [r for r in rows if all(r[k] == v for k, v in kw.items())]
res = dict(criteria=CRIT, n_episodes=len(rows), episodes_by_family={f: len([r for r in rows if r["family"] == f]) for f in ("control", "depth", "timing", "noise", "speed", "release_height")})
# P1
splits = defaultdict(set); [splits[r["pid"]].add(r["split"]) for r in rows]; res["P1"] = dict(passed=bool(len(rows) == 250 and all(r["finite"] for r in rows) and all(len(v) == 1 for v in splits.values()) and 60 not in (31, 32, 33)), finite=sum(r["finite"] for r in rows), poses=len(splits), splits_by_pose={str(k): list(v)[0] for k, v in splits.items()})
# P2
ctl = by(family="control"); res["P2"] = dict(passed=sum(r["success"] for r in ctl) >= 9, control_success=sum(r["success"] for r in ctl), by_region={g: [sum(r["success"] for r in ctl if r["region"] == g), sum(1 for r in ctl if r["region"] == g)] for g in sorted({r["region"] for r in ctl})}, failed=[(r["pid"], r["region"], round(r["final_err_cm"], 1)) for r in ctl if not r["success"]])
# P3 determinism (re-run 8 jobs into a temp dir)
tmp = Path(tempfile.mkdtemp()); WI.OUT = tmp; ident = []
for pid in (3, 8):
    for vid in (0, 3, 14, 20):
        job = dict(pose=WI.POSES[pid], variant=WI.VARIANTS[vid]); r = WI.run_one(job); a = np.load(tmp / "episodes" / f"p{pid:02d}_v{vid:02d}.npz"); b = D[f"p{pid:02d}_v{vid:02d}"]
        ident.append(all(np.array_equal(a[k], b[k], equal_nan=True) for k in ("action", "state", "box_pos", "hand_force", "finger_force")))
res["P3"] = dict(passed=all(ident), identical=sum(ident), of=len(ident))
# P4 distinct actions
bad_h = []; rms_ctl = []
for pid in range(10):
    hs = [D[f"p{pid:02d}_v{v:02d}"]["h_action"] for v in range(25)]
    if len(set(hs)) != 25: bad_h.append(pid)
    c = D[f"p{pid:02d}_v00"]["action"][:, :7]
    for v in range(1, 25):
        x = D[f"p{pid:02d}_v{v:02d}"]["action"][:, :8]; n_ = min(len(c), len(x)); rms_ctl.append((v, float(np.sqrt(np.mean((x[:n_, :7] - c[:n_]) ** 2))), float(np.abs(x[:n_, 7] - D[f"p{pid:02d}_v00"]["action"][:n_, 7]).max())))
small = [(v, r, g) for v, r, g in rms_ctl if r <= 1e-3 and g <= 1e-4]
res["P4"] = dict(passed=not bad_h and not small, poses_with_duplicate_actions=bad_h, variants_not_distinguishable_from_control=len(small), min_rms_vs_control_by_family={f: float(min(r for v, r, g in rms_ctl if WI.VARIANTS[v]["family"] == f)) for f in ("depth", "timing", "noise", "speed", "release_height")})
# P5 safety
lim_hi = None; viol = 0; maxv = {"arm": 0.0, "grip": 0.0}; nan = 0
from lab_env import LabEnv
env = S.get_env()["env"]; lo, hi = env.m.actuator_ctrlrange[:, 0], env.m.actuator_ctrlrange[:, 1]
for n, R in D.items():
    A = R["action"].astype(float); viol += int(((A < lo - 1e-6) | (A > hi + 1e-6)).any()); dA = np.abs(np.diff(A, axis=0)) * 20; arm = np.delete(np.arange(16), [7, 15]); maxv["arm"] = max(maxv["arm"], float(dA[:, arm].max())); maxv["grip"] = max(maxv["grip"], float(dA[:, [7, 15]].max())); nan += int(not np.isfinite(A).all())
res["P5"] = dict(passed=bool(viol == 0 and maxv["arm"] <= 2.0 + 1e-3 and maxv["grip"] <= 0.2 + 1e-3 and nan == 0), episodes_outside_ctrlrange=viol, max_arm_speed_rad_s=maxv["arm"], max_gripper_speed_m_s=maxv["grip"], nan=nan, clipped_range_total=int(sum(r["clipped"] for r in rows)), slewed_total=int(sum(r["slewed"] for r in rows)),
                 note="expert plans already respect the placeholder limits; slew/clip counts show how often the safety layer altered a requested target")
# P6 intervention consistency
dep = []; 
for pid in range(10):
    rr = sorted([r for r in rows if r["pid"] == pid and r["family"] in ("depth", "control")], key=lambda r: r["level"]); lv = [r["level"] for r in rr]; ac = [r["a_close"] for r in rr]
    if len(set(ac)) > 1: dep.append(st.spearmanr(lv, ac)[0])
slope = [np.polyfit([r["level"] * 100 for r in [x for x in rows if x["pid"] == pid and x["family"] in ("depth", "control")]], [r["a_close"] for r in [x for x in rows if x["pid"] == pid and x["family"] in ("depth", "control")]], 1)[0] for pid in range(10)]
tim = [(r["tc"] - D[f"p{r['pid']:02d}_v00"]["tc"] if "tc" in D[f"p{r['pid']:02d}_v00"] else None) for r in []]
ctc = {r["pid"]: r["tc"] for r in ctl}; tim_ok = [abs((r["tc"] - ctc[r["pid"]]) - r["level"]) <= 1 for r in by(family="timing") if r["tc"] is not None]
noise = []
for r in by(family="noise"):
    c = D[f"p{r['pid']:02d}_v00"]["action"][:, :7]; x = D[r["name"]]["action"][:, :7]; n_ = min(len(c), len(x)); noise.append((r["level"], float(np.sqrt(np.mean((x[:n_] - c[:n_]) ** 2))) / r["level"]))
noise_ok = [0.5 <= q <= 1.5 for _, q in noise]
ie = WI.InterventionExpert(env.m); spd = []
for r in by(family="speed"):
    pl = ie.e; pl.depth = 0.0; pl.dz = 0.0; p0 = pl.plan((WI.POSES[r["pid"]]["x"], WI.POSES[r["pid"]]["y"]), WI.POSES[r["pid"]]["yaw"]); ph = p0["phase"]; nm = p0["phase_names"]
    L = int(np.argmax(ph == nm.index("release"))) - int(np.argmax(ph == nm.index("lift"))); exp_dT = round(L / r["level"]) - L; Tc = D[f"p{r['pid']:02d}_v00"]["T"]; spd.append(abs((r["T"] - Tc) - exp_dT) <= max(2, 0.15 * abs(exp_dT)))
rel = []
for r in by(family="release_height"):
    c = [x for x in ctl if x["pid"] == r["pid"]][0]; rel.append(abs((r["tcp_z_release"] - c["tcp_z_release"]) - r["level"]) <= 0.003)
res["P6"] = dict(passed=bool(np.sum(np.array(dep) <= -0.8) >= 8 and np.mean(tim_ok) >= 0.9 and np.mean(noise_ok) >= 0.8 and np.mean(spd) >= 0.8 and np.mean(rel) >= 0.8), depth_spearman_per_pose=[round(float(x), 2) for x in dep], depth_slope_cm_per_cm_per_pose=[round(float(s), 2) for s in slope],
                 timing_close_shift_ok=[int(sum(tim_ok)), len(tim_ok)], noise_rms_over_sigma=[(l, round(q, 2)) for l, q in noise], noise_ok=[int(sum(noise_ok)), len(noise_ok)], speed_duration_ok=[int(sum(spd)), len(spd)], release_height_tcp_shift_ok=[int(sum(rel)), len(rel)])
# P7/P8 prevalence + data quality
fam = {}
for f in res["episodes_by_family"]:
    rr = [r for r in rows if r["family"] == f]; fam[f] = dict(n=len(rr), success=sum(r["success"] for r in rr), pick=sum(r["pick"] for r in rr), drop_or_misplaced=sum(r["post_pick_fail"] for r in rr), no_pick=sum(not r["pick"] for r in rr), palm=sum(r["palm"] for r in rr), cluster=sum(r["cluster"] for r in rr), plan_fail=sum(not r["plan_ok"] for r in rr), ik_fail_steps=sum(r["ik_fail"] for r in rr))
lev = {}
for l in (-0.01, 0.01, 0.02, 0.03, 0.04):
    rr = [r for r in rows if r["family"] == "depth" and abs(r["level"] - l) < 1e-9]; lev[f"{l:+.2f}"] = dict(n=len(rr), success=sum(r["success"] for r in rr), palm=sum(r["palm"] for r in rr), cluster=sum(r["cluster"] for r in rr), no_pick=sum(not r["pick"] for r in rr), a_close_mean_cm=float(np.nanmean([r["a_close"] for r in rr])))
deep_tr = [r for r in rows if r["family"] == "depth" and r["level"] >= 0.03 and r["region"] == "TRAIN"]
res["P7"] = dict(passed=bool(len(deep_tr) and np.mean([r["palm"] for r in deep_tr]) >= 0.10), by_family=fam, depth_levels=lev, palm_positive_episodes=sum(r["palm"] for r in rows), palm_in_TRAIN_region=sum(r["palm"] for r in rows if r["region"] == "TRAIN"), palm_prevalence_deep_TRAIN=float(np.mean([r["palm"] for r in deep_tr])) if deep_tr else None,
                 palm_by_split={s: [sum(r["palm"] for r in rows if r["split"] == s), sum(1 for r in rows if r["split"] == s)] for s in sorted({r["split"] for r in rows})}, overall_success=sum(r["success"] for r in rows), overall_post_pick_fail=sum(r["post_pick_fail"] for r in rows), overall_no_pick=sum(not r["pick"] for r in rows))
res["P8"] = dict(plan_failures=sum(not r["plan_ok"] for r in rows), ik_fail_steps_total=sum(r["ik_fail"] for r in rows), lengths=dict(min=min(r["T"] for r in rows), max=max(r["T"] for r in rows)))
res["all_pass"] = all(res[k]["passed"] for k in ("P1", "P2", "P3", "P4", "P5", "P6", "P7")); (OUT / "pilot_report.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8)); L_ = sorted({r["level"] for r in rows if r["family"] == "depth"} | {0.0})
for reg, col in (("TRAIN", "C0"), ("LOW_A", "C1"), ("LOW_B_strict", "C3")):
    ax[0].plot([l * 100 for l in L_], [np.nanmean([r["a_close"] for r in rows if r["region"] == reg and (r["family"] == "depth" and abs(r["level"] - l) < 1e-9 or r["family"] == "control" and l == 0)]) for l in L_], "-o", color=col, label=reg)
    ax[1].plot([l * 100 for l in L_], [np.mean([r["palm"] for r in rows if r["region"] == reg and (r["family"] == "depth" and abs(r["level"] - l) < 1e-9 or r["family"] == "control" and l == 0)]) * 100 for l in L_], "-o", color=col)
    ax[2].plot([l * 100 for l in L_], [np.mean([r["success"] for r in rows if r["region"] == reg and (r["family"] == "depth" and abs(r["level"] - l) < 1e-9 or r["family"] == "control" and l == 0)]) * 100 for l in L_], "-o", color=col)
for a_, t_ in zip(ax, ("a_close [cm]", "palm contact [%]", "task success [%]")): a_.set_xlabel("extra TCP-forward grasp shift [cm]"); a_.set_ylabel(t_)
ax[0].legend(); plt.tight_layout(); plt.savefig(OUT / "pilot_plots.png", dpi=120)
print(json.dumps({k: res[k] for k in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "all_pass")}, indent=1, default=str)[:7000])
