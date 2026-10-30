#!/usr/bin/env python3
"""H2/H3 diagnostics, step 2: analysis of the replays written by eval/h23_replay.py (existing dev rollouts; no training, no new protocol, no test seeds).
H2: the scripted expert deliberately grasps with the TCP 1 cm ahead of the box centre; do C0/S1 reproduce it, amplify it, or drift?
H3: what produces the tight cluster of S1 placements at x - x_tray ~ -3.5 cm (grasp position vs contact mechanics vs release geometry)?
Frames (kept separate everywhere):  WORLD = lab frame (x = approach direction of the fingers, tray centre (0.44, 0.30));  TCP-relative = box centre in the TCP frame,
a = along the fingers (approach), c = across the jaws (closing axis), v = vertical;  BOX-relative = TCP in the box frame (longitudinal along the 5 cm side).
Exploratory/post-hoc.  Writes outputs/h23/h23_results.json, h23_tables.md, plots P1..P6.   Run (ACT venv): python eval/h23_analysis.py"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path: sys.path.insert(0, p)
from eval.metrics import task_labels
OUT = ROOT / "outputs/h23"; REP = OUT / "replay"; TRAY = np.array([0.44, 0.30]); TRAY_TOP = 0.36 + 0.004
meta = json.loads((OUT / "replay_meta.json").read_text()); JOBS = meta["jobs"]
BOX_HALF = (0.025, 0.020)       # half sizes of the box on the table plane (long side 5 cm along the box x axis, short side 4 cm)

def ev(R):
    g = R["gripper_cmd"]; box = R["box_pos"]; n = len(g)
    cl = np.nonzero(g < 0.028)[0]; tc = int(cl[0]) if len(cl) else None
    rise = box[:, 2] - box[0, 2]; held = (np.linalg.norm(box - R["tcp_pos"], axis=1) < 0.06) & (rise > 0.03)
    tl = int(np.argmax(rise > 0.03)) if (rise > 0.03).any() else None
    tr = None
    if tc is not None:
        r = np.nonzero(g[tc:] > 0.035)[0]; tr = tc + int(r[0]) if len(r) else None
    ev = dict(close=tc, lift=tl, release=tr)
    if tc is not None: ev["hold"] = min(tc + 14, n - 1)
    if tl is not None and tr is not None and tr > tl: ev["mid"] = (tl + tr) // 2
    return ev

ROWS = []
for j in JOBS:
    R = dict(np.load(REP / f"{j['agent']}_{j['S']}_{j['idx']:03d}.npz")); T = len(R["box_pos"]); sp = R["box_speed"]
    lab = task_labels(R["box_pos"], R["tcp_pos"], sp, TRAY, TRAY_TOP); e = ev(R); fin = R["box_pos"][-1, :2] - TRAY
    r = dict(j, **lab, final=fin.tolist(), events=e, R=R)
    post_fail = lab["pick_success"] and not lab["task_success"]
    cluster = bool(post_fail and fin[0] < -0.025 and abs(fin[1]) < 0.01)
    r["group"] = "success" if lab["task_success"] else ("cluster" if cluster else ("postpick_other" if post_fail else "pick_fail"))
    ROWS.append(r)
EXP = [r for r in ROWS if r["arm"] == "expert"]; POL = [r for r in ROWS if r["arm"] != "expert"]
def at(r, name, key, comp=None):
    t = r["events"].get(name)
    if t is None: return np.nan
    v = r["R"][key][t]; return float(v if comp is None else v[comp])
def A_(r, name): return at(r, name, "box_in_tcp", 2) * 100        # cm along the fingers (approach); + = box centre ahead of TCP
def C_(r, name): return at(r, name, "box_in_tcp", 1) * 100
def V_(r, name): return at(r, name, "box_in_tcp", 0) * 100
def dpsi(r, name):
    t = r["events"].get(name)
    if t is None: return np.nan
    d = r["R"]["box_yaw"][t] - r["R"]["hand_yaw"][t]; return float(np.degrees((d + np.pi) % (2 * np.pi) - np.pi))
def ms(x): x = np.array([v for v in x if v == v], float); return None if len(x) == 0 else dict(n=int(len(x)), mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0, p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)))

GROUPS = {"expert": lambda r: r["arm"] == "expert" and r["task_success"], "C0 success": lambda r: r["arm"] == "c0" and r["group"] == "success", "S1 success": lambda r: r["arm"] == "s1" and r["group"] == "success",
          "C0 post-pick fail": lambda r: r["arm"] == "c0" and r["group"] in ("cluster", "postpick_other"), "S1 cluster (x~-3.5 cm)": lambda r: r["arm"] == "s1" and r["group"] == "cluster",
          "S1 other post-pick fail": lambda r: r["arm"] == "s1" and r["group"] == "postpick_other"}
res = dict(replay_check=meta["replay_vs_logged_box_max_abs_err_m"], counts={a: dict(Counter(r["group"] for r in POL if r["arm"] == a)) for a in ("c0", "s1")}, expert=dict(n=len(EXP), success=sum(r["task_success"] for r in EXP)))
# consistency of re-derived labels with logged outcomes
lg = []
for r in POL:
    d = ROOT / "outputs/eval" / ("e6b" if r["tseed"] == 0 else "e0") / (r["arm"] if r["tseed"] == 0 else f"{r['arm']}_seed{r['tseed']}")
    tag, cond = {"NOM": ("nom21", "nominal"), "B21": ("bound21", "unseen_pose"), "B22": ("bound22", "unseen_pose")}[r["S"]]
    lg.append((d, tag, cond))
agree = 0; tot = 0; cache = {}
for r, (d, tag, cond) in zip(POL, lg):
    k = (d, tag, cond)
    if k not in cache: cache[k] = {e["episode_index"]: e for e in (json.loads(l) for l in (d / tag / "act" / cond / "episodes.jsonl").read_text().splitlines() if l.strip())}
    tot += 1; agree += int(bool(cache[k][r["idx"]]["task_success"]) == r["task_success"])
res["label_agreement_with_logged_task_success"] = [agree, tot]

# ---------------- T1: grasp-relative box pose per group and event (TCP-relative, cm)
EVN = ["close", "hold", "lift", "mid", "release"]; T1 = {}
for gn, f in GROUPS.items():
    sel = [r for r in ROWS if f(r)]
    T1[gn] = dict(n=len(sel), **{f"{k}_{e}": ms([fn(r, e) for r in sel]) for e in EVN for k, fn in (("a", A_), ("c", C_), ("v", V_))}, **{f"dpsi_deg_{e}": ms([dpsi(r, e) for r in sel]) for e in ("close", "release")})
res["T1_tcp_relative_by_group"] = T1
# phase changes of a
T3 = {}
for gn, f in GROUPS.items():
    sel = [r for r in ROWS if f(r)]
    T3[gn] = {"close->hold": ms([A_(r, "hold") - A_(r, "close") for r in sel]), "hold->lift": ms([A_(r, "lift") - A_(r, "hold") for r in sel]), "lift->release": ms([A_(r, "release") - A_(r, "lift") for r in sel]),
              "close->release": ms([A_(r, "release") - A_(r, "close") for r in sel]), "box_shift_pre_close_cm": ms([100 * np.linalg.norm(r["R"]["box_pos"][r["events"]["close"], :2] - r["R"]["box_pos"][0, :2]) if r["events"].get("close") is not None else np.nan for r in sel])}
res["T3_phase_changes_of_a_cm"] = T3
# H2: grasp offset along the fingers at closing, all episodes with a close event; expert vs C0 vs S1 (seed level)
H2 = dict(expert_all_cm=ms([A_(r, "close") for r in EXP]), expert_design_cm=-1.0, note="ScriptedExpert.segments(): grasp = box centre + 0.01*f  (f = box long axis) => box centre is 1.0 cm BEHIND the TCP by construction")
for arm in ("c0", "s1"):
    H2[arm] = {str(s): ms([A_(r, "close") for r in POL if r["arm"] == arm and r["tseed"] == s]) for s in (0, 1, 2)}
    H2[arm + "_hold"] = {str(s): ms([A_(r, "hold") for r in POL if r["arm"] == arm and r["tseed"] == s]) for s in (0, 1, 2)}
    H2[arm + "_success_only_close"] = ms([A_(r, "close") for r in POL if r["arm"] == arm and r["group"] == "success"])
H2["expert_by_set"] = {S: ms([A_(r, "close") for r in EXP if r["S"] == S]) for S in ("NOM", "B21", "B22")}
H2["delta_vs_expert_cm_seed_means"] = {arm: [float(np.nanmean([A_(r, "close") for r in POL if r["arm"] == arm and r["tseed"] == s]) - np.nanmean([A_(r, "close") for r in EXP])) for s in (0, 1, 2)] for arm in ("c0", "s1")}
res["H2_grasp_offset_vs_expert"] = H2

# ---------------- contact mechanics at hold / mid / release
def stats_at(r, name, key):
    t = r["events"].get(name); return np.nan if t is None else float(r["R"][key][t])
T4 = {}
for gn, f in GROUPS.items():
    sel = [r for r in ROWS if f(r)]; T4[gn] = {}
    for e in ("hold", "mid", "release"):
        T4[gn][e] = dict(finger_force_N=ms([stats_at(r, e, "finger_force") for r in sel]), finger_contacts=ms([stats_at(r, e, "finger_contacts") for r in sel]), contact_a_cm=ms([100 * stats_at(r, e, "finger_contact_a") for r in sel]),
                         hand_contact_frac=float(np.mean([stats_at(r, e, "hand_force") > 0.01 for r in sel])), tilt_deg=ms([stats_at(r, e, "box_tilt") for r in sel]), gripper_q_mm=ms([1000 * stats_at(r, e, "gripper_q") for r in sel]))
res["T4_contacts"] = T4
# any hand/palm contact during the whole hold for the cluster?
res["hand_contact_any_during_close_to_release"] = {gn: float(np.mean([bool((r["R"]["hand_force"][r["events"]["close"]:(r["events"]["release"] or len(r["R"]["hand_force"]))] > 0.01).any()) for r in [x for x in ROWS if f(x)] if r["events"].get("close") is not None])) for gn, f in GROUPS.items()}

# ---------------- finger / palm extents along the approach axis (mesh vertices in the TCP frame, open pose at the start of an expert replay)
import mujoco
from sim.lab_env import LabEnv
env = LabEnv(cameras=()); m, d = env.m, env.d; env.set_perturbation(); env.reset((0.41, 0.10), 0.0)
Rt = d.xmat[env.tcp_id].reshape(3, 3); Tp = d.xpos[env.tcp_id].copy(); ext = {}
for nm in ("openarm_left_left_finger_collision", "openarm_left_right_finger_collision", "openarm_left_hand_collision"):
    g = m.geom(nm).id; mid = m.geom_dataid[g]; V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    W = d.geom_xpos[g] + V @ d.geom_xmat[g].reshape(3, 3).T; P = (W - Tp) @ Rt                      # columns: x_tcp (vertical), y_tcp (across), z_tcp (approach)
    ext[nm] = dict(a_cm=[float(100 * P[:, 2].min()), float(100 * P[:, 2].max())], c_cm=[float(100 * P[:, 1].min()), float(100 * P[:, 1].max())], v_cm=[float(100 * P[:, 0].min()), float(100 * P[:, 0].max())])
res["gripper_geometry_tcp_frame_open_pose"] = ext
FING_A = (min(ext["openarm_left_left_finger_collision"]["a_cm"][0], ext["openarm_left_right_finger_collision"]["a_cm"][0]), max(ext["openarm_left_left_finger_collision"]["a_cm"][1], ext["openarm_left_right_finger_collision"]["a_cm"][1]))
HAND_A = tuple(ext["openarm_left_hand_collision"]["a_cm"])

# ---------------- pose dependence
def sp(x, y):
    ok = [(a, b) for a, b in zip(x, y) if a == a and b == b]
    if len(ok) < 5: return None
    r, p = st.spearmanr(*zip(*ok)); return dict(rho=float(r), p=float(p), n=len(ok))
T5 = {}
for arm in ("expert", "c0", "s1"):
    sel = [r for r in ROWS if r["arm"] == arm]
    T5[arm] = dict(a_close_vs=dict(box_x=sp([r["box_xy"][0] for r in sel], [A_(r, "close") for r in sel]), box_y=sp([r["box_xy"][1] for r in sel], [A_(r, "close") for r in sel]), yaw=sp([r["box_yaw"] for r in sel], [A_(r, "close") for r in sel])),
                   a_hold_vs=dict(yaw=sp([r["box_yaw"] for r in sel], [A_(r, "hold") for r in sel])), c_close_vs_yaw=sp([r["box_yaw"] for r in sel], [C_(r, "close") for r in sel]))
# cluster vs the other S1 post-pick episodes: initial pose
s1pp = [r for r in POL if r["arm"] == "s1" and r["pick_success"]]
cl = [r for r in s1pp if r["group"] == "cluster"]; nc = [r for r in s1pp if r["group"] != "cluster"]
T5["s1_cluster_vs_rest_pose"] = {k: dict(cluster=ms([r[k] if k == "box_yaw" else r["box_xy"][0 if k == "x" else 1] for r in cl]), rest=ms([r[k] if k == "box_yaw" else r["box_xy"][0 if k == "x" else 1] for r in nc]),
                                         mwu_p=float(st.mannwhitneyu([r[k] if k == "box_yaw" else r["box_xy"][0 if k == "x" else 1] for r in cl], [r[k] if k == "box_yaw" else r["box_xy"][0 if k == "x" else 1] for r in nc]).pvalue)) for k in ("x", "y", "box_yaw")}
T5["s1_cluster_vs_rest_a_close"] = dict(cluster=ms([A_(r, "close") for r in cl]), rest=ms([A_(r, "close") for r in nc]), mwu_p=float(st.mannwhitneyu([A_(r, "close") for r in cl], [A_(r, "close") for r in nc]).pvalue))
T5["cluster_rate_by_set_per_arm"] = {arm: {S: [sum(r["group"] == "cluster" for r in POL if r["arm"] == arm and r["S"] == S), sum(1 for r in POL if r["arm"] == arm and r["S"] == S and r["pick_success"])] for S in ("NOM", "B21", "B22")} for arm in ("c0", "s1")}
yb = np.array([r["box_yaw"] for r in POL]); edges = np.percentile(yb, [0, 33.3, 66.7, 100]); T5["cluster_rate_by_yaw_tercile_s1"] = [[float(edges[i]), float(edges[i + 1]), sum(1 for r in s1pp if edges[i] <= r["box_yaw"] <= edges[i + 1] and r["group"] == "cluster"), sum(1 for r in s1pp if edges[i] <= r["box_yaw"] <= edges[i + 1])] for i in range(3)]
T5["box_pose_ranges"] = {S: dict(x=[float(min(r["box_xy"][0] for r in EXP if r["S"] == S)), float(max(r["box_xy"][0] for r in EXP if r["S"] == S))], yaw=[float(min(r["box_yaw"] for r in EXP if r["S"] == S)), float(max(r["box_yaw"] for r in EXP if r["S"] == S))]) for S in ("NOM", "B21", "B22")}
res["T5_pose_dependence"] = T5

# ---------------- which frame is the cluster 'fixed' in?  (spread of the same quantity in different frames, S1 cluster)
fx = np.array([r["final"] for r in cl]) * 100
rel_a = np.array([A_(r, "release") for r in cl]); rel_c = np.array([C_(r, "release") for r in cl]); hold_a = np.array([A_(r, "hold") for r in cl]); close_a = np.array([A_(r, "close") for r in cl])
aim = np.array([(r["R"]["tcp_pos"][r["events"]["release"], :2] - TRAY) * 100 for r in cl])
tcp_box = np.array([r["R"]["tcp_in_box"][r["events"]["hold"]] * 100 for r in cl])
res["T6_cluster_frames"] = dict(n=len(cl), final_world_x_cm=ms(fx[:, 0]), final_world_y_cm=ms(fx[:, 1]), tcp_aim_world_cm=dict(x=ms(aim[:, 0]), y=ms(aim[:, 1])), box_in_tcp_a_at_close=ms(close_a), box_in_tcp_a_at_hold=ms(hold_a), box_in_tcp_a_at_release=ms(rel_a), box_in_tcp_c_at_release=ms(rel_c),
                                 tcp_in_box_longitudinal_at_hold_cm=ms(tcp_box[:, 0]), tcp_in_box_lateral_at_hold_cm=ms(tcp_box[:, 1]),
                                 corr_final_x_with_a_release=float(np.corrcoef(fx[:, 0], rel_a)[0, 1]), corr_final_x_with_a_close=float(np.corrcoef(fx[:, 0], close_a)[0, 1]),
                                 palm_front_a_cm=HAND_A[1], finger_a_range_cm=list(FING_A), box_half_extent_along_fingers_cm=[100 * BOX_HALF[0], 100 * BOX_HALF[1]])
# geometric attractor test: distance between box rear edge/front edge and finger tip / palm front at hold
def edge_gap(r, name):
    t = r["events"].get(name)
    if t is None: return (np.nan, np.nan)
    a = r["R"]["box_in_tcp"][t, 2] * 100; dp = np.radians(dpsi(r, name)); h = 100 * (BOX_HALF[0] * abs(np.cos(dp)) + BOX_HALF[1] * abs(np.sin(dp))); return (a + h, a - h)        # front / rear edge along the fingers (cm in TCP frame)
res["T7_box_edges_along_fingers_cm_at_hold"] = {gn: dict(front=ms([edge_gap(r, "hold")[0] for r in ROWS if f(r)]), rear=ms([edge_gap(r, "hold")[1] for r in ROWS if f(r)])) for gn, f in GROUPS.items()}
# final-vs-hold box displacement (post-grasp motion) for the cluster, in the world frame
res["T6b_cluster_world_shift_hold_to_final_cm"] = dict(x=ms([100 * (r["R"]["box_pos"][-1, 0] - r["R"]["box_pos"][r["events"]["hold"], 0]) for r in cl]), y=ms([100 * (r["R"]["box_pos"][-1, 1] - r["R"]["box_pos"][r["events"]["hold"], 1]) for r in cl]))
# H3 attribution of the final offset (cluster vs successes): final = aim + box_in_tcp(release) projected to the world; use world vectors
def world_parts(r):
    t = r["events"]["release"]; Rb = r["R"]; aimv = (Rb["tcp_pos"][t, :2] - TRAY) * 100; inh = (Rb["box_pos"][t, :2] - Rb["tcp_pos"][t, :2]) * 100; dr = (Rb["box_pos"][-1, :2] - Rb["box_pos"][t, :2]) * 100; return aimv, inh, dr
for gn in ("expert", "S1 success", "C0 success", "S1 cluster (x~-3.5 cm)"):
    sel = [r for r in ROWS if GROUPS[gn](r) and r["events"].get("release") is not None]; W = [world_parts(r) for r in sel]
    res.setdefault("T8_world_decomposition_cm", {})[gn] = dict(n=len(sel), aim_x=ms([w[0][0] for w in W]), inhand_x=ms([w[1][0] for w in W]), drift_x=ms([w[2][0] for w in W]), final_x=ms([100 * r["final"][0] for r in sel]))

# ---------------- T9: is the cluster an extrapolation effect at the near edge of the workspace? (training region x in [0.38, 0.45]; evaluation 'unseen' region x in [0.36, 0.47])
XB = [(0.36, 0.38), (0.38, 0.40), (0.40, 0.42), (0.42, 0.45), (0.45, 0.47)]; T9 = {}
for arm in ("expert", "c0", "s1"):
    T9[arm] = []
    for lo, hi in XB:
        sel = [r for r in ROWS if r["arm"] == arm and lo <= r["box_xy"][0] < hi]
        T9[arm].append(dict(x_bin=[lo, hi], n=len(sel), success=sum(r["task_success"] for r in sel), pick=sum(r["pick_success"] for r in sel), cluster=sum(r["group"] == "cluster" for r in sel), a_close_cm=ms([A_(r, "close") for r in sel]), a_release_cm=ms([A_(r, "release") for r in sel]),
                            hand_contact=sum(bool((r["R"]["hand_force"][r["events"]["close"]:(r["events"]["release"] or len(r["R"]["hand_force"]))] > 0.01).any()) for r in sel if r["events"].get("close") is not None)))
res["T9_by_box_x_bin"] = T9
near = [r for r in POL if r["box_xy"][0] < 0.38]
res["T9b_near_edge_x_lt_0p38"] = {arm: dict(n=sum(r["arm"] == arm for r in near), by_seed={str(s): dict(n=sum(1 for r in near if r["arm"] == arm and r["tseed"] == s), cluster=sum(1 for r in near if r["arm"] == arm and r["tseed"] == s and r["group"] == "cluster"), success=sum(1 for r in near if r["arm"] == arm and r["tseed"] == s and r["task_success"]),
                                           a_close_cm_mean=float(np.nanmean([A_(r, "close") for r in near if r["arm"] == arm and r["tseed"] == s])) if any(r["arm"] == arm and r["tseed"] == s for r in near) else None) for s in (0, 1, 2)}) for arm in ("c0", "s1")}
res["T9c_all_cluster_episodes_x_lt_0p38"] = [sum(r["box_xy"][0] < 0.38 for r in cl), len(cl)]
# expert-policy difference in grasp depth as a function of x
res["T9d_a_close_vs_x_slope_cm_per_cm"] = {arm: float(np.polyfit([100 * r["box_xy"][0] for r in ROWS if r["arm"] == arm and r["events"].get("close") is not None], [A_(r, "close") for r in ROWS if r["arm"] == arm and r["events"].get("close") is not None], 1)[0]) for arm in ("expert", "c0", "s1")}

# per-seed success split by region of the box x (in-distribution x >= 0.38 vs near edge x < 0.38), paired by training seed
T9e = {}
for nm, sel_f in (("in_dist_x_ge_0p38", lambda r: r["box_xy"][0] >= 0.38), ("near_edge_x_lt_0p38", lambda r: r["box_xy"][0] < 0.38)):
    T9e[nm] = {arm: [[sum(1 for r in POL if r["arm"] == arm and r["tseed"] == s and sel_f(r) and r["task_success"]), sum(1 for r in POL if r["arm"] == arm and r["tseed"] == s and sel_f(r))] for s in (0, 1, 2)] for arm in ("c0", "s1")}
    c, s1_ = np.array([100 * a / b for a, b in T9e[nm]["c0"]]), np.array([100 * a / b for a, b in T9e[nm]["s1"]]); dd = s1_ - c; tt = st.t.ppf(0.975, 2) * dd.std(ddof=1) / np.sqrt(3)
    T9e[nm]["per_seed_pct"] = dict(c0=c.tolist(), s1=s1_.tolist(), diff=dd.tolist(), paired_mean=float(dd.mean()), paired_ci95=[float(dd.mean() - tt), float(dd.mean() + tt)])
res["T9e_success_by_region_per_seed"] = T9e

# ---------------- plots
COLS = {"expert": "k", "C0 success": "C0", "S1 success": "C1", "S1 cluster (x~-3.5 cm)": "C3", "C0 post-pick fail": "C9"}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for gn, col in COLS.items():
    sel = [r for r in ROWS if GROUPS[gn](r) and r["events"].get("close") is not None]
    if not sel: continue
    W = np.arange(-10, 90); M = np.full((len(sel), len(W)), np.nan)
    for i, r in enumerate(sel):
        tc = r["events"]["close"]; a = r["R"]["box_in_tcp"][:, 2] * 100
        for k, w in enumerate(W):
            if 0 <= tc + w < len(a): M[i, k] = a[tc + w]
    mu = np.nanmean(M, 0); sd = np.nanstd(M, 0); ax[0].plot(W / 20, mu, color=col, label=f"{gn} (n={len(sel)})"); ax[0].fill_between(W / 20, mu - sd, mu + sd, color=col, alpha=.12)
ax[0].set_xlabel("time since gripper-close command [s]"); ax[0].set_ylabel("box centre − TCP along fingers (a) [cm]"); ax[0].set_title("Grasp-relative box position (TCP frame)"); ax[0].legend(fontsize=7); ax[0].axhline(-1, color="gray", ls=":")
for gn, col in COLS.items():
    sel = [r for r in ROWS if GROUPS[gn](r) and r["events"].get("close") is not None]
    ax[1].hist([A_(r, "close") for r in sel], bins=np.linspace(-6, 3, 37), color=col, alpha=.45, label=gn, density=True)
ax[1].set_xlabel("a at close [cm]"); ax[1].set_title("Offset along the fingers at the close command"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "P1_grasp_offset_traces.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
for k, e in enumerate(("close", "hold", "mid", "release")):
    for i, (gn, col) in enumerate(COLS.items()):
        sel = [r for r in ROWS if GROUPS[gn](r) and r["events"].get(e) is not None]
        if not sel: continue
        a = np.array([A_(r, e) for r in sel]); h = np.array([100 * (BOX_HALF[0] * abs(np.cos(np.radians(dpsi(r, e)))) + BOX_HALF[1] * abs(np.sin(np.radians(dpsi(r, e))))) for r in sel])
        ax[k].barh(i, 2 * np.nanmean(h), left=np.nanmean(a) - np.nanmean(h), color=col, alpha=.7, height=.6)
    ax[k].axvspan(FING_A[0], FING_A[1], color="gray", alpha=.2); ax[k].axvline(0, color="k", lw=1); ax[k].axvline(HAND_A[1], color="brown", lw=1, ls="--")
    ax[k].set_title(e); ax[k].set_xlabel("position along fingers [cm] (0 = TCP/fingertip)"); ax[k].set_xlim(-14, 4)
ax[0].set_yticks(range(len(COLS))); ax[0].set_yticklabels(list(COLS), fontsize=7); plt.suptitle("Box extent (coloured, mean) vs finger extent (grey) along the approach axis; dashed = palm collision front", fontsize=9); plt.tight_layout(); plt.savefig(OUT / "P2_finger_box_overlap.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for gn, col in COLS.items():
    sel = [r for r in ROWS if GROUPS[gn](r) and r["events"].get("mid") is not None]
    if not sel: continue
    ax[0].scatter([r["R"]["finger_force"][r["events"]["hold"]] for r in sel], [A_(r, "hold") for r in sel], s=8, color=col, alpha=.5, label=gn)
    ax[1].scatter([100 * r["R"]["finger_contact_a"][r["events"]["hold"]] for r in sel], [A_(r, "hold") for r in sel], s=8, color=col, alpha=.5)
    ax[2].scatter([r["R"]["box_tilt"][r["events"]["mid"]] for r in sel], [A_(r, "mid") for r in sel], s=8, color=col, alpha=.5)
ax[0].set_xlabel("finger normal force at hold [N]"); ax[1].set_xlabel("mean finger contact position a at hold [cm]"); ax[2].set_xlabel("box tilt during transport [deg]")
for x in ax: x.set_ylabel("box a-offset [cm]")
ax[0].legend(fontsize=6); plt.tight_layout(); plt.savefig(OUT / "P3_contact_mechanics.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for arm, col in (("expert", "k"), ("c0", "C0"), ("s1", "C1")):
    sel = [r for r in ROWS if r["arm"] == arm and r["events"].get("close") is not None]
    for k, (lab, fn) in enumerate((("box x [m]", lambda r: r["box_xy"][0]), ("box y [m]", lambda r: r["box_xy"][1]), ("box yaw [rad]", lambda r: r["box_yaw"]))):
        ax[k].scatter([fn(r) for r in sel], [A_(r, "close") for r in sel], s=6, color=col, alpha=.35, label=arm); ax[k].set_xlabel(lab); ax[k].set_ylabel("a at close [cm]")
for r in cl: ax[2].scatter(r["box_yaw"], A_(r, "close"), s=22, facecolors="none", edgecolors="C3")
ax[0].legend(); plt.tight_layout(); plt.savefig(OUT / "P4_pose_dependence.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
for gn, col in COLS.items():
    sel = [r for r in ROWS if GROUPS[gn](r) and r["events"].get("release") is not None]
    ax[0].scatter([100 * r["final"][0] for r in sel], [100 * r["final"][1] for r in sel], s=8, color=col, alpha=.5, label=gn)
    ax[1].scatter([A_(r, "release") for r in sel], [C_(r, "release") for r in sel], s=8, color=col, alpha=.5)
    ax[2].scatter([A_(r, "close") for r in sel], [A_(r, "release") for r in sel], s=8, color=col, alpha=.5)
ax[0].set_xlim(-8, 8); ax[0].set_ylim(-8, 8); th = np.linspace(0, 2 * np.pi, 100); ax[0].plot(3 * np.cos(th), 3 * np.sin(th), "k--"); ax[0].set_xlabel("final box x − tray x [cm] (WORLD)"); ax[0].set_ylabel("y [cm]"); ax[0].set_aspect("equal"); ax[0].legend(fontsize=6)
ax[1].set_xlabel("a at release [cm] (TCP frame)"); ax[1].set_ylabel("c at release [cm]"); ax[2].set_xlabel("a at close [cm]"); ax[2].set_ylabel("a at release [cm]"); ax[2].plot([-6, 3], [-6, 3], "k:")
plt.tight_layout(); plt.savefig(OUT / "P5_release_geometry.png", dpi=130); plt.close()

# ---------------- P6: rendered close-ups of one paired configuration (expert vs C0 success vs S1 cluster failure), events close/hold/lift/release
key = lambda r: (r["S"], r["idx"], r["seed"])
byk = defaultdict(dict)
for r in ROWS: byk[key(r)][r["agent"]] = r
pick = None
for k, D in byk.items():
    for ts in (1, 2, 0):
        a, b = D.get(f"s1_s{ts}"), D.get(f"c0_s{ts}")
        if a is not None and b is not None and a["group"] == "cluster" and b["group"] == "success" and D["expert"]["task_success"]: pick = (k, ts); break
    if pick: break
res["render_example"] = dict(config=list(pick[0]), training_seed=pick[1]) if pick else None
if pick:
    k, ts = pick; D = byk[k]; panels = [("expert", D["expert"]), (f"C0 seed {ts} (success)", D[f"c0_s{ts}"]), (f"S1 seed {ts} (cluster failure)", D[f"s1_s{ts}"])]
    rend = mujoco.Renderer(m, 360, 360); cam = mujoco.MjvCamera(); opt = mujoco.MjvOption(); fig, axs = plt.subplots(6, 4, figsize=(13, 19))
    from sim.lab_env import LabEnv as _L
    for pi, (lab, r) in enumerate(panels):
        env.set_perturbation(); env.reset(tuple(r["box_xy"]), r["box_yaw"]); A = r["R"]["action"]; want = {r["events"][e]: e for e in ("close", "hold", "lift", "release") if r["events"].get(e) is not None}
        for t in range(len(A) + 1):
            if t in want:
                col = ["close", "hold", "lift", "release"].index(want[t])
                for vi, (az, el) in enumerate(((90, -89.0), (90, 0.0))):
                    cam.lookat[:] = d.xpos[env.tcp_id] + np.array([-0.04, 0, 0]) if want[t] != "release" else d.xpos[env.tcp_id]; cam.distance = 0.24; cam.azimuth = az; cam.elevation = el
                    rend.update_scene(d, cam, opt); axs[pi * 2 + vi, col].imshow(rend.render()); axs[pi * 2 + vi, col].axis("off"); axs[pi * 2 + vi, col].set_title(f"{lab}\n{want[t]} (t={t}) {'top' if vi == 0 else 'side'}", fontsize=8)
            if t < len(A): env.step(A[t])
    plt.tight_layout(); plt.savefig(OUT / "P6_render_close_lift_release.png", dpi=90); plt.close()

# ---------------- compact summary tables
def f2(v): return "—" if v is None else f"{v['mean']:+.2f}±{v['sd']:.2f}"
L = ["| group | n | a close | a hold | a lift | a mid | a release | c close | c release | Δψ hand−box close [°] |", "|---|---|---|---|---|---|---|---|---|---|"]
for gn, v in T1.items(): L.append(f"| {gn} | {v['n']} | {f2(v['a_close'])} | {f2(v['a_hold'])} | {f2(v['a_lift'])} | {f2(v['a_mid'])} | {f2(v['a_release'])} | {f2(v['c_close'])} | {f2(v['c_release'])} | {f2(v['dpsi_deg_close'])} |")
(OUT / "h23_tables.md").write_text("\n".join(L)); (OUT / "h23_results.json").write_text(json.dumps(res, indent=1, default=float))
print("\n".join(L)); print(json.dumps({k: res[k] for k in ("T9_by_box_x_bin", "T9b_near_edge_x_lt_0p38", "T9c_all_cluster_episodes_x_lt_0p38", "T9d_a_close_vs_x_slope_cm_per_cm", "replay_check", "counts", "expert", "label_agreement_with_logged_task_success", "gripper_geometry_tcp_frame_open_pose")}, indent=1))
