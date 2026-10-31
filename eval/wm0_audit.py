#!/usr/bin/env python3
"""WM-0 data audit for the planned Action-Conditioned World Model (analysis only: reads existing rollouts/replays/demos, writes outputs/wm0/*; no training, no simulator collection,
no ACT/upstream change).  Verifies: (1) content/units of the candidate 29-D state, (2) 16-D actuator ordering and executed-action semantics, (3) state/action/timestamp synchronisation,
(4) unique episodes / duplicates / replay and contact-label coverage, (5) train/val/test leakage, (6) use of privileged simulator state, (7) H=4,K=20 window rules (+ unit tests).
Run (ACT venv): python eval/wm0_audit.py"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, hashlib, re, numpy as np
from pathlib import Path
from collections import Counter, defaultdict
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path: sys.path.insert(0, p)
OUT = ROOT / "outputs/wm0"; OUT.mkdir(parents=True, exist_ok=True)
H, K, STRIDE = 4, 20, 5
def jl(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def sha(a): return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

# ---------------------------------------------------------------- 0. catalog of closed-loop rollouts
def source_of(d):
    s = str(d.relative_to(ROOT / "outputs")); parts = s.split("/")
    if parts[0] == "cov": return ("debug_smoke" if parts[1] == "smoke" else f"cov/{parts[2]}"), "/".join(parts[3:])
    if parts[0] == "xsweep": return f"xsweep/{parts[2]}", "/".join(parts[3:])
    if parts[:2] == ["eval", "e6b"]: return f"e6b/{parts[2]}", "/".join(parts[3:])
    if parts[:2] == ["eval", "e0"]: return f"e0/{parts[2]}", "/".join(parts[3:])
    if parts[:2] == ["eval", "audit_v1"]: return f"audit_v1/{parts[2]}", "/".join(parts[3:])
    if parts[:2] == ["eval", "full_v1"]: return "full_v1", "/".join(parts[2:])
    if parts[:2] == ["eval", "overfit3_v1"] or parts[:2] == ["eval", "_old"]: return "debug_overfit", "/".join(parts[2:])
    return "other/" + "/".join(parts[:2]), s
def tags(tag):
    t = tag.lower(); return dict(state_perturbed=bool(re.search(r"pert_|diag_right|ind_", t)), visual_perturbed=bool(re.search(r"light|cam_shift", t)))
CAT = []
for rd in sorted((ROOT / "outputs").rglob("rollouts")):
    d = rd.parent
    if not (d / "episodes.jsonl").exists(): continue
    src, tag = source_of(d); eps = {e["episode_index"]: e for e in jl(d / "episodes.jsonl")}
    for f in sorted(rd.glob("ep*.npz")):
        idx = int(f.stem[2:]); e = eps.get(idx)
        CAT.append(dict(path=str(f.relative_to(ROOT)), dir=str(d.relative_to(ROOT)), source=src, tag=tag, agent=e["agent"] if e else None, idx=idx, seed=e["seed"] if e else None, region=e["region"] if e else None, box_xy=e["box_xy"] if e else None, box_yaw=e["box_yaw"] if e else None,
                        task_success=e["task_success"] if e else None, pick=e["pick_success"] if e else None, place=e["place_success"] if e else None, clipped=e.get("clipped") if e else None, slewed=e.get("slewed") if e else None, **tags(tag)))
res = dict(n_rollout_files=len(CAT), by_source_prefix=dict(Counter(c["source"].split("/")[0] for c in CAT)))
# ---------------------------------------------------------------- load arrays once
for c in CAT:
    z = np.load(ROOT / c["path"]); c["keys"] = sorted(z.files); c["shapes"] = {k: list(z[k].shape) for k in z.files}
    c["T"] = len(z["action"]); c["h_action"] = sha(z["action"]); c["h_box"] = sha(z["box_pos"]) if "box_pos" in z.files else None; c["finite"] = bool(all(np.isfinite(z[k]).all() for k in z.files))
    g_ = lambda k: z[k] if k in z.files else None; c["_a"], c["_s"], c["_b"], c["_t"], c["_v"] = g_("action"), g_("state"), g_("box_pos"), g_("tcp_pos"), g_("box_speed"); c["_raw"] = g_("raw_action")
DEBUG = lambda c: c["source"].startswith("debug")
MAIN = [c for c in CAT if not DEBUG(c)]
FULL_KEYS = {"state", "action", "box_pos", "tcp_pos", "box_speed"}
FULLM = [c for c in MAIN if FULL_KEYS <= set(c["keys"])]
res["schema_variants"] = {"|".join(k): v for k, v in Counter(tuple(c["keys"]) for c in MAIN).items()}; res["n_main_with_full_schema"] = len(FULLM); res["n_main_missing_fields_by_source"] = dict(Counter(c["source"].split("/")[0] for c in MAIN if c not in FULLM))
# ---------------------------------------------------------------- 1/2. content, units, ordering, executed-action semantics
from lab_env import LabEnv
env = LabEnv(cameras=()); ad = env.adapter
res["ordering"] = dict(state_names=list(ad.joint_names), action_names=list(ad.names), same_order=list(ad.joint_names) == [n.replace("_ctrl", "") for n in ad.names] if False else None,
                       action_range_lo=[float(x) for x in ad.limits.lo], action_range_hi=[float(x) for x in ad.limits.hi])
st = np.concatenate([c["_s"][:] for c in FULLM]); ac = np.concatenate([c["_a"] for c in FULLM])
res["state_stats"] = dict(min=st.min(0).tolist(), max=st.max(0).tolist(), mean=st.mean(0).tolist(), std=st.std(0).tolist())
res["action_stats"] = dict(min=ac.min(0).tolist(), max=ac.max(0).tolist(), mean=ac.mean(0).tolist(), std=ac.std(0).tolist())
res["action_inside_ctrlrange"] = bool(((ac >= np.array(res["ordering"]["action_range_lo"]) - 1e-6) & (ac <= np.array(res["ordering"]["action_range_hi"]) + 1e-6)).all())
# lag: which k minimises |state[t+k] - action[t]| per joint (arm joints, grippers), idle right arm excluded from lag (constant)
lags = {}
for j in range(16):
    err = {k: float(np.mean([np.abs(c["_s"][k:, j] - c["_a"][:len(c["_a"]) - k, j]).mean() for c in FULLM[::7] if c["_s"] is not None])) for k in range(0, 5)}
    lags[ad.joint_names[j]] = dict(best_lag=min(err, key=err.get), err_by_lag={str(k): v for k, v in err.items()})
res["state_action_lag_steps"] = lags
res["state_action_lag_summary_left_arm"] = dict(Counter(v["best_lag"] for k, v in lags.items() if "left" in k))
# executed vs raw policy output
rr = [c for c in FULLM if c["_raw"] is not None and c["agent"] == "act"]; d = np.concatenate([np.abs(c["_a"] - c["_raw"]) for c in rr]) if rr else np.zeros((1, 16))
res["raw_vs_executed"] = dict(n_with_raw=len(rr), frac_steps_any_diff_gt_1e6=float((d.max(1) > 1e-6).mean()), max_abs_diff=float(d.max()), mean_abs_diff_per_dim=d.mean(0).tolist(),
                              clipped_slewed_counters=dict(clipped_mean=float(np.mean([c["clipped"] for c in rr if c["clipped"] is not None])), slewed_mean=float(np.mean([c["slewed"] for c in rr if c["slewed"] is not None]))))
# executed-action evidence: replay with the logged actions reproduces the logged box trajectory (see replay section)
# gravity-comp offset on the idle right arm (action - state)
idle = np.concatenate([c["_a"][:-1, 8:16] - c["_s"][1:, 8:16] for c in FULLM if c["_s"] is not None])
res["idle_right_arm_action_minus_next_state"] = dict(mean=idle.mean(0).tolist(), std=idle.std(0).tolist())
# ---------------------------------------------------------------- 3. lengths / synchronisation / timestamps
Tc = Counter(c["T"] for c in FULLM); shape_ok = all(c["shapes"]["state"][0] == c["T"] == c["shapes"]["action"][0] and c["shapes"]["box_pos"][0] == c["T"] + 1 and c["shapes"]["tcp_pos"][0] == c["T"] + 1 and c["shapes"]["box_speed"][0] == c["T"] + 1 for c in FULLM)
res["lengths"] = dict(T_histogram={str(k): v for k, v in sorted(Tc.items())}, shapes_consistent_state_T_action_T_box_T_plus_1=bool(shape_ok), all_finite=bool(all(c["finite"] for c in FULLM)), keysets=dict(Counter(tuple(c["keys"]) for c in FULLM).most_common(5)) and {"|".join(k): v for k, v in Counter(tuple(c["keys"]) for c in FULLM).items()})
# box_speed vs finite difference of box_pos (same time index?)
cors = {}
sel = [c for c in FULLM if c["task_success"]][::15]
for k in (-1, 0, 1):
    v = []
    for c in sel:
        fd = np.linalg.norm(np.diff(c["_b"], axis=0), axis=1) * 20; sp = c["_v"][1:]
        n = len(fd); a_, b_ = (fd[max(0, -k):n - max(0, k)], sp[max(0, k):n - max(0, -k)]) if True else (fd, sp)
        m = min(len(a_), len(b_));
        if m > 10 and a_[:m].std() > 0 and b_[:m].std() > 0: v.append(np.corrcoef(a_[:m], b_[:m])[0, 1])
    cors[str(k)] = float(np.mean(v))
res["box_speed_vs_finite_difference_corr_by_shift"] = cors
demo_ts = []
for f in sorted((ROOT / "outputs/datasets/demo_train_clipped").glob("episode_*.npz"))[:20]:
    z = np.load(f); demo_ts.append(float(np.abs(np.diff(z["timestamp"]) - 0.05).max()))
res["demo_timestamps_max_dev_from_0p05s"] = max(demo_ts); res["rollout_timestamps"] = "not stored: implicit 20 Hz, row t = observation before action t (state/action length T, box/tcp/speed length T+1 incl. final)"
# ---------------------------------------------------------------- 4. unique episodes, duplicates, replay & contact coverage
KEY = lambda c: (round(c["box_xy"][0], 6), round(c["box_xy"][1], 6), round(c["box_yaw"], 6))
res["counts"] = dict(total_files=len(CAT), debug_excluded=len(CAT) - len(MAIN), main_files=len(MAIN), unique_action_hash=len({c["h_action"] for c in MAIN}), unique_box_traj_hash=len({c["h_box"] for c in MAIN}),
                     unique_initial_states=len({KEY(c) for c in MAIN}), by_source=dict(Counter(c["source"] for c in MAIN)), state_perturbed=sum(c["state_perturbed"] for c in MAIN), visual_perturbed=sum(c["visual_perturbed"] for c in MAIN),
                     expert_rollouts=sum(c["agent"] == "expert" for c in MAIN), act_rollouts=sum(c["agent"] == "act" for c in MAIN))
grp = defaultdict(list)
for c in MAIN: grp[c["h_action"]].append(c["path"])
dups = {k: v for k, v in grp.items() if len(v) > 1}
res["exact_duplicates"] = dict(n_groups=len(dups), n_redundant_files=sum(len(v) - 1 for v in dups.values()), group_size_hist=dict(Counter(len(v) for v in dups.values())), examples=[v[:3] for v in list(dups.values())[:5]],
                               by_dir_pairs=dict(Counter(tuple(sorted({"/".join(p.split("/")[:4]) for p in v})) .__str__() for v in dups.values()).most_common(8)))
# replay inventory
def replay_keys():
    K_ = {}
    for base, meta in (("outputs/h23/replay", "outputs/h23/replay_meta.json"), ("outputs/xsweep/replay", "outputs/xsweep/replay_meta.json"), ("outputs/cov/replay", "outputs/cov/replay_meta.json")):
        for f in (ROOT / base).glob("*.npz"): K_[(base.split("/")[1], f.stem)] = str(f.relative_to(ROOT))
    return K_
RK = replay_keys(); res["replay_files"] = dict(total=len(RK), by_set=dict(Counter(k[0] for k in RK)), unique_stems_per_set={s: len({k[1] for k in RK if k[0] == s}) for s in {k[0] for k in RK}})
def replay_name(c):
    s = c["source"]; tg = c["tag"]
    if s.startswith("xsweep/"): arm, sd = re.match(r"(c0|s1)_seed(\d)", s.split("/")[1]).groups(); return ("xsweep", f"{arm}_s{sd}_SW_{c['idx']:03d}")
    if s.startswith("cov/"): br, sd = re.match(r"(.*)_seed(\d)", s.split("/")[1]).groups(); return ("cov", f"{br}_s{sd}_SW_{c['idx']:03d}") if tg.startswith("sweep") else None
    m = re.match(r"(e6b|e0)/(.*)", s)
    if m and tg.split("/")[0] in ("nom21", "bound21", "bound22"):
        who = m.group(2); arm, sd = (who, 0) if m.group(1) == "e6b" else re.match(r"(c0|s1)_seed(\d)", who).groups()
        if m.group(1) == "e6b" and arm not in ("c0", "s1"): return None
        S = {"nom21": "NOM", "bound21": "B21", "bound22": "B22"}[tg.split("/")[0]]; return ("h23", f"{arm}_s{sd}_{S}_{c['idx']:03d}")
    return None
n_rep = 0; div = 0; rep_contact = []
for c in MAIN:
    rn = replay_name(c); c["replay"] = None
    if rn and rn in RK:
        c["replay"] = RK[rn]; n_rep += 1; z = np.load(ROOT / RK[rn]); e = np.abs(z["box_pos"][:len(c["_b"])] - c["_b"][:len(z["box_pos"])]).max() if c["_b"] is not None else float("nan"); c["replay_err"] = float(e); div += int(e > 0.005) if e == e else 0
        c["_rp"] = {k: z[k] for k in ("finger_force", "finger_contacts", "hand_force", "table_force", "tray_force", "finger_contact_a", "box_tilt", "gripper_cmd")}
res["replay_coverage"] = dict(rollouts_with_replay=n_rep, of_main=len(MAIN), act_rollouts_with_replay=sum(1 for c in MAIN if c["replay"] and c["agent"] == "act"), diverging_gt_5mm=div, frac_diverging=div / max(1, n_rep),
                              replay_files_without_matching_rollout=len(RK) - n_rep, note="the remaining replay files are scripted-expert replays (no logged rollout)", unreplayed_by_source=dict(Counter(c["source"].split("/")[0] + ("/pert" if c["state_perturbed"] else "") for c in MAIN if not c["replay"] and c["agent"] == "act")))
dup_replays = Counter(c["replay"] for c in MAIN if c["replay"]); res["replay_duplicates_same_file_used_by_several_rollouts"] = {k: v for k, v in dup_replays.items() if v > 1}
# contact label coverage
rp = [c for c in MAIN if c["replay"]]; ff = np.concatenate([c["_rp"]["finger_force"] for c in rp]); hf = np.concatenate([c["_rp"]["hand_force"] for c in rp]); nf = np.concatenate([c["_rp"]["finger_contacts"] for c in rp]); ca = np.concatenate([c["_rp"]["finger_contact_a"] for c in rp])
res["contact_labels"] = dict(n_episodes=len(rp), n_steps=int(len(ff)), frac_steps_finger_contact=float((nf > 0).mean()), frac_steps_palm_contact_gt_0p01N=float((hf > 0.01).mean()), frac_steps_table_or_tray=float(np.mean(np.concatenate([(c["_rp"]["table_force"] + c["_rp"]["tray_force"]) > 0.01 for c in rp]))),
                             episodes_with_finger_contact=int(sum((c["_rp"]["finger_contacts"] > 0).any() for c in rp)), episodes_with_palm_contact=int(sum((c["_rp"]["hand_force"] > 0.01).any() for c in rp)), finger_contact_a_nan_frac=float(np.isnan(ca).mean()),
                             finger_force_N=dict(p50=float(np.percentile(ff[ff > 0], 50)), p99=float(np.percentile(ff[ff > 0], 99)), max=float(ff.max())), hand_force_N_when_in_contact=dict(p50=float(np.percentile(hf[hf > 0.01], 50)) if (hf > 0.01).any() else None, max=float(hf.max())))
# ---------------------------------------------------------------- 5. leakage audit
byk = defaultdict(list)
for c in MAIN: byk[KEY(c)].append(c)
sizes = [len(v) for v in byk.values()]
rng = np.random.default_rng(0); idx = np.arange(len(MAIN)); rng.shuffle(idx); n = len(MAIN); tr, va, te = idx[:int(.8 * n)], idx[int(.8 * n):int(.9 * n)], idx[int(.9 * n):]
trk = {KEY(MAIN[i]) for i in tr}; naive_leak = dict(val=float(np.mean([KEY(MAIN[i]) in trk for i in va])), test=float(np.mean([KEY(MAIN[i]) in trk for i in te])))
# region split on x (proposal): is any initial state in more than one region?
def region(x): return "LOW" if x < 0.38 else ("TRAIN" if x <= 0.45 else "HIGH")
span = {k: {region(c["box_xy"][0]) for c in v} for k, v in byk.items()}
# overlap with ACT training demos (poses)
demo_poses = []
for sub in ("demo_train_clipped", "demo_ext", "demo_vol"):
    for l in jl(ROOT / "outputs/datasets" / sub / "episodes.jsonl"): demo_poses.append((l["box_xy"][0], l["box_xy"][1], l["box_yaw"]))
demo_poses = np.array(demo_poses); ev_poses = np.array(sorted(byk)); dmin = np.min(np.abs(ev_poses[:, None, :] - demo_poses[None, :, :]).max(2), axis=1)
res["leakage"] = dict(unique_initial_state_groups=len(byk), group_size_hist=dict(sorted(Counter(sizes).items())), episodes_in_groups_gt1=int(sum(s for s in sizes if s > 1)), naive_random_episode_split_leak_fraction_vs_train=naive_leak,
                      groups_spanning_multiple_x_regions=int(sum(len(v) > 1 for v in span.values())), x_region_counts={r: int(sum(1 for c in MAIN if region(c["box_xy"][0]) == r)) for r in ("LOW", "TRAIN", "HIGH")},
                      act_demo_vs_eval_pose_overlap=dict(n_demo_poses=int(len(demo_poses)), n_eval_groups=int(len(ev_poses)), exact_matches_within_1e6=int((dmin < 1e-6).sum()), min_chebyshev_distance=float(dmin.min())),
                      forbidden_seeds_present=sorted({c["seed"] for c in MAIN if c["seed"] in (31, 32, 33)}), seeds_seen=sorted({c["seed"] for c in MAIN if c["seed"] is not None}),
                      rule_recommended="split by initial-state GROUP (x,y,yaw) first, then by region; hold out whole checkpoints for leave-one-checkpoint-out; never split by episode or window")
# ---------------------------------------------------------------- 7. window rules + unit tests
def windows(T, h=H, k=K, stride=STRIDE):
    """T = number of action rows (state rows = T, box/tcp/speed rows = T+1). Window at t: inputs rows t-h+1..t (state/box/contact), PAST actions t-h..t-1, future actions t..t+k-1, targets rows t+1..t+k.
    Past actions need t-h >= 0 and state targets need row t+k <= T-1, so t in [h, T-1-k]. Returns list of t."""
    return list(range(h, T - 1 - k + 1, stride))
def test_windows():
    for T in (60, 138, 200):
        ts = windows(T); assert ts and ts[0] == H and ts[-1] + K <= T - 1 and all(t - H >= 0 for t in ts); assert all(b - a == STRIDE for a, b in zip(ts, ts[1:]))
        for t in ts:
            hist = np.arange(t - H + 1, t + 1); pa = np.arange(t - H, t); act = np.arange(t, t + K); tgt = np.arange(t + 1, t + K + 1); assert hist.min() >= 0 and pa.min() >= 0 and act.max() <= T - 1 and tgt.max() <= T - 1 and (hist <= t).all() and (pa < t).all() and (tgt > t).all()
    assert windows(H + K) == [] and windows(H + K + 1) == [H]
    return "PASS"
res["window_rule"] = dict(H=H, K=K, stride=STRIDE, valid_t="t in [H, T-1-K] (T = action rows; past actions t-H..t-1 need t >= H; state targets up to row t+K <= T-1; box/contact rows have T+1 entries and could reach t+K = T but are cut at T-1 for alignment)", unit_tests=test_windows(),
                          windows_per_episode_T200=len(windows(200)), windows_per_episode_T138=len(windows(138)))
nw = 0; active = 0; moving = 0; contact_w = 0; by_split = Counter(); close_w = 0
for c in FULLM:
    T = c["T"]; ts = windows(T); nw += len(ts); sp = c["_v"]; g = c["_a"][:, 7]; split = region(c["box_xy"][0]); by_split[split] += len(ts)
    for t in ts:
        mv = np.linalg.norm(c["_b"][t + K] - c["_b"][t]) > 0.005; moving += int(mv); cl = bool(((g[t:t + K] < 0.028).any()) and not (g[t] < 0.028 and (g[t:t + K] < 0.028).all())); close_w += int(cl)
        if c.get("replay"): contact_w += int((c["_rp"]["finger_contacts"][t + 1:t + K + 1] > 0).any())
        active += int(mv or cl)
res["windows"] = dict(total_windows=nw, by_x_region=dict(by_split), windows_box_moves_gt_5mm=moving, windows_with_gripper_close_event=close_w, windows_active_any=active, frac_active=active / nw, windows_with_finger_contact_in_horizon_replayed_only=contact_w,
                      windows_from_replayed_rollouts=sum(len(windows(c["T"])) for c in FULLM if c.get("replay")))
# ---------------------------------------------------------------- 6. privileged-state table (static facts from the data pipeline)
res["privileged_state"] = {
    "state (16): joint pos + gripper": dict(privileged=False, policy_input=True, real_robot="yes (encoders)", source="ActionAdapter.state(data)"),
    "action (16) executed / raw_action": dict(privileged=False, policy_input=False, real_robot="yes (commands)", source="evaluate.py A (post clip/slew), ACTRunner.raw"),
    "tcp_pos (3)": dict(privileged="derived", policy_input=False, real_robot="derivable from joints via FK", source="data.xpos[tcp body] (sim); forward kinematics of state"),
    "box_pos (3), box_quat/yaw, box_speed": dict(privileged=True, policy_input=False, real_robot="no (needs perception)", source="data.xpos/xquat/cvel of blue_box: labelled 'eval.*', ground truth for evaluation only"),
    "box_in_tcp, tcp_in_box, box_tilt": dict(privileged=True, policy_input=False, real_robot="no", source="replay: box pose in TCP frame"),
    "contact forces/positions (finger, palm, table, tray)": dict(privileged=True, policy_input=False, real_robot="partly (gripper force/current only)", source="data.contact + mj_contactForce in replay"),
    "outcome labels pick/place/cluster": dict(privileged=True, policy_input=False, real_robot="no", source="eval.metrics.task_labels on GT box pose"),
    "images": dict(privileged=False, policy_input=True, real_robot="yes", source="not stored in rollouts; stored in the 120 expert demos")}
for c in CAT:
    for k in ("_a", "_s", "_b", "_t", "_v", "_raw", "_rp"): c.pop(k, None)
(OUT / "wm0_audit.json").write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
(OUT / "wm0_catalog.json").write_text(json.dumps([{k: v for k, v in c.items() if k not in ("keys", "shapes")} for c in CAT], default=str))
print(json.dumps({k: res[k] for k in ("counts", "exact_duplicates", "replay_coverage", "contact_labels", "leakage", "window_rule", "windows", "lengths", "state_action_lag_summary_left_arm", "raw_vs_executed", "box_speed_vs_finite_difference_corr_by_shift")}, indent=1, default=str)[:9000])
