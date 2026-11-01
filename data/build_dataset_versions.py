#!/usr/bin/env python3
"""VERSIONED dataset rebuilds (new directories only; the frozen v1 dataset, its statistics, windows, baselines and checkpoints are READ-ONLY).
  v1.1 -> outputs/wm2_v1_1 : identical to v1 except the tilt channel b[:,5] is converted from DEGREES to RADIANS for the episodes whose replay script stored degrees (h23/xsweep/cov replays); radian episodes are copied byte-for-byte.
  v2   -> outputs/wm2_v2   : v1.1 arrays (s, a, b, c unchanged) + box quaternion q (T,4) obtained by deterministic simulator replay of the SAME executed actions; windows carry contract-v2 object blocks (R6 inputs, rotation-vector targets,
         geometric yaw-validity masks, Q_t / Y_rv for Gate v3).  Episodes whose replay diverges from the v1 record are flagged and excluded from v2 windows (index keeps them).
Run (ACT venv):  python data/build_dataset_versions.py v11 | v2
Each stage records process CPU seconds (self + children) in its manifest."""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, hashlib, shutil, time, resource, subprocess, numpy as np
from pathlib import Path
from multiprocessing import Pool
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "data"), str(ROOT / "eval")):
    if p not in sys.path: sys.path.insert(0, p)
import wm1_windows as W1, wm2_common as C, geometry_v2 as Gm, contract_v2 as CV
V1 = ROOT / "outputs/wm2/dataset"; V11 = ROOT / "outputs/wm2_v1_1"; V2 = ROOT / "outputs/wm2_v2"; FROZEN = json.loads((ROOT / "outputs/wm3/freeze.json").read_text())
sha_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); sha_arr = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
KEYS = ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut", "Y_s", "Y_b", "Y_c", "Y_e")
DIV = dict(pos_m=1e-5, tilt_rad=1e-4, yaw_rad_valid_steps=1e-3, speed_mps=1e-4)         # replay-divergence thresholds, fixed before the rebuild
def cpu(): a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN); return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime
def tilt_unit(e): return "radians" if (e["kind"] == "interventional" or (e.get("replay") or "").startswith("outputs/wm1/replay")) else "degrees"
def commit(): return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
def check_v1():
    assert sha_file(V1 / "index.json") == FROZEN["wm2_dataset_index_sha256"], "v1 index changed"; return json.loads((V1 / "index.json").read_text())
def train_stats(root, idx):
    tr = [e for e in idx if e["split"] == "train" and e["replay_ok"] and e.get("v2_ok", True)]; out = {}
    for k in ("s", "a", "b", "c"):
        X = np.concatenate([np.load(root / "episodes" / f"{e['eid']}.npz")[k] for e in tr]); out[k] = dict(mean=X.mean(0).tolist(), std=X.std(0).tolist())
    return dict(source="train split, usable episodes", n_episodes=len(tr), stats=out)
def build_v11():
    t0, c0 = time.time(), cpu(); idx = check_v1()
    if V11.exists(): sys.exit(f"{V11} exists; refusing to overwrite")
    (V11 / "dataset/episodes").mkdir(parents=True); (V11 / "windows").mkdir(); out = []; n_conv = 0
    for e in idx:
        src = V1 / "episodes" / f"{e['eid']}.npz"; dst = V11 / "dataset/episodes" / f"{e['eid']}.npz"; z = np.load(src); s, a, b, c = z["s"], z["a"], z["b"], z["c"]; unit = tilt_unit(e)
        if unit == "radians": shutil.copyfile(src, dst)
        else: b2 = b.copy(); b2[:, 5] = (b[:, 5].astype(np.float64) * np.pi / 180).astype(np.float32); np.savez_compressed(dst, s=s, a=a, b=b2, c=c); n_conv += 1
        z2 = np.load(dst); assert all(np.array_equal(z[k], z2[k]) for k in ("s", "a", "c")) and np.array_equal(b[:, [0, 1, 2, 3, 4, 6]], z2["b"][:, [0, 1, 2, 3, 4, 6]])
        out.append(dict(e, dataset_version="v1.1", tilt_unit_v1=unit, tilt_converted=(unit == "degrees"), v1_file_sha256=e["file_sha256"], file_sha256=sha_file(dst), preserved_sha256=dict(s=sha_arr(s), a=sha_arr(a), c=sha_arr(c), b_without_tilt=sha_arr(b[:, [0, 1, 2, 3, 4, 6]]))))
    (V11 / "dataset/index.json").write_text(json.dumps(out, default=str)); (V11 / "dataset/stats_train.json").write_text(json.dumps(train_stats(V11 / "dataset", out), indent=1))
    C.DS = V11 / "dataset"; C.WIN = V11 / "windows"; idx2 = C.load_index(); gid = C.group_ids(idx2); wh = {}
    for sp in C.SPLITS: d = C.build_split(sp, idx2, gid); np.savez_compressed(C.WIN / f"{sp}.npz", **d); wh[sp] = sha_file(C.WIN / f"{sp}.npz")
    p = C.build_pairs(idx2, gid); np.savez_compressed(C.WIN / "pairs.npz", **p); wh["pairs"] = sha_file(C.WIN / "pairs.npz"); json.dump({str(k): v for k, v in gid.items()}, open(C.WIN / "group_ids.json", "w"))
    # window-level comparison with frozen v1 windows
    cmp = {}; by_eid = {q["eid"]: q for q in idx2}
    for sp in C.SPLITS:
        a_, b_ = np.load(ROOT / "outputs/wm2/windows" / f"{sp}.npz"), np.load(C.WIN / f"{sp}.npz"); eq = {}; conv = np.array([by_eid[e_]["tilt_converted"] for e_ in a_["eid"]])
        for k in a_.files:
            x, y = a_[k], b_[k]
            if k in ("B_hist", "Y_b"):
                m = np.ones(x.shape[-1], bool); m[5] = False; eq[k + "/non_tilt_channels_bitwise_equal"] = bool(np.array_equal(x[..., m], y[..., m])); eq[k + "/tilt_unconverted_windows_bitwise_equal"] = bool(np.array_equal(x[~conv][..., 5], y[~conv][..., 5]))
                eq[k + "/tilt_converted_windows_max_abs_diff_vs_deg2rad"] = float(np.abs(x[conv][..., 5].astype(np.float64) * np.pi / 180 - y[conv][..., 5]).max()) if conv.any() else 0.0
            elif k == "level": eq[k] = bool(np.array_equal(x, y, equal_nan=True))
            else: eq[k] = bool(np.array_equal(x, y))
        cmp[sp] = dict(same_keys=set(a_.files) == set(b_.files), n_windows=int(len(b_["t"])), n_windows_from_converted_episodes=int(conv.sum()), checks=eq)
    man = dict(version="v1.1", description="v1 with tilt converted degrees->radians for degree-source episodes; everything else bit-identical", built_from_index_sha256=FROZEN["wm2_dataset_index_sha256"], n_episodes=len(out), n_tilt_converted=n_conv, index_sha256=sha_file(V11 / "dataset/index.json"), stats_train_sha256=sha_file(V11 / "dataset/stats_train.json"),
               windows_sha256=wh, window_comparison_vs_v1=cmp, code_commit=commit(), cpu_seconds=cpu() - c0, wall_seconds=time.time() - t0)
    (V11 / "dataset/dataset_manifest.json").write_text(json.dumps(man, indent=1, default=str)); print(json.dumps({k: man[k] for k in ("n_episodes", "n_tilt_converted", "index_sha256", "cpu_seconds", "wall_seconds")}, indent=1)); print(json.dumps(cmp["train"], indent=1)[:2000])
_E = {}
def replay(job):
    eid, kind, rp, box_xy, box_yaw = job
    if "env" not in _E:
        from sim.lab_env import LabEnv
        _E["env"] = LabEnv(cameras=())
    env = _E["env"]; d = env.d; box = env.box; A = np.load(ROOT / rp)["action"]; env.set_perturbation(); env.reset(tuple(box_xy), float(box_yaw)); q, pos, spd = [], [], []
    for t in range(len(A) + 1):
        q.append(d.xquat[box].copy()); pos.append(d.xpos[box].copy()); spd.append(float(np.linalg.norm(d.cvel[box][3:])))
        if t < len(A): env.step(A[t])
    return eid, np.array(q), np.array(pos), np.array(spd)
def build_v2():
    t0, c0 = time.time(), cpu(); idx = json.loads((V11 / "dataset/index.json").read_text()); assert sha_file(V11 / "dataset/index.json") == json.loads((V11 / "dataset/dataset_manifest.json").read_text())["index_sha256"]
    if V2.exists(): sys.exit(f"{V2} exists; refusing to overwrite")
    (V2 / "dataset/episodes").mkdir(parents=True); (V2 / "windows").mkdir(); use = [e for e in idx if e["replay_ok"]]
    jobs = [(e["eid"], e["kind"], e["replay"] if e["kind"] == "observational" else e["provenance"]["source_episode"], e["box_xy"], e["box_yaw"]) for e in use]
    with Pool(4) as pool: res = {r[0]: r[1:] for r in pool.map(replay, jobs, chunksize=8)}
    t_replay = time.time() - t0; out = []; by = {}
    for e in idx:
        x = dict(e, dataset_version="v2", v2_ok=False, replay_divergence=None)
        if e["replay_ok"]:
            z = np.load(V11 / "dataset/episodes" / f"{e['eid']}.npz"); s, a, b, c = z["s"], z["a"], z["b"], z["c"]; T = len(a); q, pos, spd = res[e["eid"]]; q, pos, spd = q[:T], pos[:T], spd[:T]
            hn = Gm.hnorm(q); valid = hn >= CV.TAU_H; dyaw = np.abs(Gm.wrap(Gm.yaw_euler(q) - b[:, 3])); dv = dict(pos_m=float(np.abs(pos - b[:, :3]).max()), speed_mps=float(np.abs(spd - b[:, 4]).max()), tilt_rad=float(np.abs(Gm.tilt_rad(q) - b[:, 5]).max()), yaw_rad_valid_steps=float(dyaw[valid].max()) if valid.any() else 0.0)
            bad = {k: v for k, v in dv.items() if v > DIV[k]}; x["replay_divergence"] = dict(dv, flagged=bad); x["v2_ok"] = not bad
            if x["v2_ok"]: np.savez_compressed(V2 / "dataset/episodes" / f"{e['eid']}.npz", s=s, a=a, b=b, c=c, q=q.astype(np.float64)); x["file_sha256_v2"] = sha_file(V2 / "dataset/episodes" / f"{e['eid']}.npz"); x["preserved_sha256"] = dict(s=sha_arr(s), a=sha_arr(a), c=sha_arr(c), b=sha_arr(b))
        out.append(x); by[e["eid"]] = x
    (V2 / "dataset/index.json").write_text(json.dumps(out, default=str)); gid = C.group_ids(out)
    # ---------------- windows (same enumeration as v1: group-by-initial-state splits, stride 5, t in [H, T-1-K])
    wh = {}; stats = {}
    for sp in C.SPLITS:
        E = [e for e in out if e["split"] == sp and e["replay_ok"] and e["v2_ok"]]; cols = {k: [] for k in ("S_hist", "B_hist", "B2_hist", "C_hist", "A_hist", "A_fut", "Y_s", "Y_b", "Y_b2", "Y_c", "Y_e", "Y_yaw", "Y_yaw_valid", "Y_geo_step", "Q_t", "Y_rv", "group", "episode", "t", "active", "kind", "family", "level", "pid", "pose")}; eids = []
        for ei, e in enumerate(E):
            z = np.load(V2 / "dataset/episodes" / f"{e['eid']}.npz"); ep = dict(s=z["s"], a=z["a"], b=z["b"], c=z["c"]); q = z["q"]; T = len(ep["a"]); b2 = CV.row_features(q, ep["b"][:, :3], ep["b"][:, 4], ep["b"][:, 6]); Y, aux = CV.targets(q, ep["b"][:, :3], ep["b"][:, 4], ep["b"][:, 6])
            for t in W1.window_starts(T):
                X, Yv = W1.get_window(ep, t)
                for k in ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut"): cols[k].append(X[k])
                for k in ("Y_s", "Y_b", "Y_c", "Y_e"): cols[k].append(Yv[k])
                cols["B2_hist"].append(b2[t - W1.H + 1:t + 1]); cols["Y_b2"].append(Y[t:t + W1.K]); cols["Y_yaw"].append(aux["yaw_legacy"][t:t + W1.K]); cols["Y_yaw_valid"].append(aux["yaw_step_valid"][t:t + W1.K]); cols["Y_geo_step"].append(aux["rot_step_geodesic_sym"][t:t + W1.K]); cols["Q_t"].append(q[t]); cols["Y_rv"].append(Y[t:t + W1.K, 3:6])
                act = bool(np.linalg.norm(ep["b"][t + W1.K, :3] - ep["b"][t, :3]) > 0.005 or (ep["a"][t:t + W1.K, 7] < 0.028).any())
                cols["group"].append(gid[tuple(e["group"])]); cols["episode"].append(ei); cols["t"].append(t); cols["active"].append(act); cols["kind"].append(1 if e["kind"] == "interventional" else 0); cols["family"].append(C.FAMS[e["family"]] if e["family"] else -1); cols["level"].append(e["level"] if e["level"] is not None else np.nan)
                cols["pid"].append(e["pid"] if e["pid"] is not None else -1); cols["pose"].append([e["box_xy"][0], e["box_xy"][1], e["box_yaw"]]); eids.append(e["eid"])
        f32 = ("S_hist", "B_hist", "B2_hist", "C_hist", "A_hist", "A_fut", "Y_s", "Y_b", "Y_b2", "Y_c", "Y_e", "Y_yaw", "Y_geo_step", "Y_rv", "level", "pose"); arr = {k: np.asarray(v, np.float32 if k in f32 else (bool if k in ("active", "Y_yaw_valid") else (np.float64 if k == "Q_t" else np.int64))) for k, v in cols.items()}; arr["eid"] = np.array(eids)
        np.savez_compressed(V2 / "windows" / f"{sp}.npz", **arr); wh[sp] = sha_file(V2 / "windows" / f"{sp}.npz"); stats[sp] = int(len(arr["t"]))
    # ---------------- train-only normaliser and loss scales (explicit; contract v2)
    tr = np.load(V2 / "windows/train.npz"); f = lambda x: dict(mean=x.reshape(-1, x.shape[-1]).mean(0).tolist(), std=(x.reshape(-1, x.shape[-1]).std(0) + 1e-6).tolist())
    std_mask = [True, True, True] + [False] * 6 + [True, True, True, False, False]                         # b2 channels standardised (pos, speed, tilt, a_offset); R6, heading_valid, hnorm are bounded and left as is
    Yb2 = tr["Y_b2"].reshape(-1, 9); geo = tr["Y_geo_step"].ravel(); ok = geo <= 0.3; yw = tr["Y_yaw"].ravel(); vv = tr["Y_yaw_valid"].ravel() & (np.abs(yw) <= np.pi / 2)
    norm = dict(source="train split windows of this dataset version (train-only)", S=f(tr["S_hist"]), B2=f(tr["B2_hist"]), B2_standardise_mask=std_mask, B2_names=CV.B2_NAMES, C=f(tr["C_hist"]), A=f(np.concatenate([tr["A_hist"], tr["A_fut"]], 1)), Ys=f(tr["Y_s"]), Yb2=f(tr["Y_b2"]), Yb2_names=["dx", "dy", "dz", "rvx", "rvy", "rvz", "dspeed", "dtilt_rad", "da_offset"], Yc=f(tr["Y_c"]), pose=f(tr["pose"]))
    scales = dict(yaw_delta_std_all=float(yw.std()), yaw_delta_std_valid_nonspike=float(yw[vv].std()), rotvec_std_all=Yb2[:, 3:6].std(0).tolist(), rotvec_std_steps_with_sym_geodesic_le_0p3=Yb2[ok][:, 3:6].std(0).tolist(), n_valid_yaw_steps=int(vv.sum()), n_steps=int(len(yw)), frac_yaw_steps_masked=float(1 - tr["Y_yaw_valid"].mean()))
    (V2 / "dataset/norm_stats.json").write_text(json.dumps(norm, indent=1)); (V2 / "dataset/loss_scales.json").write_text(json.dumps(scales, indent=1))
    schema = dict(version="v2", frames="world frame z up; R = xmat (body->world); quaternion (w,x,y,z) MuJoCo xquat", units=dict(position="m", speed="m/s", tilt="rad", rotation_vector="rad (world frame, q_{k+1} conj(q_k))", yaw="rad", gripper="m"), B2_order=CV.B2_NAMES, Y_b2_order=norm["Yb2_names"], shapes=dict(B2_hist=[4, 14], Y_b2=[20, 9], Y_rv=[20, 3], Y_yaw=[20], Y_yaw_valid=[20], Q_t=[4]),
                  yaw_validity="heading_valid = hnorm >= %.2f at the CURRENT row (input); yaw_step_valid = both ends valid (target-time loss mask, never an input)" % CV.TAU_H, inputs=["S_hist", "B2_hist", "C_hist", "A_hist", "A_fut"], target_time_only=["Y_b2", "Y_rv", "Y_yaw", "Y_yaw_valid", "Y_geo_step", "Y_s", "Y_c", "Y_e"], tau_h=CV.TAU_H)
    (V2 / "dataset/schema.json").write_text(json.dumps(schema, indent=1)); (V2 / "dataset/stats_train.json").write_text(json.dumps(train_stats(V2 / "dataset", out), indent=1))
    man = dict(version="v2", built_from=dict(v1_1_index_sha256=sha_file(V11 / "dataset/index.json"), v1_index_sha256=FROZEN["wm2_dataset_index_sha256"]), n_episodes=len(out), n_replayed=len(use), n_v2_ok=sum(1 for e in out if e["v2_ok"]), n_diverged=sum(1 for e in out if e["replay_ok"] and not e["v2_ok"]), divergence_thresholds=DIV, diverged_eids=[e["eid"] for e in out if e["replay_ok"] and not e["v2_ok"]],
               windows_per_split=stats, windows_sha256=wh, index_sha256=sha_file(V2 / "dataset/index.json"), norm_stats_sha256=sha_file(V2 / "dataset/norm_stats.json"), loss_scales_sha256=sha_file(V2 / "dataset/loss_scales.json"), schema_sha256=sha_file(V2 / "dataset/schema.json"), loss_scales=scales, code_commit=commit(), replay_wall_seconds=t_replay, cpu_seconds=cpu() - c0, wall_seconds=time.time() - t0)
    (V2 / "dataset/dataset_manifest.json").write_text(json.dumps(man, indent=1, default=str)); print(json.dumps({k: man[k] for k in ("n_episodes", "n_replayed", "n_v2_ok", "n_diverged", "windows_per_split", "cpu_seconds", "wall_seconds", "loss_scales")}, indent=1))
if __name__ == "__main__": {"v11": build_v11, "v2": build_v2}[sys.argv[1]]()
