#!/usr/bin/env python3
"""INTEGRITY GATES for the versioned datasets (v1.1, v2).  PASS/FAIL per gate; overall PASS required before any model work on the version.  Reads the frozen v1 dataset read-only.
 1 schema/units  2 no initial-state leakage  3 no episode crossing  4 replay divergence handled  5 exact preserved-source hashes  6 no change to robot/action channels  7 yaw-mask geometry  8 consistent train/inference contract
Also refits the train-only normaliser of v1.1 (wm2_models.Norm) and compares it with the frozen v1 normaliser.   Run (ACT venv): python eval/dataset_integrity.py v1_1 | v2"""
import sys, json, hashlib, resource, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT / "eval"), str(ROOT / "data")):
    if p not in sys.path: sys.path.insert(0, p)
import geometry_v2 as Gm, contract_v2 as CV, wm1_windows as W1
V1 = ROOT / "outputs/wm2/dataset"; V1W = ROOT / "outputs/wm2/windows"; SPL = ["train", "val", "test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]
sha_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); sha_arr = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(); FROZEN = json.loads((ROOT / "outputs/wm3/freeze.json").read_text())
def cpu(): a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN); return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime
def main(ver):
    t0, c0 = time.time(), cpu(); root = ROOT / ("outputs/wm2_v1_1" if ver == "v1_1" else "outputs/wm2_v2"); D = root / "dataset"; WD = root / "windows"; idx = json.loads((D / "index.json").read_text()); idx1 = json.loads((V1 / "index.json").read_text()); by1 = {e["eid"]: e for e in idx1}; by = {e["eid"]: e for e in idx}
    W = {sp: {k: v for k, v in np.load(WD / f"{sp}.npz").items()} for sp in SPL}; man = json.loads((D / "dataset_manifest.json").read_text()); G = {}; usable = [e for e in idx if e["replay_ok"] and e.get("v2_ok", True)]
    # ---- 1 schema / units
    errs = []; tmax = 0.0
    for e in usable:
        z = np.load(D / "episodes" / f"{e['eid']}.npz"); T = len(z["a"])
        for k, shp in (("s", (T, 8)), ("a", (T, 8)), ("b", (T, 7)), ("c", (T, 7))):
            if z[k].shape != shp or z[k].dtype != np.float32 or not np.isfinite(z[k]).all(): errs.append((e["eid"], k))
        tmax = max(tmax, float(z["b"][:, 5].max()))
        if ver == "v2":
            q = z["q"]; R = Gm.quat_to_R(q)
            if q.shape != (T, 4) or abs(np.linalg.norm(q, axis=1) - 1).max() > 1e-9 or np.abs(R @ R.transpose(0, 2, 1) - np.eye(3)).max() > 1e-9: errs.append((e["eid"], "q"))
    if ver == "v2":
        sch = json.loads((D / "schema.json").read_text()); n = len(W["train"]["t"]); shp_ok = W["train"]["B2_hist"].shape == (n, 4, 14) and W["train"]["Y_b2"].shape == (n, 20, 9) and W["train"]["Y_rv"].shape == (n, 20, 3) and W["train"]["Q_t"].shape == (n, 4) and W["train"]["Y_yaw_valid"].dtype == bool
        b2 = np.concatenate([W[sp]["B2_hist"].reshape(-1, 14) for sp in SPL]); hv = b2[:, 12]; unit_ok = bool(b2[:, 10].min() >= 0 and b2[:, 10].max() <= np.pi + 1e-6 and set(np.unique(hv)) <= {0.0, 1.0} and b2[:, 13].min() >= 0 and b2[:, 13].max() <= 1 + 1e-6 and np.isfinite(b2).all()); sch_ok = sch["units"]["tilt"] == "rad" and sha_file(D / "schema.json") == man["schema_sha256"]
        G["1_schema_units"] = dict(passed=bool(not errs and shp_ok and unit_ok and sch_ok and tmax <= np.pi + 1e-6), episode_errors=errs[:5], window_shapes_ok=shp_ok, b2_ranges_ok=unit_ok, schema_hash_ok=sch_ok, tilt_max_rad_in_v1_1_b=tmax)
    else: G["1_schema_units"] = dict(passed=bool(not errs and tmax <= np.pi + 1e-6), episode_errors=errs[:5], tilt_max_rad=tmax, tilt_range_ok=tmax <= np.pi + 1e-6)
    # ---- 2 no initial-state leakage
    gs = {sp: set(W[sp]["group"].tolist()) for sp in SPL}; overlap = [(a, b) for a in SPL for b in SPL if a < b and gs[a] & gs[b]]; split_same = all(by[e]["split"] == by1[e]["split"] and by[e]["group"] == by1[e]["group"] for e in by)
    g2s = {}; [g2s.setdefault(tuple(e["group"]), set()).add(e["split"]) for e in idx]; G["2_no_initial_state_leakage"] = dict(passed=bool(not overlap and split_same and all(len(v) == 1 for v in g2s.values())), window_group_overlaps=overlap, splits_and_groups_identical_to_v1=split_same, groups_in_multiple_splits=sum(len(v) > 1 for v in g2s.values()))
    # ---- 3 no episode crossing
    cross = []; bad_t = 0
    for sp in SPL:
        w = W[sp]; es = set(w["eid"].tolist()); cross += [e for e in es if by[e]["split"] != sp]
        for e in es:
            T = by[e]["T"]
            tt = w["t"][w["eid"] == e]
            if not (tt.min() >= W1.H and tt.max() <= T - 1 - W1.K and (np.diff(tt) == W1.STRIDE).all()): bad_t += 1
    in_two = len({e for sp in SPL for e in set(W[sp]["eid"].tolist())}) != sum(len(set(W[sp]["eid"].tolist())) for sp in SPL); G["3_no_episode_crossing"] = dict(passed=bool(not cross and bad_t == 0 and not in_two), episodes_in_wrong_split=len(cross), episodes_with_bad_window_range=bad_t, episode_in_two_splits=bool(in_two))
    # ---- 4 replay divergence
    if ver == "v2":
        flagged = [e for e in idx if e["replay_ok"] and not e["v2_ok"]]; in_windows = [e["eid"] for e in flagged if any(e["eid"] in set(W[sp]["eid"].tolist()) for sp in SPL)]; ok_ep = [e for e in idx if e.get("v2_ok")]
        mx = {k: max(e["replay_divergence"][k] for e in ok_ep) for k in ("pos_m", "speed_mps", "tilt_rad", "yaw_rad_valid_steps")}
        G["4_replay_divergence_handled"] = dict(passed=bool(not in_windows and len(ok_ep) + len(flagged) == sum(1 for e in idx if e["replay_ok"])), n_replayed=len(ok_ep) + len(flagged), n_flagged=len(flagged), flagged_in_windows=in_windows, flagged=[(e["eid"], e["replay_divergence"]["flagged"]) for e in flagged][:10], max_divergence_over_accepted=mx, thresholds=man["divergence_thresholds"])
    else: G["4_replay_divergence_handled"] = dict(passed=True, note="v1.1 involves no replay (v1 lineage preserved; v1 replay_ok flags copied)")
    # ---- 5 exact preserved-source hashes
    miss = 0; changed_unexpected = 0; file_ok = 0
    for e in idx:
        src = V1 / "episodes" / f"{e['eid']}.npz"
        if sha_file(src) != by1[e["eid"]]["file_sha256"]: miss += 1
    for e in usable:
        z1 = np.load(V1 / "episodes" / f"{e['eid']}.npz"); z = np.load(D / "episodes" / f"{e['eid']}.npz")
        if not (all(np.array_equal(z1[k], z[k]) for k in ("s", "a", "c")) and np.array_equal(z1["b"][:, [0, 1, 2, 3, 4, 6]], z["b"][:, [0, 1, 2, 3, 4, 6]])): changed_unexpected += 1
        if ver == "v1_1" and not e["tilt_converted"] and sha_file(D / "episodes" / f"{e['eid']}.npz") == by1[e["eid"]]["file_sha256"]: file_ok += 1
    n_unconv = sum(1 for e in usable if not e.get("tilt_converted", True)) if ver == "v1_1" else None
    w1ok = all(sha_file(V1W / f"{sp}.npz") == h for sp, h in FROZEN["wm2_windows_sha256"].items() if sp != "pairs")
    G["5_exact_preserved_source_hashes"] = dict(passed=bool(miss == 0 and changed_unexpected == 0 and sha_file(V1 / "index.json") == FROZEN["wm2_dataset_index_sha256"] and w1ok and (ver != "v1_1" or file_ok == n_unconv)), v1_episode_files_matching_v1_index=len(idx) - miss, v1_episode_files_total=len(idx), episodes_with_changed_preserved_arrays=changed_unexpected,
                                              v1_index_hash_frozen=sha_file(V1 / "index.json") == FROZEN["wm2_dataset_index_sha256"], v1_windows_hashes_frozen=w1ok, unconverted_v1_1_files_byte_identical=(f"{file_ok}/{n_unconv}" if ver == "v1_1" else None))
    # ---- 6 robot / action channels
    key = {(str(w["eid"][i]), int(w["t"][i])): i for sp in SPL for w in [np.load(V1W / f"{sp}.npz")] for i in range(len(w["t"]))} if False else None
    diffs = {}
    for sp in SPL:
        o = np.load(V1W / f"{sp}.npz"); n = W[sp]; om = {(str(a), int(t)): i for i, (a, t) in enumerate(zip(o["eid"], o["t"]))}; sel = np.array([om[(str(a), int(t))] for a, t in zip(n["eid"], n["t"])])
        diffs[sp] = {k: bool(np.array_equal(o[k][sel], n[k])) for k in ("S_hist", "A_hist", "A_fut", "Y_s", "C_hist", "Y_c", "Y_e")}
    G["6_no_robot_action_channel_change"] = dict(passed=bool(changed_unexpected == 0 and all(all(v.values()) for v in diffs.values())), window_channels_bitwise_equal_to_v1=diffs)
    # ---- 7 yaw mask geometry (v2)
    if ver == "v2":
        ev = json.load(open(ROOT / "outputs/yaw_audit/yaw_events.json")); evb = {}; viol = {"singular_not_flagged": 0, "singular_n": 0, "physical_flagged": 0, "physical_n": 0, "wrap_flagged": 0, "wrap_n": 0}; recomputed_ok = True; nmasked = 0; ntot = 0
        for e in usable[:]:
            z = np.load(D / "episodes" / f"{e['eid']}.npz"); q = z["q"]; w_, x_, y_, z_ = q.T; R00 = w_ ** 2 + x_ ** 2 - y_ ** 2 - z_ ** 2; R10 = 2 * (x_ * y_ + w_ * z_); hn = np.sqrt(R00 ** 2 + R10 ** 2); v = hn >= CV.TAU_H; sv = v[:-1] & v[1:]
            _, aux = CV.targets(q, z["b"][:, :3], z["b"][:, 4], z["b"][:, 6]); recomputed_ok &= bool(np.array_equal(sv, aux["yaw_step_valid"])); nmasked += int((~sv).sum()); ntot += len(sv); evb[e["eid"]] = sv
        for x in ev:
            sv = evb.get(x["eid"])
            if sv is None: continue
            cat = "singular" if x["cat"].startswith("S") else ("physical" if x["cat"].startswith("P") else "wrap")
            viol[cat + "_n"] += 1
            if cat == "singular" and sv[x["t"]]: viol["singular_not_flagged"] += 1
            if cat == "physical" and not sv[x["t"]]: viol["physical_flagged"] += 1
            if cat == "wrap" and not sv[x["t"]]: viol["wrap_flagged"] += 1
        # window-level mask equals the independent recomputation
        wm = True
        for sp in ("train", "val"):
            w = W[sp]; es = {e: np.load(D / "episodes" / f"{e}.npz")["q"] for e in set(w["eid"][:2000].tolist())}
            for i in range(min(2000, len(w["t"]))):
                q = es[str(w["eid"][i])]; t = int(w["t"][i]); hn = Gm.hnorm(q[t:t + 21]); sv = (hn[:-1] >= CV.TAU_H) & (hn[1:] >= CV.TAU_H)
                if not np.array_equal(sv, w["Y_yaw_valid"][i]): wm = False; break
        G["7_yaw_mask_geometry"] = dict(passed=bool(recomputed_ok and viol["singular_not_flagged"] == 0 and viol["wrap_flagged"] == 0 and wm), independent_recompute_equal=recomputed_ok, window_masks_equal_recompute_first2000_train_val=wm, event_check=viol, masked_steps=nmasked, steps=ntot, frac_masked=nmasked / max(ntot, 1), note="physical rotations flagged invalid (hnorm<0.2) are expected in a minority (see data-contract report) and are still supervised by the rotation-vector target")
    else: G["7_yaw_mask_geometry"] = dict(passed=True, note="v1.1 has no mask (not applicable)")
    # ---- 8 consistent train / inference contract (v2: rebuild windows from TRUNCATED episodes: rows <= t only)
    if ver == "v2":
        rng = np.random.default_rng(0); w = W["train"]; ii = rng.choice(len(w["t"]), 300, replace=False); ok = True; leak = []
        for i in ii:
            e = str(w["eid"][i]); t = int(w["t"][i]); z = np.load(D / "episodes" / f"{e}.npz"); q, b, s, c, a = z["q"][:t + 1], z["b"][:t + 1], z["s"][:t + 1], z["c"][:t + 1], z["a"][:t + CV.K]
            b2 = CV.row_features(q, b[:, :3], b[:, 4], b[:, 6]); X = dict(S_hist=s[t - 3:t + 1], B2_hist=b2[t - 3:t + 1], C_hist=c[t - 3:t + 1], A_hist=a[t - 4:t], A_fut=a[t:t + CV.K])
            for k, ref in (("S_hist", "S_hist"), ("B2_hist", "B2_hist"), ("C_hist", "C_hist"), ("A_hist", "A_hist"), ("A_fut", "A_fut")):
                if not np.array_equal(X[k], w[ref][i]): ok = False; leak.append((e, t, k))
        sch = json.loads((D / "schema.json").read_text()); G["8_consistent_train_inference_contract"] = dict(passed=bool(ok and set(sch["inputs"]) == {"S_hist", "B2_hist", "C_hist", "A_hist", "A_fut"} and not (set(sch["inputs"]) & set(sch["target_time_only"]))), windows_checked=300, inputs_recomputed_from_rows_le_t_equal_stored=ok, mismatches=leak[:5], schema_inputs=sch["inputs"], target_time_only=sch["target_time_only"],
                                                              note="inputs are rebuilt from episode rows <= t and the candidate action sequence only (the same CV.row_features used in training); masks and rotation-vector targets are target-time only")
    else: G["8_consistent_train_inference_contract"] = dict(passed=True, note="v1.1 keeps the v1 window contract unchanged (window inputs bitwise equal to v1 outside the tilt channel; see gate 6 and manifest)")
    # ---- normaliser (v1.1)
    extra = {}
    if ver == "v1_1":
        import wm2_models as M
        tr = {k: v for k, v in W["train"].items()}; n_new = M.Norm(tr); o = {k: v for k, v in np.load(V1W / "train.npz").items()}; n_old = M.Norm(o); same = {}
        for name in ("S", "B", "C", "A", "Ys", "Yb", "Yc", "pose"):
            mo, so = getattr(n_old, name); mn, sn = getattr(n_new, name); eqm = np.isclose(mo, mn, rtol=0, atol=0); eqs = np.isclose(so, sn, rtol=0, atol=0); same[name] = dict(mean_equal=eqm.tolist(), std_equal=eqs.tolist())
        diff_ch = {n: [i for i in range(len(v["mean_equal"])) if not (v["mean_equal"][i] and v["std_equal"][i])] for n, v in same.items()}
        ns = {n: dict(mean=getattr(n_new, n)[0].tolist(), std=getattr(n_new, n)[1].tolist()) for n in ("S", "B", "C", "A", "Ys", "Yb", "Yc", "pose")}; ns["source"] = "wm2_models.Norm fitted on the v1.1 TRAIN windows (train-only)"; (D / "norm_stats.json").write_text(json.dumps(ns, indent=1))
        extra = dict(norm_stats_sha256=sha_file(D / "norm_stats.json"), channels_whose_statistics_changed_vs_v1={n: v for n, v in diff_ch.items() if v}, tilt_B_std_v1_vs_v1_1=[float(n_old.B[1][5]), float(n_new.B[1][5])], tilt_Yb_std_v1_vs_v1_1=[float(n_old.Yb[1][5]), float(n_new.Yb[1][5])])
        G["6_no_robot_action_channel_change"]["normaliser_other_channels_unchanged"] = all(set(v) <= {5} for n, v in diff_ch.items() if n in ("B", "Yb")) and all(not v for n, v in diff_ch.items() if n not in ("B", "Yb")); G["6_no_robot_action_channel_change"]["passed"] = bool(G["6_no_robot_action_channel_change"]["passed"] and G["6_no_robot_action_channel_change"]["normaliser_other_channels_unchanged"])
    res = dict(version=ver, overall="PASS" if all(g["passed"] for g in G.values()) else "FAIL", gates=G, extra=extra, cpu_seconds=cpu() - c0, wall_seconds=time.time() - t0, dataset_index_sha256=sha_file(D / "index.json"), manifest_sha256=sha_file(D / "dataset_manifest.json")); (root / "integrity.json").write_text(json.dumps(res, indent=1, default=str))
    print(res["overall"], {k: v["passed"] for k, v in G.items()}); print(json.dumps(extra, default=str)[:600])
if __name__ == "__main__": main(sys.argv[1])
