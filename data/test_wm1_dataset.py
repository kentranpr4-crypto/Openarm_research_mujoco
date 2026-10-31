#!/usr/bin/env python3
"""Tests for the WM-1 dataset (outputs/wm1/dataset): window rule/alignment/no boundary crossing, group-split integrity and region rules, lineage and hashes, live-vs-stored equality, privileged flags,
replay divergence flags, NaN checks, train-only statistics, duplicates, untouched originals, forbidden seeds.  Run (ACT venv): python data/test_wm1_dataset.py ; exit 1 on failure."""
import sys, json, hashlib, numpy as np
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "data"))
import wm1_windows as W
DS = ROOT / "outputs/wm1/dataset"; idx = json.loads((DS / "index.json").read_text()); I = json.loads((DS / "integrity.json").read_text()); res = {}
def ok(name, cond, info=None): res[name] = dict(passed=bool(cond), info=info); print(("PASS " if cond else "FAIL ") + name, info if info is not None else "")
# T1 window rule on synthetic lengths
for T in (60, 138, 200):
    ts = W.window_starts(T); assert ts and ts[0] == W.H and ts[-1] + W.K <= T - 1 and all(b - a == W.STRIDE for a, b in zip(ts, ts[1:]))
ok("T1_window_rule_bounds", W.window_starts(W.H + W.K) == [] and W.window_starts(W.H + W.K + 1) == [W.H])
# T2 alignment + no crossing on real episodes (all usable)
eps = {e["eid"]: e for e in idx}; bad = 0; n = 0
for e in idx:
    if not e["replay_ok"]: continue
    z = np.load(DS / "episodes" / f"{e['eid']}.npz"); ep = {k: z[k] for k in z.files}; T = len(ep["a"])
    assert all(len(ep[k]) == T for k in "sbc") and e["T"] == T
    for t in W.window_starts(T):
        X, Y = W.get_window(ep, t); n += 1
        c1 = np.array_equal(X["A_hist"][-1], ep["a"][t - 1]) and np.array_equal(X["A_fut"][0], ep["a"][t]) and np.array_equal(X["S_hist"][-1], ep["s"][t]) and np.array_equal(X["A_hist"][0], ep["a"][t - W.H]) and np.array_equal(X["S_hist"][0], ep["s"][t - W.H + 1])
        c2 = np.allclose(Y["Y_s"][0], ep["s"][t + 1] - ep["s"][t]) and np.allclose(Y["Y_s"][-1], ep["s"][t + W.K] - ep["s"][t + W.K - 1]) and np.array_equal(Y["Y_c"][-1], ep["c"][t + W.K]) and t + W.K <= T - 1 and t - W.H >= 0
        bad += int(not (c1 and c2))
ok("T2_window_alignment_no_boundary_crossing", bad == 0, dict(windows_checked=n, bad=bad))
# T3 split integrity
g2s = defaultdict(set)
for e in idx: g2s[tuple(e["group"])].add(e["split"])
reg_ok = all((e["region"] == "TRAIN" and e["split"] in ("train", "val", "test_id")) or (e["region"] == "LOW_A" and e["split"] in ("train", "val", "test_low_heldout")) or (e["region"] == "LOW_B_strict" and e["split"] == "test_ood_low_strict") or (e["region"] == "HIGH" and e["split"] == "test_ood_high") for e in idx)
ok("T3a_group_single_split", all(len(v) == 1 for v in g2s.values()), dict(groups=len(g2s)))
ok("T3b_region_rules", reg_ok); ok("T3c_no_forbidden_seeds", not any(e["seed"] in (31, 32, 33) for e in idx), sorted({e["seed"] for e in idx if e["seed"] is not None}))
sp_groups = defaultdict(set); [sp_groups[e["split"]].add(tuple(e["group"])) for e in idx]; ok("T3d_group_sets_disjoint", all(not (sp_groups[a] & sp_groups[b]) for a in sp_groups for b in sp_groups if a < b))
ok("T3e_every_split_nonempty_and_low_split_between_train_and_heldout", all(len(v) > 0 for v in sp_groups.values()) and len([g for g in sp_groups["train"] if g[0] < 0.38]) > 0 and len(sp_groups["test_low_heldout"]) > 0, {k: len(v) for k, v in sp_groups.items()})
# T4 lineage + live equality + hashes
n_chk = 0; bad_h = 0; bad_eq = 0
for e in idx:
    assert e["replay"] and (ROOT / e["replay"]).exists()
    if e["agent"] == "expert" and e["rollout"] is None: continue
    z = np.load(ROOT / e["rollout"]); h = hashlib.sha1(np.ascontiguousarray(z["action"]).tobytes()).hexdigest()[:16]; bad_h += int(h != e["h_action"]); d = np.load(DS / "episodes" / f"{e['eid']}.npz")
    bad_eq += int(not (np.array_equal(d["a"], z["action"][:, :8]) and np.array_equal(d["s"], z["state"][:, :8]))); n_chk += 1
ok("T4_lineage_hash_and_live_equality", bad_h == 0 and bad_eq == 0, dict(checked=n_chk, bad_hash=bad_h, bad_equal=bad_eq))
hs = [e["h_action"] for e in idx if e["h_action"]]; ok("T5_no_duplicate_action_hash", len(hs) == len(set(hs)), dict(n=len(hs), flagged_duplicates=I["duplicates_flagged"]))
# T6 divergence flags + exclusion from windows
nd = sum(not e["replay_ok"] for e in idx); ok("T6_divergent_replays_flagged", nd == I["n_diverged"] and all(e["replay_box_err_m"] is None or (e["replay_box_err_m"] > 0.005) == (not e["replay_ok"]) for e in idx), dict(n_diverged=nd))
# T7 finite + shapes
nanb = 0
for e in idx[::3]:
    d = np.load(DS / "episodes" / f"{e['eid']}.npz"); nanb += int(not all(np.isfinite(d[k]).all() for k in d.files)); assert d["s"].shape[1] == 8 and d["a"].shape[1] == 8 and d["b"].shape[1] == 7 and d["c"].shape[1] == 7
ok("T7_finite_and_shapes", nanb == 0, dict(sampled=len(idx[::3])))
# T8 privileged flags documented
ok("T8_privileged_vs_observable_documented", set(I["privileged_fields"]) >= {"privileged", "robot_observable"} and any("b (" in x for x in I["privileged_fields"]["privileged"]))
# T9 train-only stats
st = json.loads((DS / "stats_train.json").read_text())["stats"]; tr = [e for e in idx if e["split"] == "train" and e["replay_ok"]]
S_ = np.concatenate([np.load(DS / "episodes" / f"{e['eid']}.npz")["s"] for e in tr]); ok("T9_train_only_stats", np.allclose(st["s"]["mean"], S_.mean(0), atol=1e-5) and np.allclose(st["s"]["std"], S_.std(0), atol=1e-5), dict(n_train_episodes=len(tr)))
ok("T10_originals_untouched", I["originals_unmodified"], dict(rollouts=I["n_source_rollouts_checked"], replays=I["n_existing_replays_checked"]))
(DS / "tests.json").write_text(json.dumps(res, indent=1)); fails = [k for k, v in res.items() if not v["passed"]]; print("ALL PASS" if not fails else f"FAILED: {fails}"); sys.exit(1 if fails else 0)
