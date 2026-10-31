#!/usr/bin/env python3
"""WM-1 dataset builder (contract v1; docs/ai-native/tasks/2026-10-31-wm0-data-audit-and-wm1-plan.md).  Analysis/data only: no learned model, no policy change, nothing outside outputs/wm1 is written.
Sources: the existing closed-loop ACT rollouts (state/action logged live), replays that carry the privileged box/contact blocks (existing replays are REFERENCED, never modified; rollouts without replay are replayed
into outputs/wm1/replay), and scripted-expert episodes (replayed with joint-state capture).  Split is by initial-state GROUP (x,y,yaw) with an x-region rule (below), never by episode/window.
Run (ACT venv):  python data/wm1_build_dataset.py [--workers 4]"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, hashlib, argparse, time, numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Pool
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "data")):
    if p not in sys.path: sys.path.insert(0, p)
import wm1_sim as S, wm1_windows as W
OUT = ROOT / "outputs/wm1"; DS = OUT / "dataset"; REP = OUT / "replay"
def sha_file(p): h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def region_of(x): return "LOW_B_strict" if x < 0.36 else ("LOW_A" if x < 0.38 else ("TRAIN" if x <= 0.45 else "HIGH"))
SPLIT_RULE = {"TRAIN": (0.8, 0.1), "LOW_A": (0.5, 0.1)}      # (train, val) fractions of GROUPS; remainder = test (test_id / test_low_heldout); LOW_B_strict, HIGH = test only (never used for tuning)

# ----------------------------------------------------------------------------- replays that are missing (new files only)
def _replay_job(j):
    f = REP / f"{j['name']}.npz"
    if f.exists(): return j["name"], "cached"
    if j["kind"] == "rollout":
        with np.load(ROOT / j["roll"]) as z: A = z["action"]
    else:
        E = S.get_env(); from policies.baseline import ScriptedExpert
        if "expert" not in S._ENV: S._ENV["expert"] = ScriptedExpert(E["env"].m, control_hz=20)
        plan = S._ENV["expert"].plan(tuple(j["box_xy"]), j["box_yaw"]); A = np.asarray(plan["ctrl"], np.float32)
    R = S.simulate(A, j["box_xy"], j["box_yaw"]); REP.mkdir(parents=True, exist_ok=True); np.savez_compressed(f, **R); return j["name"], "new"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args(); t0 = time.time()
    cat = json.loads((OUT.parent / "wm0/wm0_catalog.json").read_text()); main_ = [c for c in cat if not c["source"].startswith("debug")]
    # source-file fingerprints (to prove the originals are untouched)
    fp_before = {c["path"]: (Path(ROOT / c["path"]).stat().st_size, Path(ROOT / c["path"]).stat().st_mtime_ns) for c in main_}
    rep_roots = [ROOT / "outputs/h23/replay", ROOT / "outputs/xsweep/replay", ROOT / "outputs/cov/replay"]; fp_rep = {str(f): (f.stat().st_size, f.stat().st_mtime_ns) for r_ in rep_roots for f in r_.glob("*.npz")}
    # --- duplicates (identical executed-action hash): keep the first path, flag the rest
    seen = {}; dups = []; keep = []
    for c in sorted(main_, key=lambda c: c["path"]):
        if c["h_action"] in seen: dups.append(dict(path=c["path"], duplicate_of=seen[c["h_action"]])); continue
        seen[c["h_action"]] = c["path"]; keep.append(c)
    # --- jobs: rollouts without replay, scripted expert replays (h23 + xsweep expert poses)
    jobs = []
    for k, c in enumerate(keep):
        c["eid"] = f"r{k:05d}"
        if not c.get("replay"): jobs.append(dict(kind="rollout", name=f"roll_{c['eid']}", roll=c["path"], box_xy=c["box_xy"], box_yaw=c["box_yaw"]))
    expert_eps = []
    h23 = json.loads((ROOT / "outputs/h23/replay_meta.json").read_text())["jobs"]; xs = json.loads((ROOT / "outputs/xsweep/replay_meta.json").read_text())["jobs"]
    for src, J in (("h23", h23), ("xsweep", xs)):
        for j in J:
            if j["agent"] != "expert": continue
            e = dict(kind="expert", name=f"expert_{src}_{j['S']}_{j['idx']:03d}", box_xy=j["box_xy"], box_yaw=j["box_yaw"], source=f"expert_replay/{src}", tag=j["S"], idx=j["idx"], seed=j.get("seed")); expert_eps.append(e); jobs.append(e)
    print(f"{len(keep)} unique rollouts ({len(dups)} duplicates flagged), {sum(j['kind']=='rollout' for j in jobs)} to replay, {len(expert_eps)} expert replays", flush=True)
    with Pool(a.workers) as pool: stat = dict(pool.map(_replay_job, jobs, chunksize=4))
    # --- episodes
    EP = []; excluded = []
    for c in keep:
        T = c["T"]
        with np.load(ROOT / c["path"]) as z: st, ac, bp = z["state"], z["action"], z["box_pos"]
        if c.get("replay"): rp = np.load(ROOT / c["replay"]); rep_src = c["replay"]
        else: rp = np.load(REP / f"roll_{c['eid']}.npz"); rep_src = str((REP / f"roll_{c['eid']}.npz").relative_to(ROOT))
        err = float(np.abs(rp["box_pos"][:len(bp)] - bp[:len(rp["box_pos"])]).max()); R = {k: rp[k] for k in rp.files}
        # b, c blocks from the replay arrays (same field names in both replay formats)
        R["action"] = ac; b, cc = S.contact_features(dict(R, action=ac)); s = st[:, :8].astype(np.float32); a = ac[:, :8].astype(np.float32)
        if len(b) != len(s) or len(cc) != len(s): excluded.append(dict(path=c["path"], reason="length mismatch")); continue
        ok = err <= 0.005
        EP.append(dict(eid=c["eid"], source=c["source"], tag=c["tag"], agent=c["agent"], idx=c["idx"], seed=c["seed"], box_xy=c["box_xy"], box_yaw=c["box_yaw"], rollout=c["path"], replay=rep_src, replay_box_err_m=err, replay_ok=ok,
                       state_perturbed=c["state_perturbed"], visual_perturbed=c["visual_perturbed"], h_action=c["h_action"], pick=c["pick"], place=c["place"], task_success=c["task_success"],
                       final_xy_err_m=float(np.linalg.norm(rp["box_pos"][-1, :2] - np.array([0.44, 0.30]))), final_box=rp["box_pos"][-1].tolist(), s=s, a=a, b=b, c=cc, bp_final=rp["box_pos"][-1], lineage="rollout+replay"))
    for j in expert_eps:
        R = dict(np.load(REP / f"{j['name']}.npz")); T = len(R["action"]); b, cc = S.contact_features(R)
        EP.append(dict(eid=f"x_{j['name']}", source=j["source"], tag=j["tag"], agent="expert", idx=j["idx"], seed=j["seed"], box_xy=j["box_xy"], box_yaw=j["box_yaw"], rollout=None, replay=str((REP / f"{j['name']}.npz").relative_to(ROOT)), replay_box_err_m=None,
                       replay_ok=bool(np.isfinite(R["state"]).all()), state_perturbed=False, visual_perturbed=False, h_action=None, pick=None, place=None, task_success=None, final_xy_err_m=float(np.linalg.norm(R["box_pos"][-1, :2] - np.array([0.44, 0.30]))),
                       final_box=R["box_pos"][-1].tolist(), s=R["state"][:T, :8].astype(np.float32), a=R["action"][:, :8].astype(np.float32), b=b, c=cc, bp_final=R["box_pos"][-1], lineage="scripted expert replay (state captured)"))
    # --- groups and split (rank based inside each region -> exact proportions)
    key = lambda e: (round(e["box_xy"][0], 6), round(e["box_xy"][1], 6), round(e["box_yaw"], 6))
    groups = defaultdict(list)
    for e in EP: groups[key(e)].append(e)
    byreg = defaultdict(list)
    for g in groups: byreg[region_of(g[0])].append(g)
    split_of = {}
    for reg, gs in byreg.items():
        gs = sorted(gs, key=lambda g: hashlib.sha1(f"{g}|wm1".encode()).hexdigest()); n = len(gs)
        if reg in SPLIT_RULE:
            ntr = max(1, int(round(SPLIT_RULE[reg][0] * n))); nva = max(1, int(round(SPLIT_RULE[reg][1] * n))); te = "test_id" if reg == "TRAIN" else "test_low_heldout"
            for i, g in enumerate(gs): split_of[g] = "train" if i < ntr else ("val" if i < ntr + nva else te)
        else:
            for g in gs: split_of[g] = "test_ood_low_strict" if reg == "LOW_B_strict" else "test_ood_high"
    for e in EP:
        g = key(e); e["group"] = list(g); e["region"] = region_of(g[0]); e["split"] = split_of[g]
    # --- write episodes + index
    (DS / "episodes").mkdir(parents=True, exist_ok=True)
    for e in EP: np.savez_compressed(DS / "episodes" / f"{e['eid']}.npz", s=e["s"], a=e["a"], b=e["b"], c=e["c"])
    meta_keys = [k for k in EP[0] if k not in ("s", "a", "b", "c", "bp_final")]; index = [{k: e.get(k) for k in meta_keys} | dict(T=int(len(e["a"]))) for e in EP]
    (DS / "index.json").write_text(json.dumps(index, default=str)); (DS / "excluded_duplicates.json").write_text(json.dumps(dups, indent=1))
    # --- train-only statistics (rows of train episodes with replay_ok)
    tr = [e for e in EP if e["split"] == "train" and e["replay_ok"]]; st_ = {k: dict(mean=np.concatenate([e[k] for e in tr]).mean(0).tolist(), std=np.concatenate([e[k] for e in tr]).std(0).tolist()) for k in ("s", "a", "b", "c")}
    (DS / "stats_train.json").write_text(json.dumps(dict(source="train-split rows only", n_episodes=len(tr), n_rows=int(sum(len(e["a"]) for e in tr)), stats=st_), indent=1))
    # --- dataset statistics, windows, labels
    S_ = {}; usable = [e for e in EP if e["replay_ok"]]
    for sp in sorted({e["split"] for e in EP}):
        es = [e for e in usable if e["split"] == sp]; nw = 0; lab = np.zeros(4); act = 0
        for e in es:
            for t in W.window_starts(len(e["a"])):
                _, Y = W.get_window(e, t); lab += Y["Y_e"]; nw += 1
        close_pos = 0; palm_ep = 0; cl_ep = 0
        for e in es:
            g = e["a"][:, 7]; ci = np.nonzero(g < 0.028)[0]
            if len(ci):
                re_ = np.nonzero(g[ci[0]:] > 0.035)[0]; re_ = ci[0] + int(re_[0]) if len(re_) else len(g); palm_ep += int((e["c"][ci[0]:re_, 4] > 0.01).any())
            pp = e["pick"] and not e["task_success"] if e["pick"] is not None else False; fin = np.array(e["final_box"][:2]) - np.array([0.44, 0.30]); cl_ep += int(bool(pp) and fin[0] < -0.025 and abs(fin[1]) < 0.01)
        S_[sp] = dict(episodes=len(es), groups=len({tuple(e["group"]) for e in es}), windows=nw, window_label_prevalence=dict(zip(["close_cmd_in_horizon", "palm_contact_in_horizon", "lifted_gt3cm_at_horizon_end", "placed_at_horizon_end"], (lab / max(nw, 1)).round(4).tolist())),
                      window_positives=dict(zip(["close", "palm", "lifted", "placed"], lab.astype(int).tolist())), episodes_with_palm_contact=palm_ep, cluster_failures=cl_ep, by_source=dict(Counter(e["source"].split("/")[0] for e in es)),
                      agents=dict(Counter(e["agent"] for e in es)), state_perturbed=sum(e["state_perturbed"] for e in es))
    # --- integrity
    fp_after = {p: (Path(ROOT / p).stat().st_size, Path(ROOT / p).stat().st_mtime_ns) for p in fp_before}; fpr_after = {str(f): (f.stat().st_size, f.stat().st_mtime_ns) for r_ in rep_roots for f in r_.glob("*.npz")}
    leak = defaultdict(set)
    for e in EP: leak[tuple(e["group"])].add(e["split"])
    seeds = sorted({e["seed"] for e in EP if e["seed"] is not None}); I = dict(
        originals_unmodified=bool(fp_before == fp_after and fp_rep == fpr_after), n_source_rollouts_checked=len(fp_before), n_existing_replays_checked=len(fp_rep),
        groups=len(groups), groups_in_multiple_splits=int(sum(len(v) > 1 for v in leak.values())), episodes_by_split=dict(Counter(e["split"] for e in EP)), groups_by_split=dict(Counter(split_of[g] for g in groups)),
        region_split_rule=dict(rule="rank-based within region by sha1(group|wm1)", TRAIN="80% train / 10% val / 10% test_id of GROUPS", LOW_A="x in [0.36,0.38): 50% train / 10% val / 40% test_low_heldout", LOW_B_strict="x<0.36: test only (never tuning)", HIGH="x>0.45: test only"),
        forbidden_seeds_31_33=[s for s in seeds if s in (31, 32, 33)], seeds=seeds, duplicates_flagged=len(dups), excluded_other=excluded, diverged_replays=[e["eid"] for e in EP if not e["replay_ok"]], n_diverged=sum(not e["replay_ok"] for e in EP),
        lineage_complete=bool(all(e["replay"] and (e["rollout"] or e["agent"] == "expert") for e in EP)), privileged_fields=dict(privileged=["b (box x,y,z,yaw,speed,tilt,a_offset)", "c (finger/palm/table/tray forces, contact position)", "episode labels pick/place/cluster"], robot_observable=["s (left joints, gripper)", "a (executed targets)"],
                                                                                                                      note="b and c come from simulator ground truth; s and a are proprioception/commands"), wall_s=time.time() - t0)
    (DS / "integrity.json").write_text(json.dumps(I, indent=1, default=str)); (DS / "dataset_stats.json").write_text(json.dumps(dict(n_episodes=len(EP), usable_replay_ok=len(usable), by_split=S_, replay_status=dict(Counter(stat.values()))), indent=1, default=str))
    print(json.dumps(I, indent=1, default=str)[:2500]); print(json.dumps(S_, indent=1, default=str)[:3500])

if __name__ == "__main__": main()
