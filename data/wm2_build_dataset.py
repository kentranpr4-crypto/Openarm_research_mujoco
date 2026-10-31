#!/usr/bin/env python3
"""WM-2 dataset freeze: observational WM-1 dataset (2 300 episodes) + intervention episodes (pilot 250 + full 2 250 = 2 500) -> outputs/wm2/dataset (new dir; nothing existing is modified).
Intervention episodes come straight from the simulator with the executed (clipped/slewed) actions; their privileged blocks (b, c) use the same contract-v1 features.  Records continuous outcomes, contact labels,
slip, grasp-relative offsets and provenance.  Splits: observational episodes keep their WM-1 group split; intervention episodes take the split of their POSE (manifest) -> all 25 variants of a pose share the split.
Intervention episodes with plan_ok == False (IK failure) are kept in the index with replay_ok=False (excluded from windows) like diverged replays.
Run (ACT venv): python data/wm2_build_dataset.py"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, hashlib, subprocess, shutil, numpy as np
from pathlib import Path
from collections import Counter, defaultdict
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "data")):
    if p not in sys.path: sys.path.insert(0, p)
import wm1_sim as S, wm1_windows as W
W1 = ROOT / "outputs/wm1"; OUT = ROOT / "outputs/wm2/dataset"
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    if OUT.exists(): sys.exit(f"{OUT} exists; refusing to overwrite (move it aside to rebuild)")
    (OUT / "episodes").mkdir(parents=True); idx1 = json.loads((W1 / "dataset/index.json").read_text()); man = json.loads((W1 / "intervention_full/manifest.json").read_text()); pose = {p["pid"]: p for p in man["poses"]}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(); IDX = []
    for e in idx1:                                                                       # observational: copy files unchanged, keep metadata
        shutil.copy(W1 / "dataset/episodes" / f"{e['eid']}.npz", OUT / "episodes" / f"{e['eid']}.npz")
        IDX.append(dict(e, kind="observational", family=None, level=None, pid=None, provenance=dict(rollout=e["rollout"], replay=e["replay"], wm1_dataset="outputs/wm1/dataset")))
    n_obs = len(IDX); inter = []
    for d_, label in ((W1 / "pilot", "pilot"), (W1 / "intervention_full", "full")):
        for r in json.loads((d_ / ("pilot_index.json" if label == "pilot" else "index.json")).read_text())["episodes"]: inter.append((d_, label, r))
    for d_, label, r in inter:
        R = dict(np.load(d_ / "episodes" / f"{r['name']}.npz")); T = len(R["action"]); b, c = S.contact_features(R); a = R["action"][:, :8].astype(np.float32); s = R["state"][:T, :8].astype(np.float32); p = pose[r["pid"]]
        g = a[:, 7]; ci = np.nonzero(g < 0.028)[0]; tc = int(ci[0]) if len(ci) else None; tr = None
        if tc is not None:
            re_ = np.nonzero(g[tc:] > 0.035)[0]; tr = tc + int(re_[0]) if len(re_) else None
        a_close = float(R["box_in_tcp"][tc, 2] * 100) if tc is not None else None; a_rel = float(R["box_in_tcp"][tr, 2] * 100) if tr is not None else None
        fin = R["box_pos"][-1, :2] - np.array([0.44, 0.30]); post = bool(r["pick_success"]) and not bool(r["task_success"])
        eid = f"i_{r['name']}"; np.savez_compressed(OUT / "episodes" / f"{eid}.npz", s=s, a=a, b=b, c=c)
        IDX.append(dict(eid=eid, source=f"intervention/{label}", tag=r["family"], agent="scripted_intervention", idx=r["vid"], seed=60, box_xy=[p["x"], p["y"]], box_yaw=p["yaw"], rollout=None, replay=None, replay_box_err_m=None, replay_ok=bool(r["plan_ok"] and r["finite"]),
                        state_perturbed=False, visual_perturbed=False, h_action=r["h_action"], pick=bool(r["pick_success"]), place=bool(r["place_success"]), task_success=bool(r["task_success"]), final_xy_err_m=float(np.linalg.norm(fin)), final_box=R["box_pos"][-1].tolist(),
                        lineage="scripted intervention (simulator, executed actions)", group=[round(p["x"], 6), round(p["y"], 6), round(p["yaw"], 6)], region=p["region"], split=p["split"], T=int(T), kind="interventional", family=r["family"], level=r["level"], pid=r["pid"],
                        outcomes=dict(final_xy_err_m=float(np.linalg.norm(fin)), final_z_err_m=float(r["final_box_z_err_m"]), max_rise_m=float(r["max_rise_m"]), a_close_cm=a_close, a_release_cm=a_rel, slip_cm=(a_rel - a_close) if (a_close is not None and a_rel is not None) else None,
                                      close_step=tc, release_step=tr, palm_max_force_N=float(R["hand_force"][tc:(tr or T)].max()) if tc is not None else 0.0, palm_contact=bool(tc is not None and (R["hand_force"][tc:(tr or T)] > 0.01).any()), finger_force_hold_N=float(R["finger_force"][min(tc + 14, T)]) if tc is not None else 0.0,
                                      post_pick_fail=post, cluster=bool(post and fin[0] < -0.025 and abs(fin[1]) < 0.01), clipped_range=int(r["clipped_range"]), slewed=int(r["slewed"]), plan_ok=bool(r["plan_ok"]), ik_fail_steps=int(r["ik_fail_steps"])),
                        provenance=dict(manifest_sha256=sha256(W1 / "intervention_full/manifest.json"), source_episode=str((d_ / "episodes" / f"{r['name']}.npz").relative_to(ROOT)), code_commit=commit, variant_config=dict(family=r["family"], level=r["level"], seed=r.get("seed")))))
    for e in IDX: e["file_sha256"] = sha256(OUT / "episodes" / f"{e['eid']}.npz")
    (OUT / "index.json").write_text(json.dumps(IDX, default=str))
    # statistics on usable (replay_ok) episodes
    use = [e for e in IDX if e["replay_ok"]]; stats = dict(n_episodes=len(IDX), observational=n_obs, interventional=len(IDX) - n_obs, usable=len(use), unusable=len(IDX) - len(use), by_kind_split={f"{k}/{sp}": sum(1 for e in IDX if e["kind"] == k and e["split"] == sp) for k in ("observational", "interventional") for sp in sorted({e["split"] for e in IDX})},
                                                 groups_by_split={sp: len({tuple(e["group"]) for e in IDX if e["split"] == sp}) for sp in sorted({e["split"] for e in IDX})})
    # integrity
    g2s = defaultdict(set); [g2s[tuple(e["group"])].add(e["split"]) for e in IDX]; pose_split = defaultdict(set); [pose_split[e["pid"]].add(e["split"]) for e in IDX if e["pid"] is not None]
    hs = [e["h_action"] for e in IDX if e["h_action"]]; dupe = len(hs) - len(set(hs)); old_groups = {tuple(e["group"]) for e in idx1}; new_groups = {tuple(e["group"]) for e in IDX if e["kind"] == "interventional"}
    inter_ep = [e for e in IDX if e["kind"] == "interventional"]; per_pose = Counter(e["pid"] for e in inter_ep)
    I = dict(groups=len(g2s), groups_in_multiple_splits=sum(len(v) > 1 for v in g2s.values()), poses=len(pose_split), poses_with_multiple_splits=sum(len(v) > 1 for v in pose_split.values()), interventional_trajectories=len(inter_ep), variants_per_pose_all_25=all(v == 25 for v in per_pose.values()) and len(per_pose) == 100,
             interventional_groups_overlapping_observational_groups=len(old_groups & new_groups), duplicate_action_hashes=dupe, forbidden_seeds=[e["seed"] for e in IDX if e["seed"] in (31, 32, 33)], plan_failures=[e["eid"] for e in inter_ep if not e["outcomes"]["plan_ok"]],
             non_finite=[e["eid"] for e in inter_ep if not e["replay_ok"] and e["outcomes"]["plan_ok"]], originals_untouched_note="wm1 dataset/pilot dirs are read only (copied/derived)", dataset_index_sha256=sha256(OUT / "index.json"))
    # prevalence (interventional)
    prev = {}
    for fam in ("control", "depth", "timing", "noise", "speed", "release_height"):
        ee = [e for e in inter_ep if e["family"] == fam]; prev[fam] = dict(n=len(ee), success=sum(e["task_success"] for e in ee), pick=sum(e["pick"] for e in ee), post_pick_fail=sum(e["outcomes"]["post_pick_fail"] for e in ee), palm=sum(e["outcomes"]["palm_contact"] for e in ee), cluster=sum(e["outcomes"]["cluster"] for e in ee), plan_fail=sum(not e["outcomes"]["plan_ok"] for e in ee))
    byreg = {r: dict(n=sum(1 for e in inter_ep if e["region"] == r), success=sum(e["task_success"] for e in inter_ep if e["region"] == r), palm=sum(e["outcomes"]["palm_contact"] for e in inter_ep if e["region"] == r)) for r in ("TRAIN", "LOW_A", "LOW_B_strict", "HIGH")}
    # determinism spot check (60 random interventional episodes re-simulated into memory)
    import wm1_intervene as WI
    rng = np.random.default_rng(0); samp = rng.choice(len(inter_ep), 60, replace=False); same = 0; tmp = ROOT / "outputs/wm2/_determinism_tmp"; WI.OUT = tmp; poses_ = {p["pid"]: p for p in man["poses"]}; vars_ = {v["vid"]: v for v in man["variants"]}
    for i in samp:
        e = inter_ep[i]; j = dict(pose=poses_[e["pid"]], variant=vars_[e["idx"]]); WI.run_one(j); n = f"p{e['pid']:02d}_v{e['idx']:02d}"; a2 = np.load(tmp / "episodes" / f"{n}.npz"); src = ROOT / e["provenance"]["source_episode"]; a1 = np.load(src)
        same += int(all(np.array_equal(a1[k], a2[k], equal_nan=True) for k in ("action", "state", "box_pos", "hand_force", "finger_force")))
    shutil.rmtree(tmp, ignore_errors=True); I["determinism_spot_check_bitwise_identical"] = [same, len(samp)]
    # train-only stats over usable train episodes
    tr = [e for e in use if e["split"] == "train"]; stt = {}
    for k in ("s", "a", "b", "c"):
        X = np.concatenate([np.load(OUT / "episodes" / f"{e['eid']}.npz")[k] for e in tr]); stt[k] = dict(mean=X.mean(0).tolist(), std=X.std(0).tolist())
    (OUT / "stats_train.json").write_text(json.dumps(dict(source="train split, usable episodes only (observational + interventional)", n_episodes=len(tr), stats=stt), indent=1))
    manifest = dict(stats=stats, integrity=I, prevalence_by_family=prev, prevalence_by_region=byreg, index_sha256=I["dataset_index_sha256"], splits={sp: sorted({e["eid"] for e in IDX if e["split"] == sp})[:3] for sp in sorted({e["split"] for e in IDX})}, code_commit=commit)
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=1, default=str)); print(json.dumps(manifest, indent=1, default=str)[:5000])

if __name__ == "__main__": main()
