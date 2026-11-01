#!/usr/bin/env python3
"""INDEPENDENT held-out set: raw simulator episodes (outputs/indep/raw) -> contract-v1 episode files + index (same schema as the wm2 dataset), standard windows per stratum, phase-aligned family windows and safety/integrity audit.
Everything is written under outputs/indep (the wm2 dataset is read-only here).  Windows reuse eval/wm2_common.py and eval/wm3_data.py with the dataset root redirected.  Run (ACT venv): python data/indep_build.py"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, hashlib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "data"), str(ROOT / "eval")):
    if p not in sys.path: sys.path.insert(0, p)
import wm1_sim as S, wm2_common as C, wm3_data as D
OUT = ROOT / "outputs/indep"; RAW = OUT / "raw"; DS = OUT / "dataset"; WIN = OUT / "windows"; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if DS.exists(): sys.exit(f"{DS} exists; refusing to overwrite")
    (DS / "episodes").mkdir(parents=True); WIN.mkdir(); man = json.loads((OUT / "manifest.json").read_text()); pose = {p["pid"]: p for p in man["poses"]}; raw = json.loads((RAW / "index.json").read_text())["episodes"]; IDX = []; saf = []
    from control.safety_limits import SafetyLimits
    from sim.lab_env import LabEnv
    lim = SafetyLimits.from_model(S.get_env()["env"].m)
    for r in raw:
        R = dict(np.load(RAW / "episodes" / f"{r['name']}.npz")); T = len(R["action"]); b, c = S.contact_features(R); a = R["action"][:, :8].astype(np.float32); s = R["state"][:T, :8].astype(np.float32); p = pose[r["pid"]]
        g = a[:, 7]; ci = np.nonzero(g < 0.028)[0]; tc = int(ci[0]) if len(ci) else None; tr = None
        if tc is not None:
            re_ = np.nonzero(g[tc:] > 0.035)[0]; tr = tc + int(re_[0]) if len(re_) else None
        a_close = float(R["box_in_tcp"][tc, 2] * 100) if tc is not None else None; a_rel = float(R["box_in_tcp"][tr, 2] * 100) if tr is not None else None; fin = R["box_pos"][-1, :2] - np.array([0.44, 0.30]); post = bool(r["pick_success"]) and not bool(r["task_success"])
        eid = f"x_{r['name']}"; np.savez_compressed(DS / "episodes" / f"{eid}.npz", s=s, a=a, b=b, c=c)
        ex = R["action"].astype(np.float64); rate = np.abs(np.diff(ex, axis=0)) * 20 / lim.max_speed; viol_rate = int((rate > 1 + 1e-4).sum()); viol_rng = int(((ex < lim.lo - 1e-6) | (ex > lim.hi + 1e-6)).sum()); saf.append((float(rate[:, :7].max() * 2.0), float(rate[:, 7].max() * 0.2), float(ex[:, 7].min()), float(ex[:, 7].max()), viol_rate, viol_rng, float(rate.max())))
        IDX.append(dict(eid=eid, source="indep/raw", tag=r["family"], agent="scripted_intervention", idx=r["vid"], seed=7001, box_xy=[p["x"], p["y"]], box_yaw=p["yaw"], replay_ok=bool(r["plan_ok"] and r["finite"]), h_action=r["h_action"], pick=bool(r["pick_success"]), place=bool(r["place_success"]), task_success=bool(r["task_success"]),
                        group=[round(p["x"], 6), round(p["y"], 6), round(p["yaw"], 6)], region=p["region"], split=p["split"], T=int(T), kind="interventional", family=r["family"], level=r["level"], pid=r["pid"],
                        outcomes=dict(final_xy_err_m=float(np.linalg.norm(fin)), final_z_err_m=float(r["final_box_z_err_m"]), max_rise_m=float(r["max_rise_m"]), a_close_cm=a_close, a_release_cm=a_rel, close_step=tc, release_step=tr, palm_max_force_N=float(R["hand_force"][tc:(tr or T)].max()) if tc is not None else 0.0,
                                      palm_contact=bool(tc is not None and (R["hand_force"][tc:(tr or T)] > 0.01).any()), post_pick_fail=post, clipped_range=int(r["clipped_range"]), slewed=int(r["slewed"]), plan_ok=bool(r["plan_ok"]), ik_fail_steps=int(r["ik_fail_steps"])),
                        provenance=dict(manifest_sha256=sha(OUT / "manifest.json"), source_episode=str((RAW / "episodes" / f"{r['name']}.npz").relative_to(ROOT)), variant_config=dict(family=r["family"], level=r["level"], seed=r.get("seed")))))
    for e in IDX: e["file_sha256"] = sha(DS / "episodes" / f"{e['eid']}.npz")
    (DS / "index.json").write_text(json.dumps(IDX, default=str))
    C.DS = DS; C.WIN = WIN; idx = C.load_index(); gid = C.group_ids(idx); nwin = {}
    for sp in C.SPLITS[2:]: d = C.build_split(sp, idx, gid); np.savez_compressed(WIN / f"{sp}.npz", **d); nwin[sp] = dict(windows=int(len(d["t"])), active=int(d["active"].sum()), groups=len(set(d["group"].tolist())))
    p_ = C.build_pairs(idx, gid); np.savez_compressed(WIN / "pairs.npz", **p_); fw = D.build(); np.savez_compressed(OUT / "family_windows.npz", **fw)
    dev = json.loads((ROOT / "outputs/wm2/dataset/index.json").read_text()); devg = {tuple(e["group"]) for e in dev}; newg = {tuple(e["group"]) for e in IDX}; per = {}
    for e in IDX: per.setdefault(e["pid"], set()).add(e["idx"])
    sm = np.array(saf)
    audit = dict(n_episodes=len(IDX), expected=man["n_episodes"], usable=sum(e["replay_ok"] for e in IDX), plan_failures=[e["eid"] for e in IDX if not e["outcomes"]["plan_ok"]], groups=len(newg), groups_overlapping_development=len(devg & newg),
                 groups_in_multiple_splits=sum(len({e["split"] for e in IDX if tuple(e["group"]) == g}) > 1 for g in newg), poses_with_all_25_variants=sum(len(v) == 25 for v in per.values()), forbidden_seeds_used=[e["seed"] for e in IDX if e["seed"] in (31, 32, 33)],
                 max_executed_arm_rate_rad_s=float(sm[:, 0].max()), max_executed_gripper_rate_m_s=float(sm[:, 1].max()), max_rate_over_limit_ratio=float(sm[:, 6].max()), rate_limit_violations=int(sm[:, 4].sum()), ctrlrange_violations=int(sm[:, 5].sum()), gripper_cmd_min_max=[float(sm[:, 2].min()), float(sm[:, 3].max())], ctrlrange_gripper=[float(lim.lo[7]), float(lim.hi[7])], windows=nwin, family_windows={f: int(len(fw[f"{f}/pid"])) for f in D.FAM},
                 prevalence=dict(control_success=float(np.mean([e["task_success"] for e in IDX if e["family"] == "control"])), by_family={f: dict(n=sum(1 for e in IDX if e["family"] == f), success=sum(e["task_success"] for e in IDX if e["family"] == f), palm=sum(e["outcomes"]["palm_contact"] for e in IDX if e["family"] == f)) for f in C.FAMS}),
                 index_sha256=sha(DS / "index.json"), family_windows_sha256=sha(OUT / "family_windows.npz"), windows_sha256={sp: sha(WIN / f"{sp}.npz") for sp in C.SPLITS[2:]} | {"pairs": sha(WIN / "pairs.npz")})
    (OUT / "dataset_audit.json").write_text(json.dumps(audit, indent=1, default=str)); print(json.dumps(audit, indent=1, default=str))
if __name__ == "__main__": main()
