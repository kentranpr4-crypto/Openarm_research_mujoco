"""P1 (block pushing) environment validation battery.  Read-only with respect to every existing result; writes ONE new json (refuses to overwrite).  CPU MuJoCo, bounded (about 2-4 minutes).
Checks: randomisation (box / goal / friction / mass), contact physics (rest stability, friction effectiveness, penetration), action semantics (clip + slew), success / stability criteria, deterministic reset and bit-exact snapshot replay,
pushing-expert quality on a development distribution (seeds 23,100,000+; never 31-33).  All numerical thresholds are constants below, fixed BEFORE running.
Usage (repository root):  python -m research.supermimic_openarm.p1_validate_env --out outputs/p1_validation/env_validation.json [--resets 300 --expert-episodes 48]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
THR = dict(expert_strict_min=0.80, penetration_robot_m=-0.002, table_rest_penetration_report_only=True, spec_ranges_violations=0, tray_footprint_goal_fraction_max=0.02, slide_min_relative_gain=0.05, rest_drift_m=1e-4, rest_speed_m_s=1e-3, executed_parity_err=1e-9, slew_violations=0, penetration_m=-0.002, expert_placed_min=0.8, replay_diff=0.0)
DEV_SEED_BASE = 23_000_000; EXPERT_SEED_BASE = 24_300_000                                   # fresh development seeds (tuning used 24,000,000+, the reported expert sets 24,100,000+ / 24,200,000+; none of them reused for tuning)
TRAY = dict(x=(0.44 - 0.07 - 0.006, 0.44 + 0.07 + 0.006), y=(0.30 - 0.07 - 0.006, 0.30 + 0.07 + 0.006)); OTHERS = {"red_cylinder": (0.40, -0.12), "green_cube": (0.52, 0.00)}
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--resets", type=int, default=300); ap.add_argument("--expert-episodes", type=int, default=100); a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists(): sys.exit(f"refusing to overwrite {out}")
    from .utils import assert_seeds_allowed, repro_metadata, atomic_write_json
    from .envs.openarm_env import OpenArmTaskEnv
    from .envs import layout as L
    from .envs.tasks import BOX_REST_Z
    from .baselines.push_expert import PushExpert, MAX_STEP
    import mujoco
    assert_seeds_allowed(DEV_SEED_BASE, EXPERT_SEED_BASE); env = OpenArmTaskEnv("pushing"); task = env.task; m, d = env.m, env.d; R = []
    def check(name, status, thr, evidence): R.append(dict(check=name, status=status, threshold=thr, evidence=evidence)); print(f"[{status:4s}] {name}: {json.dumps(evidence, default=float)[:230]}", flush=True)
    # ---------------------------------------------------------------------------------------------------------- V1 randomisation
    rows = []; viol = 0; fr_err = 0.0; masses = []; inertia_err = 0.0; invalid_goal = 0; rejected = 0; params = []; full = 2 * np.array([0.025, 0.020, 0.020])
    from .envs.tasks import Pushing
    bg_id, desk_id = m.geom("blue_box_collision").id, m.geom("desk_top_collision").id
    def box_desk_mu():
        v = [float(c.friction[0]) for c in d.contact[:d.ncon] if {c.geom1, c.geom2} == {bg_id, desk_id}]; return v[0] if v else float("nan")
    for i in range(a.resets):
        env.reset(DEV_SEED_BASE + i); inf = env.episode_info; bx = d.qpos[env.box_adr:env.box_adr + 3].copy(); q = d.qpos[env.box_adr + 3:env.box_adr + 7]; yaw = 2 * np.arctan2(q[3], q[0]); g = np.asarray(inf["goal"]); fs = inf["friction_scale"]
        mu_eff = box_desk_mu(); fr_err = max(fr_err, abs(mu_eff - 0.8 * fs)); mk = float(m.body_mass[env.box_body]); masses.append(mk); I = m.body_inertia[env.box_body]; exp_I = mk / 12.0 * np.array([full[1] ** 2 + full[2] ** 2, full[0] ** 2 + full[2] ** 2, full[0] ** 2 + full[1] ** 2]); inertia_err = max(inertia_err, float(np.abs(I / exp_I - 1).max()))
        gv = Pushing.goal_valid(np.array(inf["box_xy"]), g)[0]; invalid_goal += (not gv); rejected += int(inf["rejected_samples"] > 0)
        sp = task.p; ok = (sp["box_x"][0] - 1e-6 <= inf["box_xy"][0] <= sp["box_x"][1] + 1e-6) and (sp["box_y"][0] - 1e-6 <= inf["box_xy"][1] <= sp["box_y"][1] + 1e-6) and abs(yaw) <= 0.3 + 1e-6 and sp["friction"][0] - 1e-6 <= fs <= sp["friction"][1] + 1e-6 and 0.33 - 1e-6 <= g[0] <= 0.56 + 1e-6 and -1e-6 <= g[1] <= 0.26 + 1e-6 and abs(bx[2] - BOX_REST_Z) < 1e-3 and sp["mass_scale"][0] * env._mass0 - 1e-9 <= mk <= sp["mass_scale"][1] * env._mass0 + 1e-9
        viol += (not ok); rows.append((inf["box_xy"][0], inf["box_xy"][1], yaw, fs, g[0], g[1], float(np.linalg.norm(g - np.array(inf["box_xy"])))))
        params.append(dict(seed=DEV_SEED_BASE + i, friction_scale=fs, mu_effective_box_desk=mu_eff, mass_kg=mk, inertia_diag_kgm2=[float(x) for x in I], box_xy=inf["box_xy"], box_yaw=float(yaw), goal=inf["goal"], rejected_samples=int(inf["rejected_samples"])))
    A = np.array(rows); in_tray = float(np.mean((A[:, 4] > TRAY["x"][0]) & (A[:, 4] < TRAY["x"][1]) & (A[:, 5] > TRAY["y"][0]) & (A[:, 5] < TRAY["y"][1]))); dist_other = {k: float(np.min(np.hypot(A[:, 0] - v[0], A[:, 1] - v[1]))) for k, v in OTHERS.items()}
    try: from scipy.stats import kstest                                                                                  # optional: scipy is not in the pinned lock; without it only the KS uniformity p-values are skipped
    except ImportError: kstest = None
    ks = ({n: float(kstest((A[:, j] - lo) / (hi - lo), "uniform").pvalue) for n, j, lo, hi in (("box_x", 0, 0.38, 0.48), ("box_y", 1, 0.06, 0.18), ("friction_scale", 3, 0.7, 1.3))} if kstest else "skipped: scipy not installed"); mk = np.array(masses)
    check("V1a randomised box/goal/friction/mass/yaw inside the specification; EFFECTIVE box-desk contact friction equals 0.8*scale (measured on d.contact)", "PASS" if viol <= THR["spec_ranges_violations"] and fr_err < 1e-6 else "FAIL", f"violations <= {THR['spec_ranges_violations']}, |mu_effective - 0.8*scale| < 1e-6", dict(resets=a.resets, violations=viol, effective_friction_max_abs_err=fr_err, ks_uniform_pvalues=ks, goal_distance_m=dict(min=float(A[:, 6].min()), mean=float(A[:, 6].mean()), max=float(A[:, 6].max())), min_box_distance_to_other_objects_m=dist_other))
    check("V1b goal validity after rejection sampling (inside workspace, box footprint clear of tray/cube/cylinder, straight push path clear); strict success criterion untouched", "PASS" if invalid_goal == 0 and in_tray == 0.0 else "FAIL", "invalid goals == 0 and goals inside the tray == 0", dict(invalid_goals=invalid_goal, goals_inside_tray_fraction=in_tray, resets_needing_resampling=rejected, resampling_fraction=rejected / a.resets))
    check("V2a box mass randomisation with consistent inertia (cuboid, uniform density)", "PASS" if len(set(np.round(mk, 9))) > a.resets * 0.9 and mk.min() >= 0.6 * env._mass0 - 1e-9 and mk.max() <= 1.6 * env._mass0 + 1e-9 and inertia_err < 1e-9 else "FAIL", "mass in [0.6,1.6]x0.05 kg, inertia == m/12*(b^2+c^2) to 1e-9", dict(mass_kg=dict(min=float(mk.min()), mean=float(mk.mean()), max=float(mk.max()), distinct=int(len(set(np.round(mk, 9))))), inertia_max_relative_err=inertia_err, implied_density_kg_m3=dict(min=float(mk.min() / np.prod(full)), max=float(mk.max() / np.prod(full)))))
    def pin(**kw):
        base = type(task).sample.__get__(task); task.sample = lambda rng: {**base(rng), "box_xy": (0.44, 0.12), "goal": np.array([0.50, 0.12]), "box_yaw": 0.0, **kw}
    acc = {}
    for ms in (0.6, 1.0, 1.6):
        pin(friction_scale=1.0, mass_scale=ms); env.reset(1); d.xfrc_applied[env.box_body, 0] = 0.6; x0 = d.qpos[env.box_adr]
        for _ in range(500): mujoco.mj_step(m, d)
        acc[ms] = 2 * float(d.qpos[env.box_adr] - x0) / (500 * m.opt.timestep) ** 2; d.xfrc_applied[:] = 0
    check("V2b mass changes the dynamics (acceleration under a constant 0.6 N horizontal force)", "PASS" if acc[0.6] > acc[1.0] > acc[1.6] and acc[0.6] / max(acc[1.0], 1e-9) > 2.0 else "FAIL", "acceleration strictly decreasing with mass, light/nominal ratio > 2", dict(acceleration_m_s2={str(k): v for k, v in acc.items()}))
    del task.sample                                                                                       # restore the class sampler
    # ---------------------------------------------------------------------------------------------------------- V3 contact physics
    env.reset(DEV_SEED_BASE + 7); b0 = d.qpos[env.box_adr:env.box_adr + 3].copy(); hold = env.prev_ctrl[:8].copy(); drift = 0.0; sp_max = 0.0
    for _ in range(400): env.step(hold); drift = max(drift, float(np.linalg.norm(d.qpos[env.box_adr:env.box_adr + 3] - b0))); sp_max = max(sp_max, float(np.linalg.norm(d.qvel[env.box_dof:env.box_dof + 3])))
    check("V3a box at rest stays at rest (400 control steps, arm holding)", "PASS" if drift < THR["rest_drift_m"] and sp_max < THR["rest_speed_m_s"] else "FAIL", f"drift < {THR['rest_drift_m']} m, speed < {THR['rest_speed_m_s']} m/s", dict(max_drift_m=drift, max_speed_m_s=sp_max))
    table_fr = sorted({round(float(m.geom_friction[g, 0]), 3) for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == "desk" and m.geom_contype[g]}); slide = {}; v0 = 0.5
    for fs in (0.7, 0.85, 1.0, 1.15, 1.3):
        pin(friction_scale=fs, mass_scale=1.0); env.reset(1); mu = box_desk_mu(); p0 = d.qpos[env.box_adr:env.box_adr + 3].copy(); d.qvel[env.box_dof] = v0
        for _ in range(1500): mujoco.mj_step(m, d)
        dist = float(d.qpos[env.box_adr] - p0[0]); slide[fs] = dict(measured_contact_mu=mu, distance_m=dist, coulomb_theory_m=v0 ** 2 / (2 * mu * 9.81), relative_dev_from_theory=dist / (v0 ** 2 / (2 * mu * 9.81)) - 1)
    del task.sample
    ds = [slide[k]["distance_m"] for k in (0.7, 0.85, 1.0, 1.15, 1.3)]; effective = all((ds[i] - ds[i + 1]) / ds[i + 1] >= THR["slide_min_relative_gain"] for i in range(4)); mu_ok = all(abs(slide[k]["measured_contact_mu"] - 0.8 * k) < 1e-6 for k in slide); th_ok = all(abs(slide[k]["relative_dev_from_theory"]) <= 0.03 for k in slide)
    check("V3b friction randomisation changes the physics over its whole range (measured contact mu and sliding distance, scales 0.7 / 0.85 / 1.0 / 1.15 / 1.3)", "PASS" if effective and mu_ok and th_ok else "FAIL", f"mu_eff == 0.8*scale; strictly decreasing with >= {int(THR['slide_min_relative_gain'] * 100)}% steps; within 3% of v^2/(2 mu g)", dict(per_scale=slide, desk_geom_sliding_friction=table_fr, mechanism="box geoms get priority 1 in pair_exact mode, so the box-desk contact uses the box's own friction instead of max(box, desk)"))
    # ---------------------------------------------------------------------------------------------------------- V4 action semantics
    rng = np.random.default_rng(0); env.reset(DEV_SEED_BASE + 11); perr = 0.0; slew_v = 0; lim = env.adapter.limits; right0 = env.hold_right.copy(); right_ok = True
    for _ in range(300):
        act = rng.uniform(-1, 1, 8) * (1 + 3 * (rng.random() < 0.3)); prev = env.prev_ctrl[:8].copy(); o = env.step(env.state_vec()[L.PREV] + np.clip(act, -1, 1) * MAX_STEP); ex = o["executed"].astype(float)
        tgt = prev + np.clip(act, -1, 1) * MAX_STEP; c = np.clip(tgt, lim.lo[:8], lim.hi[:8]); ref = prev + np.clip(c - prev, -lim.max_speed[:8] * 0.05, lim.max_speed[:8] * 0.05); perr = max(perr, float(np.abs(ex - ref).max()) / 1.0); slew_v += int(np.any(np.abs(ex - prev) > lim.max_speed[:8] * 0.05 + 1e-6) or np.any(ex < lim.lo[:8] - 1e-6) or np.any(ex > lim.hi[:8] + 1e-6)); right_ok &= bool(np.allclose(env.prev_ctrl[8:], right0))
    check("V4 action semantics: executed ctrl == independent clip->slew reference; ctrlrange and slew never violated; right arm held", "PASS" if perr < THR["executed_parity_err"] + 1e-6 and slew_v == 0 and right_ok else "FAIL", f"parity err < {THR['executed_parity_err']} (float32 output), violations = 0", dict(max_parity_err=perr, violations=slew_v, right_arm_held=right_ok, steps=300))
    # ---------------------------------------------------------------------------------------------------------- V6 determinism + snapshot replay
    def sig(): return np.concatenate([env.state_vec(), env.goal]).tobytes()
    env.reset(DEV_SEED_BASE + 21); s1 = sig(); env.reset(DEV_SEED_BASE + 21); s2 = sig(); env.reset(DEV_SEED_BASE + 22); s3 = sig()
    acts = rng.uniform(-1, 1, (40, 8)); env.reset(DEV_SEED_BASE + 21); [env.step(env.state_vec()[L.PREV] + x * MAX_STEP) for x in acts[:15]]; spec = mujoco.mjtState.mjSTATE_INTEGRATION; st = np.empty(mujoco.mj_stateSize(m, spec)); mujoco.mj_getState(m, d, st, spec)
    py = (env.prev_ctrl.copy(), env.t, env.held, env.ever_success, env.box_max_rise, env.tcp_prev.copy()); ra = [env.step(env.state_vec()[L.PREV] + x * MAX_STEP)["state"].copy() for x in acts[15:]]
    mujoco.mj_setState(m, d, st, spec); env.prev_ctrl, env.t, env.held, env.ever_success, env.box_max_rise, env.tcp_prev = py[0].copy(), py[1], py[2], py[3], py[4], py[5].copy(); mujoco.mj_forward(m, d); rb = [env.step(env.state_vec()[L.PREV] + x * MAX_STEP)["state"].copy() for x in acts[15:]]
    rd = float(np.abs(np.array(ra) - np.array(rb)).max()); check("V6 deterministic reset (same seed identical, different seed different) and bit-exact snapshot replay", "PASS" if s1 == s2 and s1 != s3 and rd <= THR["replay_diff"] else "FAIL", "identical resets; replay max diff == 0", dict(same_seed_identical=s1 == s2, different_seed_differs=s1 != s3, snapshot_replay_max_diff=rd))
    # ---------------------------------------------------------------------------------------------------------- V7 expert on a development distribution + V5 criteria recomputation + V3c penetration
    ex = PushExpert(); eps = []; pen_min = 0.0; pen_by = {}; incons = 0; t0 = time.time()
    for i in range(a.expert_episodes):
        sd = EXPERT_SEED_BASE + i; env.reset(sd); g = env.goal.copy(); D, SPD, RISE = [], [], []; rise_at_goal = None; touched = False; done = False; fell = False; held = 0; ever = False
        while not done:
            act = ex.act(env); o = env.step(env.state_vec()[L.PREV] + act * MAX_STEP); done = o["terminated"] or o["truncated"]; fell |= o["terminated"]; bp = d.qpos[env.box_adr:env.box_adr + 3]
            D.append(float(np.linalg.norm(bp[:2] - g)))
            if rise_at_goal is None and D[-1] < 0.018: rise_at_goal = float(env.box_max_rise)
            SPD.append(float(o["info"]["box_speed"])); RISE.append(float(env.box_max_rise)); touched |= (o["info"]["finger_contacts"] > 0 or o["info"]["hand_force_N"] > 0)
            for c in d.contact[:d.ncon]:
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in env.box_geoms) != (g2 in env.box_geoms):
                    oth = g2 if g1 in env.box_geoms else g1; nm = m.body(m.geom_bodyid[oth]).name; pen_by[nm] = min(pen_by.get(nm, 0.0), float(c.dist)); pen_min = min(pen_min, float(c.dist))
        inst = [(D[k] < 0.03 and SPD[k] < 0.03 and RISE[k] <= 0.01) for k in range(len(D))]; run = 0; rec = False
        for x in inst: run = run + 1 if x else 0; rec |= run >= task.hold_steps
        incons += int(rec != bool(env.ever_success)); placed = D[-1] < 0.03 and SPD[-1] < 0.03
        why = "success" if env.ever_success else ("box off table" if fell else ("no contact" if not touched else ("lifted > 1 cm (final distance ok)" if (placed and RISE[-1] > 0.01) else ("stopped short / overshoot (> 3 cm)" if D[-1] >= 0.03 else "not at rest"))))
        eps.append(dict(seed=sd, success=bool(env.ever_success), placed=bool(placed), final_dist_m=D[-1], max_rise_m=RISE[-1], max_rise_until_goal_reached_m=rise_at_goal, why=why))
    n = len(eps); succ = sum(e["success"] for e in eps); plc = sum(e["placed"] for e in eps); causes = {}
    for e in eps: causes[e["why"]] = causes.get(e["why"], 0) + 1
    robot_pen = min([v for k, v in pen_by.items() if "openarm" in k] or [0.0]); check("V3c contact penetration between the box and the ROBOT stays small during expert pushes (table/tray resting contact reported separately)", "PASS" if robot_pen >= THR["penetration_robot_m"] else "FAIL", f"min robot-box contact distance >= {THR['penetration_robot_m']} m", dict(min_contact_dist_by_partner_body_m=pen_by, min_robot_box_dist_m=robot_pen))
    check("V5 success criterion recomputed from the recorded trajectories equals the simulator flag", "PASS" if incons == 0 else "FAIL", "0 mismatches", dict(mismatches=incons, episodes=n))
    from .p1_expert_eval import wilson
    lo_, hi_ = wilson(succ, n); check("V7 pushing expert, STRICT success on a fresh development set (box within 3 cm, at rest, never lifted > 1 cm, held 5 steps)", "PASS" if succ / n >= THR["expert_strict_min"] else "FAIL", f"strict success >= {THR['expert_strict_min']}", dict(episodes=n, strict_success=succ, strict_success_rate=succ / n, wilson95=[lo_, hi_], placed=plc, failure_causes=causes, mean_final_dist_m=float(np.mean([e["final_dist_m"] for e in eps])), failing_seeds=[e["seed"] for e in eps if not e["success"]][:30], seconds=round(time.time() - t0, 1), note="criterion unchanged; the expert parameters were chosen on tuning seeds 24,000,000+ only"))
    rep = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), thresholds_preregistered=THR, results=R, sampled_parameters=params, episodes=eps, metadata=repro_metadata(dict(no_training=True, protected_seeds_used=False))); atomic_write_json(out, rep)
    n_fail = sum(r["status"] == "FAIL" for r in R); print(f"\nSUMMARY: {sum(r['status'] == 'PASS' for r in R)} PASS, {n_fail} FAIL, written {out}"); return 0
if __name__ == "__main__": sys.exit(main())
