"""READ-ONLY audit of the PickPlace contact / friction behaviour (does NOT modify any file and does NOT apply the pushing fix).  Writes one NEW json.
Questions: which geom pairs touch, what contact priority / friction does MuJoCo actually use for each pair, and does the task's box-friction randomisation (scale 0.8-1.2 of 0.8) change the physics of the pairs that matter for PICK-AND-PLACE
(box-desk, finger-box = grasp, box-tray)?  Effective friction is read from `d.contact[i].friction` of the actual contact, not inferred from the XML.
Usage (repo root):  python -m research.supermimic_openarm.p1_pickplace_audit --out outputs/p1_validation/pickplace_audit.json"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(argv); out = Path(a.out)
    if out.exists(): sys.exit(f"refusing to overwrite {out}")
    import mujoco
    from .envs.openarm_env import OpenArmTaskEnv
    from .utils import atomic_write_json, repro_metadata
    env = OpenArmTaskEnv("pick_place", None, 0, 20); m, d = env.m, env.d; box = set(env.box_geoms); rep = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())); name = lambda g: f"{m.body(m.geom_bodyid[g]).name}/{m.geom(g).name or g}"
    def collide_geoms(body_names): return [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in body_names and m.geom_contype[g]]
    partners = {"desk": collide_geoms(["desk"]), "tray": collide_geoms(["tray"]), "finger": collide_geoms(["openarm_left_right_finger", "openarm_left_left_finger"]), "red_cylinder": collide_geoms(["red_cylinder"]), "green_cube": collide_geoms(["green_cube"])}
    rep["geoms"] = {k: [dict(name=name(g), friction=[float(x) for x in m.geom_friction[g]], priority=int(m.geom_priority[g]), condim=int(m.geom_condim[g]), solmix=float(m.geom_solmix[g])) for g in v] for k, v in partners.items()}
    rep["box"] = [dict(name=name(g), friction=[float(x) for x in m.geom_friction[g]], priority=int(m.geom_priority[g]), condim=int(m.geom_condim[g])) for g in sorted(box) if m.geom_contype[g]]
    rep["task_randomisation"] = dict(box_friction_scale=list(env.task.p["friction"]), note="the task scales the box geom's sliding friction; 0.8 x scale")
    # --- contact combination rule, evaluated analytically from the model (MuJoCo: higher priority wins; equal priority -> element-wise max) -------------------------------------------
    def combined(g1, g2):
        p1, p2 = m.geom_priority[g1], m.geom_priority[g2]; f1, f2 = m.geom_friction[g1], m.geom_friction[g2]
        return (f1 if p1 > p2 else f2 if p2 > p1 else np.maximum(f1, f2)), ("geom1" if p1 > p2 else "geom2" if p2 > p1 else "max")
    bcol = [g for g in sorted(box) if m.geom_contype[g]][0]; table = {}
    for scale in (0.8, 0.9, 1.0, 1.1, 1.2):
        env.task.sample = (lambda rng, _s=scale, _b=type(env.task).sample.__get__(env.task): {**_b(rng), "friction_scale": _s}); env.reset(3); row = {}
        for pname, gl in partners.items():
            if not gl: continue
            mus = []; rules = set()
            for g in gl:
                f, rule = combined(bcol, g); mus.append(float(f[0])); rules.add(rule)
            row[pname] = dict(effective_sliding_mu_min=min(mus), effective_sliding_mu_max=max(mus), rule=sorted(rules))
        table[str(scale)] = dict(box_geom_mu=float(m.geom_friction[bcol][0]), pairs=row)
    rep["effective_friction_by_scale_analytic"] = table
    # --- measured on real contacts: box resting on the desk (and on the tray after a forced placement) ---------------------------------------------------------------------------------
    meas = {}
    for scale in (0.8, 1.0, 1.2):
        env.task.sample = (lambda rng, _s=scale, _b=type(env.task).sample.__get__(env.task): {**_b(rng), "friction_scale": _s}); env.reset(3); mus = {}
        for c in d.contact[:d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in box) != (g2 in box): oth = g2 if g1 in box else g1; mus.setdefault(m.body(m.geom_bodyid[oth]).name, []).append(float(c.friction[0]))
        meas[str(scale)] = {k: [min(v), max(v)] for k, v in mus.items()}
        d.qpos[env.box_adr:env.box_adr + 3] = [0.44, 0.30, 0.3805 + 0.004]; d.qvel[:] = 0; mujoco.mj_forward(m, d)
        for _ in range(500): mujoco.mj_step(m, d)
        tray = [float(c.friction[0]) for c in d.contact[:d.ncon] if (int(c.geom1) in box) != (int(c.geom2) in box) and m.body(m.geom_bodyid[int(c.geom2) if int(c.geom1) in box else int(c.geom1)]).name == "tray"]
        meas[str(scale)]["tray_after_placement"] = [min(tray), max(tray)] if tray else None
    rep["measured_contact_friction_on_real_contacts"] = meas
    # --- finger-box: is the grasp friction randomised? (fingers vs box, from the combination rule at the pick-place scale range) ---------------------------------------------------------
    f_rows = [table[k]["pairs"]["finger"] for k in table]; desk_rows = [table[k]["pairs"]["desk"] for k in table]; tray_rows = [table[k]["pairs"]["tray"] for k in table]
    spread = lambda rows: max(r["effective_sliding_mu_max"] for r in rows) - min(r["effective_sliding_mu_min"] for r in rows)
    rep["findings"] = dict(grasp_finger_box_effective_mu_spread_over_scale_range=spread(f_rows), box_desk_effective_mu_spread=spread(desk_rows), box_tray_effective_mu_spread=spread(tray_rows),
        fraction_of_scale_range_with_effect_on_desk=float(np.mean([desk_rows[i]["effective_sliding_mu_max"] > desk_rows[0]["effective_sliding_mu_max"] + 1e-9 for i in range(len(desk_rows))])),
        grasp_friction_randomised=bool(spread(f_rows) > 1e-9), box_desk_defect_confirmed=bool(any(r["rule"] == ["max"] for r in desk_rows) and spread(desk_rows) < (0.96 - 0.64) - 1e-9), box_tray_defect_confirmed=bool(any(r["rule"] == ["max"] for r in tray_rows) and spread(tray_rows) < (0.96 - 0.64) - 1e-9))
    rep["metadata"] = repro_metadata(dict(read_only=True, modifies_pickplace=False)); atomic_write_json(out, rep); print(json.dumps(rep["findings"], indent=1)); print(json.dumps(rep["effective_friction_by_scale_analytic"], indent=0)[:1500]); print("measured:", json.dumps(meas)); print("written", out); return 0
if __name__ == "__main__": sys.exit(main())
