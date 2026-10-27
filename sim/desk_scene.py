#!/usr/bin/env python3
"""OpenArm v1 desk scene: table + 3 free objects (blue box, red cylinder, green cube), built with MjSpec on top of
openarm_mujoco/v1/scene.xml (upstream files untouched). Checks AC-01..04 and renders views.
Usage: .venv/bin/python desk_scene.py [--viewer]   (headless cloud: MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa)
Dimensions/masses/friction are plausible placeholders, NOT measured from the real desk or objects."""
import os; os.environ.setdefault("MUJOCO_GL", "egl")
import sys, json, numpy as np, mujoco, imageio.v2 as iio
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)

TABLE_TOP_Z = 0.36                       # tcp sits at z≈0.477 in the "home" keyframe; objects <=6 cm tall leave >=5 cm clearance
TABLE_C, TABLE_HALF = np.array([0.48, 0.0]), np.array([0.22, 0.40])   # centre x,y / half extents x,y (0.44 x 0.80 m)
TOP_T, LEG_R = 0.02, 0.025
FRICTION = [0.8, 0.005, 0.0001]
# name -> (geom type, half-size, rgba, mass kg, xy position)
OBJECTS = {
    "blue_box":     (mujoco.mjtGeom.mjGEOM_BOX,      [0.025, 0.020, 0.020], [0.15, 0.35, 0.95, 1], 0.05, (0.42,  0.12)),
    "red_cylinder": (mujoco.mjtGeom.mjGEOM_CYLINDER, [0.020, 0.030, 0],     [0.90, 0.15, 0.15, 1], 0.08, (0.40, -0.12)),
    "green_cube":   (mujoco.mjtGeom.mjGEOM_BOX,      [0.020, 0.020, 0.020], [0.15, 0.75, 0.25, 1], 0.04, (0.52,  0.00)),
}

def build_spec(with_camera=True):
    spec = mujoco.MjSpec.from_file(str(HERE / "openarm_mujoco/v1/scene.xml"))
    wb = spec.worldbody
    table = wb.add_body(name="desk", pos=[*TABLE_C, 0])
    kw = dict(friction=FRICTION, rgba=[0.08, 0.08, 0.08, 1])
    table.add_geom(name="desk_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[*TABLE_HALF, TOP_T / 2], pos=[0, 0, TABLE_TOP_Z - TOP_T / 2], **kw)
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        h = (TABLE_TOP_Z - TOP_T) / 2
        table.add_geom(name=f"desk_leg{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[LEG_R, h, 0],
                       pos=[sx * (TABLE_HALF[0] - 0.03), sy * (TABLE_HALF[1] - 0.03), h], **kw)
    for name, (gt, size, rgba, mass, (x, y)) in OBJECTS.items():
        hz = size[2] if gt == mujoco.mjtGeom.mjGEOM_BOX else size[1]
        b = wb.add_body(name=name, pos=[x, y, TABLE_TOP_Z + hz + 0.0005])   # 0.5 mm clearance, settles on contact
        b.add_freejoint(name=f"{name}_free")
        b.add_geom(name=f"{name}_geom", type=gt, size=size, rgba=rgba, mass=mass, friction=FRICTION)
    if with_camera:   # same D435-like head camera as realsense_cam.py (spec values unverified vs datasheet)
        p = np.radians(35); c, s = np.cos(p), np.sin(p)
        spec.body("openarm_body_link0").add_camera(name="rs_d435", pos=[0.06, 0, 0.80], xyaxes=[0, -1, 0, s, 0, c], fovy=42.5)
    wb.add_camera(name="overview", pos=[1.25, -0.9, 0.95], xyaxes=[0.58, 0.81, 0, -0.30, 0.21, 0.93])
    return spec

def reset(m, d):
    mujoco.mj_resetData(m, d)
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    # keyframe qpos covers only the arm DOFs after adding free joints; copy by joint address instead of relying on size
    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    for j, n in enumerate(jn):
        if n and "free" not in n:
            a = m.jnt_qposadr[j]; d.qpos[a] = m.key_qpos[k][a]
    d.ctrl[:] = m.key_ctrl[k][:m.nu]
    mujoco.mj_forward(m, d)

def main():
    m = build_spec().compile(); d = mujoco.MjData(m)
    if "--viewer" in sys.argv:            # interactive window: desk + 3 objects + robot at the home pose
        import mujoco.viewer
        reset(m, d); mujoco.viewer.launch(m, d); return
    res = {"nq": m.nq, "nv": m.nv, "nu": m.nu, "nbody": m.nbody, "ngeom": m.ngeom}
    print("AC-01 loaded:", res)
    reset(m, d)
    tcp = {s: d.body(f"openarm_{s}_hand_tcp").xpos.copy() for s in ("left", "right")}
    p0 = {n: d.body(n).xpos.copy() for n in OBJECTS}
    bad_contact = lambda: [(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in d.contact[:d.ncon]]
    nan = False; maxv = 0.0
    for _ in range(5000):
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all(): nan = True; break
        for n in OBJECTS: maxv = max(maxv, np.linalg.norm(d.body(n).cvel[3:]))
    drift = {n: float(np.linalg.norm(d.body(n).xpos - p0[n])) for n in OBJECTS}
    zerr = {n: float(d.body(n).xpos[2] - p0[n][2]) for n in OBJECTS}
    speed = {n: float(np.linalg.norm(d.body(n).cvel[3:])) for n in OBJECTS}
    pairs = {tuple(sorted(p)) for p in bad_contact()}
    robot_obj = sorted(p for p in pairs if any(o in "".join(p) for o in OBJECTS) and "desk" not in "".join(p))
    robot_desk = sorted(p for p in pairs if any("desk" in x for x in p) and not any(o in "".join(p) for o in OBJECTS))
    print(f"AC-02 NaN={nan} steps=5000 | drift(m)={ {k: round(v,4) for k,v in drift.items()} } | final speed(m/s)={ {k: round(v,5) for k,v in speed.items()} }")
    print("  contacts object<->non-desk:", robot_obj, "| desk<->robot:", robot_desk)
    # AC-04: horizontal distance to nearest tcp + position-only damped-LS IK to hover 3 cm above each object's top
    reach = {}
    for n, (gt, size, _, _, _) in OBJECTS.items():
        reset(m, d)
        top = d.body(n).xpos + np.array([0, 0, (size[2] if gt == mujoco.mjtGeom.mjGEOM_BOX else size[1]) + 0.03])
        best = None; hmin = 9.0
        for s in ("left", "right"):
            dd = mujoco.MjData(m); reset(m, dd)
            bid = m.body(f"openarm_{s}_hand_tcp").id
            jids = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                    and m.body(m.jnt_bodyid[j]).name.startswith(f"openarm_{s}_link")]
            dofs = [m.jnt_dofadr[j] for j in jids]
            for _ in range(300):
                mujoco.mj_forward(m, dd)
                e = top - dd.body(bid).xpos
                if np.linalg.norm(e) < 1e-3: break
                J = np.zeros((3, m.nv)); mujoco.mj_jacBody(m, dd, J, None, bid)
                Jr = J[:, dofs]
                dq = Jr.T @ np.linalg.solve(Jr @ Jr.T + 1e-4 * np.eye(3), e)
                for k, j in enumerate(jids):
                    a = m.jnt_qposadr[j]; lo, hi = m.jnt_range[j]
                    dd.qpos[a] = np.clip(dd.qpos[a] + 0.5 * dq[k], lo, hi)
            err = float(np.linalg.norm(top - dd.body(bid).xpos))
            hd = float(np.linalg.norm((d.body(n).xpos - tcp[s])[:2]))
            hmin = min(hmin, hd)
            if best is None or err < best[1]: best = (s, err, hd)
        reach[n] = {"arm": best[0], "ik_residual_m": round(best[1], 4), "horiz_dist_nearest_home_tcp_m": round(hmin, 3)}
    print("AC-04 reach (position-only IK, joint limits on, no collision check, no orientation):")
    for n, r in reach.items(): print("  ", n, r)
    # renders
    r = mujoco.Renderer(m, 480, 640)
    for cam in ("rs_d435", "overview"):
        reset(m, d)
        for _ in range(500): mujoco.mj_step(m, d)
        r.update_scene(d, camera=cam); iio.imwrite(OUT / f"desk_{cam}.png", r.render())
    r.close()
    json.dump({"model": res, "drift": drift, "zerr": zerr, "speed": speed, "nan": nan,
               "robot_object_contacts": robot_obj, "reach": reach,
               "tcp_home": {k: v.round(3).tolist() for k, v in tcp.items()},
               "obj_home": {k: v.round(3).tolist() for k, v in p0.items()}}, open(OUT / "desk_scene_check.json", "w"), indent=1)

if __name__ == "__main__":
    main()
