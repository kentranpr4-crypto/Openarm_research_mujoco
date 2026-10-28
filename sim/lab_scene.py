#!/usr/bin/env python3
"""OpenArm v1 Robotics Manipulation Lab: floor, walls, black workbench, 3 objects, target tray, studio lighting, cameras.

Built with MjSpec on top of the UNMODIFIED upstream openarm_mujoco/v1/openarm_bimanual.xml (we do not use upstream
scene.xml, so floor/sky/lights are ours). Appearance and physics are separate layers, following the upstream convention:
    visual    geoms: group 2, contype=conaffinity=0, density 0  (materials/textures, never collide, add no mass)
    collision geoms: group 3, contype=conaffinity=1             (hidden by default, carry mass/friction)
Visual-only decoration lives outside the workspace volume WORKSPACE_BOX (checked by `--check`).

Usage (from sim/):  python lab_scene.py [--check] [--render] [--viewer] [--export-xml]
Headless cloud:     MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python lab_scene.py --check --render
All dimensions / masses / friction are placeholders, not measured from a real bench."""
import os; os.environ.setdefault("MUJOCO_GL", "egl")
import sys, json, argparse, subprocess, numpy as np, mujoco
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cameras import add_cameras, CameraRig, ALL_CAMERAS, OVERVIEW
from ik import ArmIK
import imageio.v2 as iio

UPSTREAM = HERE / "openarm_mujoco/v1/openarm_bimanual.xml"
ASSETS = HERE / "assets"
OUTDIR = HERE.parent / "outputs" / "lab_scene"

# ---- layout (metres; robot base at origin facing +x) -------------------------------------------------------------
TABLE_TOP_Z, TABLE_C, TABLE_HALF = 0.36, (0.48, 0.0), (0.22, 0.40)      # same bench as desk_scene.py
TOP_T, LEG_HALF = 0.02, 0.025
FRICTION = [0.8, 0.005, 0.0001]
# tray: inner 14 cm (> finger span ~11 cm while holding the box), rim 12 mm (< finger bottom when grasping at GRASP_DZ); centre x<=0.44 keeps
# the left TCP inside its reach envelope at place height (IK map: err<1 mm for x<=0.44, y<=0.30; >10 mm beyond x=0.48)
TRAY_C, TRAY_INNER, TRAY_RIM_H, TRAY_RIM_T, TRAY_BASE_T = (0.44, 0.30), (0.14, 0.14), 0.012, 0.006, 0.004
OBJECTS = {   # name -> (shape, half-size [x,y,z] or [r,half-h], rgba, mass kg, (x,y))   same as desk_scene.py
    "blue_box":     ("box", [0.025, 0.020, 0.020], [0.12, 0.33, 0.95, 1], 0.05, (0.42, 0.12)),
    "red_cylinder": ("cyl", [0.020, 0.030],        [0.90, 0.15, 0.15, 1], 0.08, (0.40, -0.12)),
    "green_cube":   ("box", [0.020, 0.020, 0.020], [0.15, 0.75, 0.25, 1], 0.04, (0.52, 0.00)),
}
WORKSPACE_BOX = np.array([[0.20, -0.45, 0.36], [0.75, 0.45, 0.85]])      # nothing but desk/tray/objects/robot may enter
ROOM = dict(x0=-0.8, x1=2.4, hy=2.0, h=2.8, t=0.05)

G, MT = mujoco.mjtGeom, mujoco

def _vis(body, name, gtype, size, pos, material=None, rgba=None, euler=None):
    kw = dict(name=name, type=gtype, size=size, pos=list(pos), group=2, contype=0, conaffinity=0, density=0)
    if material: kw["material"] = material
    if rgba is not None: kw["rgba"] = list(rgba)
    if euler is not None: kw["euler"] = list(euler)
    return body.add_geom(**kw)

def _col(body, name, gtype, size, pos, mass=None, friction=FRICTION):
    kw = dict(name=name, type=gtype, size=size, pos=list(pos), group=3, contype=1, conaffinity=1, friction=friction,
              rgba=[1, 0, 1, 0.3])
    if mass is not None: kw["mass"] = mass
    return body.add_geom(**kw)

def _mat(spec, name, tex=None, rgba=(1, 1, 1, 1), repeat=(1, 1), uniform=False, reflectance=0.0, specular=0.3, shininess=0.3, emission=0.0):
    m = spec.add_material(name=name, rgba=list(rgba), reflectance=reflectance, specular=specular, shininess=shininess, emission=emission)
    if tex:
        m.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = tex; m.texrepeat = list(repeat); m.texuniform = uniform
    return m

def build_lab_spec(with_cameras=True, assets_dir=ASSETS):
    spec = mujoco.MjSpec.from_file(str(UPSTREAM))
    spec.texturedir = str(assets_dir)
    spec.modelname = "openarm_v1_lab"
    wb = spec.worldbody
    # ---- global look: studio ambient, high-res shadow map, big offscreen buffer ----------------------------------
    v = spec.visual
    v.headlight.ambient = [0.28, 0.28, 0.30]; v.headlight.diffuse = [0.25, 0.25, 0.25]; v.headlight.specular = [0, 0, 0]
    v.quality.shadowsize = 8192; v.quality.offsamples = 8
    v.global_.offwidth, v.global_.offheight = 1280, 960
    v.rgba.haze = [0.80, 0.82, 0.85, 1]
    v.map.shadowscale = 0.6          # tighter shadow frustum = sharper shadows
    spec.stat.center = [0.40, 0.0, 0.42]; spec.stat.extent = 0.9; spec.stat.meansize = 0.05   # viewer frames the bench, not the room
    v.global_.azimuth, v.global_.elevation = 145.0, -18.0

    # ---- textures + materials ----------------------------------------------------------------------------------
    for n in ("floor_concrete", "wall_paint", "table_laminate", "tray_metal"):
        spec.add_texture(name=n, type=mujoco.mjtTexture.mjTEXTURE_2D, file=f"{n}.png")
    spec.add_texture(name="studio_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.92, 0.93, 0.95], rgb2=[0.55, 0.58, 0.64], width=512, height=3072)
    _mat(spec, "m_floor", "floor_concrete", repeat=(1.2, 1.2), uniform=True, reflectance=0.12, specular=0.4, shininess=0.5)
    _mat(spec, "m_wall", "wall_paint", repeat=(1, 1), uniform=True, specular=0.05, shininess=0.1)
    _mat(spec, "m_wall_dark", rgba=(0.20, 0.23, 0.28, 1), specular=0.1, shininess=0.2)
    _mat(spec, "m_table", "table_laminate", repeat=(2, 2), uniform=True, specular=0.5, shininess=0.6, reflectance=0.04)
    _mat(spec, "m_tray", "tray_metal", repeat=(2, 2), uniform=True, specular=0.7, shininess=0.7)
    _mat(spec, "m_edge", rgba=(0.75, 0.77, 0.80, 1), specular=0.8, shininess=0.8)
    _mat(spec, "m_leg", rgba=(0.16, 0.17, 0.19, 1), specular=0.5, shininess=0.5)
    _mat(spec, "m_zone", rgba=(1.0, 0.78, 0.10, 1), specular=0.1, shininess=0.1)
    _mat(spec, "m_mat", rgba=(0.12, 0.14, 0.17, 1), specular=0.05, shininess=0.1)
    _mat(spec, "m_lamp", rgba=(1, 1, 1, 1), emission=1.0)
    for n, rgba in (("m_blue", OBJECTS["blue_box"][2]), ("m_red", OBJECTS["red_cylinder"][2]), ("m_green", OBJECTS["green_cube"][2])):
        _mat(spec, n, rgba=rgba, specular=0.6, shininess=0.6)

    # ---- room (visual only; robot cannot reach walls) + floor ---------------------------------------------------
    room = wb.add_body(name="room")
    r = ROOM; cx, cz = (r["x0"] + r["x1"]) / 2, r["h"] / 2
    _col(wb, "floor_collision", G.mjGEOM_PLANE, [0, 0, 0.05], [0, 0, 0])                       # physics floor (group 3)
    _vis(room, "floor_visual", G.mjGEOM_PLANE, [3.0, 3.0, 0.05], [cx, 0, 0], material="m_floor")
    _vis(room, "wall_back", G.mjGEOM_BOX, [r["t"], r["hy"], cz], [r["x0"] - r["t"], 0, cz], material="m_wall")
    _vis(room, "wall_front", G.mjGEOM_BOX, [r["t"], r["hy"], cz], [r["x1"] + r["t"], 0, cz], material="m_wall")
    _vis(room, "wall_left", G.mjGEOM_BOX, [(r["x1"] - r["x0"]) / 2, r["t"], cz], [cx, r["hy"] + r["t"], cz], material="m_wall")
    _vis(room, "wall_right", G.mjGEOM_BOX, [(r["x1"] - r["x0"]) / 2, r["t"], cz], [cx, -r["hy"] - r["t"], cz], material="m_wall")
    _vis(room, "deco_back_wainscot", G.mjGEOM_BOX, [0.004, r["hy"], 0.45], [r["x0"] + 0.004, 0, 0.45], material="m_wall_dark")
    _vis(room, "deco_back_skirting", G.mjGEOM_BOX, [0.006, r["hy"], 0.04], [r["x0"] + 0.012, 0, 0.04], material="m_edge")
    _vis(room, "deco_floor_mat", G.mjGEOM_BOX, [0.85, 0.75, 0.001], [0.45, 0, 0.001], material="m_mat")   # 2 mm mat under bench
    for i, y in enumerate((-0.9, 0.0, 0.9)):                                                              # ceiling light panels
        _vis(room, f"deco_ceiling_lamp{i}", G.mjGEOM_BOX, [0.45, 0.30, 0.01], [0.55, y, r["h"] - 0.01], material="m_lamp")

    # ---- workbench: collision (group 3) and visual (group 2) share the same dimensions --------------------------
    desk = wb.add_body(name="desk", pos=[*TABLE_C, 0])
    hx, hy = TABLE_HALF
    top_c = [0, 0, TABLE_TOP_Z - TOP_T / 2]
    _col(desk, "desk_top_collision", G.mjGEOM_BOX, [hx, hy, TOP_T / 2], top_c)
    _vis(desk, "desk_top_visual", G.mjGEOM_BOX, [hx, hy, TOP_T / 2], top_c, material="m_table")
    for tag, (px, py, sx, sy) in {"xp": (hx, 0, 0.004, hy), "xn": (-hx, 0, 0.004, hy), "yp": (0, hy, hx, 0.004), "yn": (0, -hy, hx, 0.004)}.items():
        _vis(desk, f"desk_trim_{tag}", G.mjGEOM_BOX, [sx, sy, TOP_T / 2 + 0.0005], [px, py, TABLE_TOP_Z - TOP_T / 2], material="m_edge")   # metal edge trim
    leg_h = (TABLE_TOP_Z - TOP_T) / 2
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        p = [sx * (hx - 0.03), sy * (hy - 0.03), leg_h]
        _col(desk, f"desk_leg{i}_collision", G.mjGEOM_BOX, [LEG_HALF, LEG_HALF, leg_h], p)
        _vis(desk, f"desk_leg{i}_visual", G.mjGEOM_BOX, [LEG_HALF, LEG_HALF, leg_h], p, material="m_leg")

    # ---- target tray (static): base + 4 rims, yellow zone tape around it ---------------------------------------
    tray = wb.add_body(name="tray", pos=[*TRAY_C, TABLE_TOP_Z])
    ix, iy = TRAY_INNER[0] / 2, TRAY_INNER[1] / 2
    _col(tray, "tray_base_collision", G.mjGEOM_BOX, [ix + TRAY_RIM_T, iy + TRAY_RIM_T, TRAY_BASE_T / 2], [0, 0, TRAY_BASE_T / 2])
    _vis(tray, "tray_base_visual", G.mjGEOM_BOX, [ix + TRAY_RIM_T, iy + TRAY_RIM_T, TRAY_BASE_T / 2], [0, 0, TRAY_BASE_T / 2], material="m_tray")
    rim_z = TRAY_BASE_T + TRAY_RIM_H / 2
    for n, (px, py, sx, sy) in {"xp": (ix + TRAY_RIM_T / 2, 0, TRAY_RIM_T / 2, iy + TRAY_RIM_T), "xn": (-ix - TRAY_RIM_T / 2, 0, TRAY_RIM_T / 2, iy + TRAY_RIM_T),
                                "yp": (0, iy + TRAY_RIM_T / 2, ix, TRAY_RIM_T / 2), "yn": (0, -iy - TRAY_RIM_T / 2, ix, TRAY_RIM_T / 2)}.items():
        _col(tray, f"tray_rim_{n}_collision", G.mjGEOM_BOX, [sx, sy, TRAY_RIM_H / 2], [px, py, rim_z])
        _vis(tray, f"tray_rim_{n}_visual", G.mjGEOM_BOX, [sx, sy, TRAY_RIM_H / 2], [px, py, rim_z], material="m_tray")
    zx, zy, tape = ix + TRAY_RIM_T + 0.025, iy + TRAY_RIM_T + 0.025, 0.006       # place-zone border (visual decal, 1 mm)
    for n, (px, py, sx, sy) in {"xp": (zx, 0, tape, zy + tape), "xn": (-zx, 0, tape, zy + tape), "yp": (0, zy, zx, tape), "yn": (0, -zy, zx, tape)}.items():
        _vis(tray, f"zone_tape_{n}", G.mjGEOM_BOX, [sx, sy, 0.0005], [px, py, 0.0005], material="m_zone")

    # ---- objects: free bodies; collision primitive carries mass/friction, visual twin carries the look ----------
    for name, (shape, size, rgba, mass, (x, y)) in OBJECTS.items():
        hz = size[2] if shape == "box" else size[1]
        b = wb.add_body(name=name, pos=[x, y, TABLE_TOP_Z + hz + 0.0005])
        b.add_freejoint(name=f"{name}_free")
        gt = G.mjGEOM_BOX if shape == "box" else G.mjGEOM_CYLINDER
        sz = size if shape == "box" else [size[0], size[1], 0]
        _col(b, f"{name}_collision", gt, sz, [0, 0, 0], mass=mass)
        _vis(b, f"{name}_visual", gt, sz, [0, 0, 0], material={"blue_box": "m_blue", "red_cylinder": "m_red", "green_cube": "m_green"}[name])

    # ---- studio lighting: key (shadow) + fill + rim. MuJoCo's classic renderer has hard shadow maps; "softness" comes
    #      from the shadow-map resolution, ambient headlight and the 3 lights, NOT from physically soft shadows. -----
    key = dict(type=mujoco.mjtLightType.mjLIGHT_SPOT, cutoff=55, exponent=6, attenuation=[1, 0, 0])
    wb.add_light(name="key_light", pos=[1.3, -0.9, 2.1], dir=[-0.55, 0.42, -0.72], diffuse=[0.85, 0.83, 0.80], specular=[0.35, 0.35, 0.35], castshadow=True, **key)
    wb.add_light(name="fill_light", pos=[1.2, 1.1, 1.8], dir=[-0.5, -0.5, -0.7], diffuse=[0.35, 0.38, 0.44], specular=[0.05, 0.05, 0.05], castshadow=False, **key)
    wb.add_light(name="rim_light", pos=[-0.5, 0.0, 1.9], dir=[0.6, 0.0, -0.8], diffuse=[0.30, 0.30, 0.32], specular=[0.1, 0.1, 0.1], castshadow=False, **key)
    if with_cameras: add_cameras(spec)
    return spec

def compile_lab(**kw):
    m = build_lab_spec(**kw).compile(); d = mujoco.MjData(m); reset_home(m, d); return m, d

def reset_home(m, d):
    """Arms to keyframe 'home', objects at their authored poses (free-joint qpos from the compiled model defaults)."""
    mujoco.mj_resetData(m, d)
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE: d.qpos[m.jnt_qposadr[j]] = m.key_qpos[k][m.jnt_qposadr[j]]
    d.ctrl[:] = m.key_ctrl[k][:m.nu]; mujoco.mj_forward(m, d)

# =====================================================================================================================
# acceptance checks
# =====================================================================================================================
def _aabb(m, d, g):
    """World AABB of geom g (exact for primitives via size + rotation, mesh via vertices)."""
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]; vs = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = vs @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]; return w.min(0), w.max(0)
    s = m.geom_size[g].copy()
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE: s = np.array([s[0] or 5, s[1] or 5, 0.0])   # infinite plane: zero thickness
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER: s = np.array([s[0], s[0], s[1]])
    ext = np.abs(d.geom_xmat[g].reshape(3, 3)) @ s
    return d.geom_xpos[g] - ext, d.geom_xpos[g] + ext

def _inertia_box(mass, h): return mass / 3 * np.array([h[1]**2 + h[2]**2, h[0]**2 + h[2]**2, h[0]**2 + h[1]**2])
def _inertia_cyl(mass, r, hh): return np.array([mass / 12 * (3 * r * r + 4 * hh * hh)] * 2 + [mass * r * r / 2])

def run_checks(out=None):
    res, ok = {}, {}
    m, d = compile_lab(); out = out or OUTDIR; out.mkdir(parents=True, exist_ok=True)
    # AC-01 load
    res["AC-01"] = dict(nq=m.nq, nv=m.nv, nu=m.nu, nbody=m.nbody, ngeom=m.ngeom, ncam=m.ncam, nlight=m.nlight, ntex=m.ntex, nmat=m.nmat)
    cams = [m.camera(i).name for i in range(m.ncam)]; res["AC-01"]["cameras"] = cams
    ok["AC-01"] = (m.nu == 16 and m.nq == 18 + 21 and m.nv == 18 + 18 and all(c in cams for c in ALL_CAMERAS) and m.nlight == 3)
    # AC-02 stability 5 s with the arms holding 'home' ctrl
    p0 = {n: d.body(n).xpos.copy() for n in OBJECTS}; nan = False
    for _ in range(5000):
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all(): nan = True; break
    drift = {n: float(np.linalg.norm(d.body(n).xpos - p0[n])) for n in OBJECTS}
    speed = {n: float(np.linalg.norm(d.body(n).cvel[3:])) for n in OBJECTS}
    gname = lambda g: m.geom(g).name
    robot_hits = sorted({(gname(c.geom1), gname(c.geom2)) for c in d.contact[:d.ncon]
                         if (any(o in gname(c.geom1) + gname(c.geom2) for o in OBJECTS) or "desk" in gname(c.geom1) + gname(c.geom2) or "tray" in gname(c.geom1) + gname(c.geom2))
                         and "openarm" in gname(c.geom1) + gname(c.geom2)})
    inertia_err = {}
    for n, (shape, size, _, mass, _) in OBJECTS.items():
        b = m.body(n).id; ref = _inertia_box(mass, size) if shape == "box" else _inertia_cyl(mass, *size)
        inertia_err[n] = float(np.max(np.abs(np.sort(m.body_inertia[b]) - np.sort(ref)) / np.sort(ref)))
    res["AC-02"] = dict(nan=nan, drift_m=drift, final_speed_mps=speed, robot_contacts=robot_hits,
                        mass_kg={n: float(m.body_mass[m.body(n).id]) for n in OBJECTS}, inertia_rel_err_vs_analytic=inertia_err)
    ok["AC-02"] = (not nan and max(drift.values()) < 2e-3 and max(speed.values()) < 1e-3 and not robot_hits and max(inertia_err.values()) < 1e-3)
    # AC-03 rendering: segmentation proves who is visible in the overview camera
    reset_home(m, d)
    for _ in range(300): mujoco.mj_step(m, d)
    vis = {}
    with CameraRig(m, 640, 480) as rig:
        seg = rig.segmentation(d, OVERVIEW); ids, types = seg[..., 0], seg[..., 1]
        bodies = np.array([m.body(m.geom_bodyid[i]).name if t == mujoco.mjtObj.mjOBJ_GEOM and i >= 0 else "" for i, t in zip(ids.ravel(), types.ravel())])
        for key, pred in {"robot": lambda s: s.startswith("openarm"), "desk": lambda s: s == "desk", "tray": lambda s: s == "tray",
                          **{n: (lambda s, n=n: s == n) for n in OBJECTS}}.items():
            vis[key] = int(np.count_nonzero([pred(s) for s in bodies]))
    res["AC-03"] = dict(overview_visible_pixels=vis); ok["AC-03"] = all(v > 150 for v in vis.values())
    # AC-04 reach: blue box (grasp pose, pre-grasp hover) and tray centre, hand pointing +x, left arm, 6-DoF IK
    ik = ArmIK(m, "left"); reset_home(m, d); R0 = d.body("openarm_left_hand_tcp").xmat.reshape(3, 3).copy()
    targets = {"blue_box_grasp": d.body("blue_box").xpos + [0.01, 0, 0.005], "blue_box_pregrasp": d.body("blue_box").xpos + [-0.09, 0, 0.005],
               "tray_drop": np.array([*TRAY_C, TABLE_TOP_Z + TRAY_BASE_T + 0.050])}
    reach = {}
    for k, t in targets.items():
        dd = mujoco.MjData(m); reset_home(m, dd); conv = ik.solve(dd, t, R0)
        reach[k] = dict(converged=bool(conv), pos_err_m=float(np.linalg.norm(t - dd.xpos[ik.tcp])), target=np.round(t, 3).tolist())
    res["AC-04"] = reach; ok["AC-04"] = all(r["converged"] for r in reach.values())
    # AC-05 visual / collision separation + upstream untouched
    bad = []
    for g in range(m.ngeom):
        grp, ct, ca, nm = int(m.geom_group[g]), int(m.geom_contype[g]), int(m.geom_conaffinity[g]), gname(g)
        if nm and not nm.startswith("openarm"):          # named geoms = ours; upstream geoms are all "openarm_*"
            if grp == 2 and (ct or ca): bad.append(("visual-collides", nm))
            if grp == 3 and not (ct and ca): bad.append(("collision-inert", nm))
            if grp not in (2, 3): bad.append(("bad-group", nm))
    git = subprocess.run(["git", "-C", str(HERE / "openarm_mujoco"), "status", "--porcelain"], capture_output=True, text=True)
    res["AC-05"] = dict(violations=bad, upstream_git_status=git.stdout.strip() or "clean", upstream_rc=git.returncode)
    ok["AC-05"] = (not bad and git.returncode == 0 and not git.stdout.strip())
    # AC-06 no obstruction: deco geoms, lights and world cameras outside WORKSPACE_BOX
    reset_home(m, d); lo, hi = WORKSPACE_BOX; hits = []
    inside = lambda a, b: bool(np.all(a < hi) and np.all(b > lo))
    for g in range(m.ngeom):
        nm = gname(g)
        if nm and (nm.startswith(("deco_", "wall_", "floor_visual")) ) and inside(*_aabb(m, d, g)): hits.append(nm)
    for i in range(m.nlight):
        if np.all(d.light_xpos[i] > lo) and np.all(d.light_xpos[i] < hi): hits.append(m.light(i).name)
    for i in range(m.ncam):
        if m.cam_bodyid[i] == 0 and np.all(d.cam_xpos[i] > lo) and np.all(d.cam_xpos[i] < hi): hits.append(m.camera(i).name)
    res["AC-06"] = dict(intruders=hits); ok["AC-06"] = not hits
    res["pass"] = ok; res["all_pass"] = all(ok.values())
    json.dump(res, open(out / "lab_check.json", "w"), indent=1, default=float)
    return res

def render_all(out=None, w=960, h=720):
    out = out or OUTDIR; out.mkdir(parents=True, exist_ok=True)
    m, d = compile_lab()
    for _ in range(600): mujoco.mj_step(m, d)
    imgs = {}
    with CameraRig(m, w, h) as rig:
        for cam in ALL_CAMERAS:
            imgs[cam] = rig.rgb(d, cam); iio.imwrite(out / f"{cam}.png", imgs[cam])
    top = np.concatenate([imgs[OVERVIEW], imgs["front"]], 1); bot = np.concatenate([imgs["wrist_left"], imgs["wrist_right"]], 1)
    iio.imwrite(out / "montage.png", np.concatenate([top, bot], 0)[::2, ::2])
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--render", action="store_true")
    ap.add_argument("--viewer", action="store_true"); ap.add_argument("--export-xml", action="store_true"); ap.add_argument("--out", type=Path, default=OUTDIR)
    a = ap.parse_args()
    if not any([a.check, a.render, a.viewer, a.export_xml]): a.check = True
    if a.export_xml:
        from export_scene import export_xml; print("exported:", export_xml(a.out))
    if a.check:
        r = run_checks(a.out); print(json.dumps({k: r[k] for k in r if k not in ("pass",)}, indent=1, default=float)[:3000]); print("PASS:", r["pass"], "| ALL:", r["all_pass"])
    if a.render: print("rendered to", render_all(a.out))
    if a.viewer:
        import mujoco.viewer
        m, d = compile_lab(); mujoco.viewer.launch(m, d)

if __name__ == "__main__":
    main()
