"""Regression tests for the repaired P1 (pushing) environment.  Run: python -m pytest research/supermimic_openarm/tests/test_p1_env.py -q
Covers: effective friction of the actual box-desk pair, mass/inertia consistency, goal validity, strict success criterion unchanged, deterministic reset + bit-exact snapshot replay,
executed-action alignment, action safety (clip + slew), reward correctness, and P0 compatibility (reaching / pick_place trajectories bit-identical to git HEAD)."""
import os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pytest
mujoco = pytest.importorskip("mujoco")
from research.supermimic_openarm.envs.openarm_env import OpenArmTaskEnv
from research.supermimic_openarm.envs.tasks import Pushing, BOX_REST_Z
from research.supermimic_openarm.envs import layout as L
from research.supermimic_openarm.baselines.push_expert import MAX_STEP
REPO = Path(__file__).resolve().parents[3]; SEEDS = range(23_400_000, 23_400_040)
@pytest.fixture(scope="module")
def env(): return OpenArmTaskEnv("pushing", None, 150, 20)
def _fixed(e, **kw):
    """pin sampled parameters (everything else from the real sampler) for physical measurements"""
    base = type(e.task).sample.__get__(e.task); e.task.sample = lambda rng: {**base(rng), "box_xy": (0.44, 0.12), "goal": np.array([0.5, 0.12]), "box_yaw": 0.0, **kw}
    return e
def _slide(e, v0=0.5):
    d = e.d; d.qvel[e.box_dof] = v0; x0 = d.qpos[e.box_adr]
    for _ in range(1500): mujoco.mj_step(e.m, d)
    return float(d.qpos[e.box_adr] - x0)
def _box_desk_mu(e):
    bg, desk = e.m.geom("blue_box_collision").id, e.m.geom("desk_top_collision").id
    mus = [float(c.friction[0]) for c in e.d.contact[:e.d.ncon] if {c.geom1, c.geom2} == {bg, desk}]; assert mus; return mus[0]
def test_effective_friction_matches_sampled_value_and_physics(env):
    e = OpenArmTaskEnv("pushing", dict(friction_mode="pair_exact"), 150, 20); out = {}
    for sc in (0.7, 0.85, 1.0, 1.15, 1.3):
        _fixed(e, friction_scale=sc, mass_scale=1.0); e.reset(1); mu = _box_desk_mu(e); assert mu == pytest.approx(0.8 * sc, abs=1e-9)     # effective contact friction of the ACTUAL pair
        dist = _slide(e); theory = 0.5 ** 2 / (2 * mu * 9.81); out[sc] = dist; assert dist == pytest.approx(theory, rel=0.03)          # Coulomb stopping distance v^2/(2 mu g)
    v = [out[k] for k in (0.7, 0.85, 1.0, 1.15, 1.3)]; assert all(a > b for a, b in zip(v, v[1:])) and v[0] / v[-1] > 1.7              # strictly decreasing, large spread
def test_legacy_mode_reproduces_the_defect_and_is_not_default(env):
    assert env.task.p["friction_mode"] == "pair_exact"
    e = OpenArmTaskEnv("pushing", dict(friction_mode="legacy"), 150, 20); d = {}
    for sc in (0.7, 1.0): _fixed(e, friction_scale=sc, mass_scale=1.0); e.reset(1); d[sc] = _slide(e)
    assert d[0.7] == pytest.approx(d[1.0], rel=1e-9)                                                                                       # scale <= 1 had no effect (documented defect)
def test_mass_and_inertia_consistent_with_box_geometry(env):
    full = 2 * np.array([0.025, 0.020, 0.020]); seen = []
    for sd in SEEDS:
        env.reset(sd); inf = env.episode_info; m = env.m.body_mass[env.box_body]; I = env.m.body_inertia[env.box_body]
        assert 0.6 * env._mass0 - 1e-12 <= m <= 1.6 * env._mass0 + 1e-12 and inf["mass_kg"] == pytest.approx(m)
        exp = m / 12.0 * np.array([full[1] ** 2 + full[2] ** 2, full[0] ** 2 + full[2] ** 2, full[0] ** 2 + full[1] ** 2]); assert np.allclose(I, exp, rtol=1e-9)    # uniform cuboid
        assert 300.0 <= m / float(np.prod(full)) <= 1100.0; assert env.m.body_subtreemass[env.box_body] == pytest.approx(m); seen.append(m)
    assert max(seen) / min(seen) > 1.5                                                                                                    # really randomised, absolute (not cumulative) each reset
    env.reset(SEEDS[0]); a = env.m.body_mass[env.box_body]; env.reset(SEEDS[1]); env.reset(SEEDS[0]); assert env.m.body_mass[env.box_body] == a
def test_mass_changes_the_dynamics():
    acc = []
    for ms in (0.6, 1.0, 1.6):
        e = _fixed(OpenArmTaskEnv("pushing", None, 150, 20), friction_scale=1.0, mass_scale=ms); e.reset(1); e.d.xfrc_applied[e.box_body, 0] = 0.6; x0 = e.d.qpos[e.box_adr]
        for _ in range(500): mujoco.mj_step(e.m, e.d)
        acc.append(2 * float(e.d.qpos[e.box_adr] - x0) / (500 * e.m.opt.timestep) ** 2)
    assert acc[0] > acc[1] > acc[2] and acc[0] / max(acc[1], 1e-9) > 2.0                                                                    # heavier box accelerates less under the same force
def test_geometry_constants_match_the_compiled_model(env):
    m = env.m; tray = m.body("tray").pos; assert (tray[0], tray[1]) == pytest.approx(Pushing.OBST_RECTS["tray"][:2]); assert m.geom("tray_base_collision").size[0] == pytest.approx(Pushing.OBST_RECTS["tray"][2])
    cube = m.body("green_cube").pos; assert (cube[0], cube[1]) == pytest.approx(Pushing.OBST_RECTS["green_cube"][:2]); assert m.geom("green_cube_collision").size[0] == pytest.approx(Pushing.OBST_RECTS["green_cube"][2])
    cyl = m.body("red_cylinder").pos; assert (cyl[0], cyl[1]) == pytest.approx(Pushing.OBST_CIRCLES["red_cylinder"][:2]); assert m.geom("red_cylinder_collision").size[0] == pytest.approx(Pushing.OBST_CIRCLES["red_cylinder"][2])
def test_goals_valid_reachable_and_success_criterion_unchanged(env):
    t = env.task; rej = 0
    for sd in range(23_500_000, 23_500_300):
        c = t.sample(np.random.default_rng(sd)); b, g = np.array(c["box_xy"]), np.asarray(c["goal"]); ok, why = Pushing.goal_valid(b, g); assert ok, why; rej += c["rejected_samples"] > 0
        x0, x1, y0, y1 = Pushing.WORKSPACE; assert x0 <= g[0] <= x1 and y0 <= g[1] <= y1 and not (abs(g[0] - 0.44) < 0.076 and abs(g[1] - 0.30) < 0.076) and Pushing._clearance(g) >= 0.0         # never inside the tray; box footprint (circumradius + 5 mm) clear of every obstacle
        assert 0.07 - 1e-9 <= np.linalg.norm(g - b) <= 0.13 + 1e-9
    assert rej > 0                                                                                                                         # the rejection path is exercised
    assert not Pushing.goal_valid(np.array([0.44, 0.12]), np.array([0.44, 0.30]))[0] and not Pushing.goal_valid(np.array([0.44, 0.12]), np.array([0.60, 0.12]))[0]
    assert t.p["tol"] == 0.03 and t.p["speed_tol"] == 0.03 and t.p["max_rise"] == 0.01 and t.hold_steps == 5 and t.max_steps == 150                           # strict criterion untouched
    ctx = dict(box_pos=np.array([0.5, 0.1, BOX_REST_Z]), goal=np.array([0.5, 0.1]), box_speed=0.0, box_max_rise=0.0105); assert not t.success_now(ctx)         # a lift > 1 cm is still a failure
def test_deterministic_reset_and_bit_exact_snapshot_replay(env):
    sig = lambda: np.concatenate([env.state_vec(), env.goal]).tobytes(); env.reset(23_400_001); a = sig(); env.reset(23_400_001); b = sig(); env.reset(23_400_002); c = sig(); assert a == b and a != c
    rng = np.random.default_rng(0); acts = rng.uniform(-1, 1, (30, 8)); env.reset(23_400_001); [env.step(env.state_vec()[L.PREV] + x * MAX_STEP) for x in acts[:10]]
    spec = mujoco.mjtState.mjSTATE_INTEGRATION; st = np.empty(mujoco.mj_stateSize(env.m, spec)); mujoco.mj_getState(env.m, env.d, st, spec)
    py = (env.prev_ctrl.copy(), env.t, env.held, env.ever_success, env.box_max_rise, env.tcp_prev.copy()); ra = [env.step(env.state_vec()[L.PREV] + x * MAX_STEP)["state"].copy() for x in acts[10:]]
    mujoco.mj_setState(env.m, env.d, st, spec); env.prev_ctrl, env.t, env.held, env.ever_success, env.box_max_rise, env.tcp_prev = py[0].copy(), py[1], py[2], py[3], py[4], py[5].copy(); mujoco.mj_forward(env.m, env.d)
    rb = [env.step(env.state_vec()[L.PREV] + x * MAX_STEP)["state"].copy() for x in acts[10:]]; assert np.array_equal(np.array(ra), np.array(rb))
def test_executed_action_alignment_and_action_safety(env):
    rng = np.random.default_rng(1); env.reset(23_400_003); lim = env.adapter.limits; right0 = env.hold_right.copy(); prev = env.state_vec()[L.PREV].copy()
    for _ in range(120):
        tgt = prev + rng.uniform(-3, 3, 8) * MAX_STEP * 4.0                                                                                # deliberately over-limit commands
        o = env.step(tgt); ex = o["executed"]; s = o["state"]
        assert np.all(ex >= lim.lo[:8] - 1e-6) and np.all(ex <= lim.hi[:8] + 1e-6)                                                         # clip
        assert np.all(np.abs(ex - prev) <= MAX_STEP * 1.0000001 + 1e-6)                                                                    # slew limit
        assert np.allclose(s[L.PREV], ex, atol=1e-6)                                                                                       # state[t+1].prev_ctrl == executed action t
        assert np.array_equal(env.d.ctrl[8:], right0); prev = ex.copy()                                                                    # right arm held
def test_reward_correctness_against_independent_numpy_formula(env):
    rng = np.random.default_rng(2); env.reset(23_400_004)
    for _ in range(40):
        o = env.step(env.state_vec()[L.PREV] + rng.uniform(-1, 1, 8) * MAX_STEP); s, tcp, g = o["state"], o["tcp"], o["goal"]; bp = s[L.BOX_POS]; d = np.linalg.norm(bp[:2] - g); dt = np.linalg.norm(tcp - bp)
        ref = -3 * d - 0.5 * dt + 2 * np.exp(-(d / 0.04) ** 2) - 0.1 * max(0.0, bp[2] - BOX_REST_Z - 0.01) / 0.01; assert o["reward"] == pytest.approx(ref, abs=1e-4)
_COMPAT = '''
import sys, hashlib, numpy as np
from research.supermimic_openarm.envs.openarm_env import OpenArmTaskEnv
h = hashlib.sha256()
for task in ("reaching", "pick_place"):
    e = OpenArmTaskEnv(task, None, 0, 20)
    for sd in (3, 17):
        rng = np.random.default_rng(sd); s, g, t = e.reset(sd); h.update(np.concatenate([s, g, t]).tobytes())
        for _ in range(60):
            o = e.step(e.state_vec()[16:24].astype(np.float64) + rng.uniform(-1, 1, 8) * 0.05); h.update(np.concatenate([o["state"], [o["reward"]], o["tcp"]]).astype(np.float64).tobytes())
print(h.hexdigest())
'''
def test_p0_compat_reaching_and_pick_place_bit_identical_to_git_head():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); tar = subprocess.run(["git", "archive", "HEAD", "research/supermimic_openarm"], cwd=REPO, capture_output=True, check=True).stdout; subprocess.run(["tar", "-x", "-C", str(td)], input=tar, check=True)
        for f in os.listdir(REPO):
            if f not in ("research", "outputs", ".git") and not f.startswith(".venv"): os.symlink(REPO / f, td / f)
        script = td / "compat.py"; script.write_text(_COMPAT); h = {}
        for name, root in (("head", td), ("work", REPO)):
            r = subprocess.run([sys.executable, str(script)], cwd=root, env={**os.environ, "PYTHONPATH": str(root), "MUJOCO_GL": "egl"}, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-800:]; h[name] = r.stdout.strip().splitlines()[-1]
        assert h["head"] == h["work"], h
