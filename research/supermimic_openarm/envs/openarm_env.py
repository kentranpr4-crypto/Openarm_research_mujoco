"""Single OpenArm v1 task environment on CPU MuJoCo.  Reuses the repository's lab scene (`sim/lab_scene.py`, built with MjSpec on top of the UNMODIFIED upstream XML)
and the repository's original `ActionAdapter` / `SafetyLimits` for every control step (clip to ctrlrange + slew limit).  No rendering, no cameras, no ACT, no dataset.
Observations returned here are the PRIVILEGED simulator state vector (see layout.py); `deployment_observation()` returns only non-privileged quantities."""
import os, sys
from pathlib import Path
import numpy as np
from . import layout as L
from .tasks import make_task, BOX_REST_Z
from ..utils import REPO
_MODS = {}
def _load_sim():
    if _MODS: return _MODS
    sim = str(REPO / "sim")
    for p in (sim, str(REPO)):
        if p not in sys.path: sys.path.insert(0, p)
    try: import mujoco, lab_scene
    except ImportError as e: raise ImportError(f"cannot import the simulation dependencies ({e}). Run research/supermimic_openarm/scripts/setup_ubuntu.sh") from e
    if not Path(lab_scene.UPSTREAM).is_file(): raise FileNotFoundError(f"upstream OpenArm model missing: {lab_scene.UPSTREAM}. Run research/supermimic_openarm/scripts/setup_ubuntu.sh (clones the pinned upstream openarm_mujoco; it is never modified)")
    from control.action_adapter import ActionAdapter
    _MODS.update(mujoco=mujoco, lab_scene=lab_scene, ActionAdapter=ActionAdapter); return _MODS
class OpenArmTaskEnv:
    def __init__(self, task_name, task_params=None, max_steps=0, control_hz=20):
        M = _load_sim(); self.mj, self.lab = M["mujoco"], M["lab_scene"]; self.task = make_task(task_name, **(task_params or {})); self.max_steps = int(max_steps) or self.task.max_steps
        self.m, self.d = self.lab.compile_lab(with_cameras=False); self.adapter = M["ActionAdapter"](self.m, control_hz); self.hz = control_hz
        self.substeps = int(round(1.0 / (control_hz * self.m.opt.timestep))); mj = self.mj
        self.box_body = self.m.body("blue_box").id; self.box_jnt = self.m.joint("blue_box_free"); self.box_adr = int(self.m.jnt_qposadr[self.box_jnt.id]); self.box_dof = int(self.m.jnt_dofadr[self.box_jnt.id])
        self.tcp_id = self.m.body("openarm_left_hand_tcp").id
        self.box_geoms = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == self.box_body}; self._fric0 = self.m.geom_friction[list(self.box_geoms)].copy(); self._prio0 = self.m.geom_priority[list(self.box_geoms)].copy(); self._mass0 = float(self.m.body_mass[self.box_body]); self._inertia0 = self.m.body_inertia[self.box_body].copy()
        names = ("openarm_left_right_finger", "openarm_left_left_finger"); fb = [self.m.body(n).id for n in names]
        self.finger_geoms = {b: {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == b and self.m.geom_contype[g]} for b in fb}
        hand = {self.m.body(n).id for n in ("openarm_left_hand", "openarm_left_link7")}; self.hand_geoms = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in hand and self.m.geom_contype[g]}
        self.left_idx = np.arange(8); self.hold_right = None; self.t = 0; self.goal = None; self._f6 = np.zeros(6)
    # ------------------------------------------------------------------------------------------------------------------------------------
    def state_vec(self):
        d, A = self.d, self.adapter; s = np.empty(L.STATE_DIM, np.float32)
        s[L.Q] = d.qpos[A.qadr[:8]]; s[L.DQ] = d.qvel[A.vadr[:8]]; s[L.PREV] = self.prev_ctrl[:8]
        s[L.BOX_POS] = d.qpos[self.box_adr:self.box_adr + 3]; s[L.BOX_QUAT] = d.qpos[self.box_adr + 3:self.box_adr + 7]; s[L.BOX_VEL] = d.qvel[self.box_dof:self.box_dof + 3]; return s
    def tcp(self): return self.d.xpos[self.tcp_id].astype(np.float32).copy()
    def _contacts(self):
        d, mj = self.d, self.mj; fforce = 0.0; hforce = 0.0; touching = set()
        for i in range(d.ncon):
            c = d.contact[i]; g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self.box_geoms: o = g2
            elif g2 in self.box_geoms: o = g1
            else: o = None
            if o is None: continue
            mj.mj_contactForce(self.m, d, i, self._f6); fn = abs(float(self._f6[0]))
            for b, gs in self.finger_geoms.items():
                if o in gs: fforce += fn; touching.add(b)
            if o in self.hand_geoms: hforce += fn
        return fforce, len(touching), hforce
    def deployment_observation(self, state, goal, tcp):
        """non-privileged observation dictionary (what a real controller could know without perception)"""
        return dict(q=state[L.Q].copy(), dq=state[L.DQ].copy(), prev_ctrl=state[L.PREV].copy(), tcp_pos=np.asarray(tcp, np.float32).copy(), goal=np.asarray(goal, np.float32).copy())
    # ------------------------------------------------------------------------------------------------------------------------------------
    def reset(self, seed):
        rng = np.random.default_rng(int(seed)); cfg = self.task.sample(rng); self.goal = np.asarray(cfg["goal"], np.float32); mj, m, d = self.mj, self.m, self.d
        gl = list(self.box_geoms); m.geom_friction[gl] = self._fric0 * np.array([cfg["friction_scale"], 1.0, 1.0])
        # pushing only (cfg carries the keys; reaching / pick_place keep the legacy behaviour bit-exactly): MuJoCo combines two geoms' friction with max() when their priorities are equal, so a box friction below the desk's 0.8 had no effect.
        # priority 1 on the box makes the box-desk pair use the box's own friction (finger geoms already have priority 1, so finger-box contacts are unchanged: max(finger 1.0, box)).
        m.geom_priority[gl] = 1 if cfg.get("friction_mode") == "pair_exact" else self._prio0
        if "mass_scale" in cfg:                                                                                 # mass and principal inertia scale together (uniform density, fixed geometry); recomputed from the originals every reset
            m.body_mass[self.box_body] = self._mass0 * cfg["mass_scale"]; m.body_inertia[self.box_body] = self._inertia0 * cfg["mass_scale"]; mj.mj_setConst(m, d)
        self.lab.reset_home(m, d); a = self.box_adr; yaw = cfg["box_yaw"]; d.qpos[a:a + 3] = [cfg["box_xy"][0], cfg["box_xy"][1], BOX_REST_Z]; d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]; d.qvel[:] = 0
        mj.mj_forward(m, d)
        for _ in range(int(0.2 / m.opt.timestep)): mj.mj_step(m, d)
        self.prev_ctrl = d.ctrl.copy(); self.hold_right = self.prev_ctrl[8:].copy(); self.t = 0; self.held = 0; self.ever_success = False; self.box_max_rise = 0.0; self.tcp_prev = self.tcp()
        self.episode_return = 0.0; self.episode_info = dict(seed=int(seed), friction_scale=cfg["friction_scale"], mass_kg=float(m.body_mass[self.box_body]), inertia_diag_kgm2=m.body_inertia[self.box_body].tolist(), mass_scale=cfg.get("mass_scale"), friction_mode=cfg.get("friction_mode"), rejected_samples=cfg.get("rejected_samples"), box_yaw=float(cfg["box_yaw"]), box_xy=list(map(float, cfg["box_xy"])), goal=self.goal.tolist())
        s = self.state_vec(); return s, self.goal.copy(), self.tcp()
    def step(self, target_left):
        """target_left: (8,) ABSOLUTE joint-position target for the left arm+gripper.  The original ActionAdapter clips and slew-limits it."""
        import torch
        tl = np.asarray(target_left, float)
        if tl.shape != (8,): raise ValueError(f"target must have shape (8,), got {tl.shape}")
        full = np.concatenate([tl, self.hold_right]); ctrl, n_rng, n_slew = self.adapter.to_ctrl(full, self.prev_ctrl)        # raises on NaN/Inf
        self.d.ctrl[:] = ctrl; self.prev_ctrl = ctrl.copy()
        for _ in range(self.substeps): self.mj.mj_step(self.m, self.d)
        self.t += 1; finite = bool(np.isfinite(self.d.qpos).all() and np.isfinite(self.d.qvel).all())
        if not finite: s = self.state_vec_safe if hasattr(self, "state_vec_safe") else np.zeros(L.STATE_DIM, np.float32)
        else: s = self.state_vec(); self.state_vec_safe = s
        tcp = self.tcp() if finite else self.tcp_prev; ff, nf, hf = self._contacts() if finite else (0.0, 0, 0.0)
        speed = float(np.linalg.norm(self.d.qvel[self.box_dof:self.box_dof + 3])) if finite else 0.0; tcp_speed = float(np.linalg.norm(tcp - self.tcp_prev) * self.hz); self.tcp_prev = tcp
        self.box_max_rise = max(self.box_max_rise, float(s[L.BOX_POS][2] - BOX_REST_Z))
        ctx = dict(tcp=tcp, goal=self.goal, tcp_speed=tcp_speed, box_pos=s[L.BOX_POS], box_speed=speed, box_max_rise=self.box_max_rise, gripper_open_m=float(s[7]), finger_contacts=nf)
        ok = self.task.success_now(ctx); self.held = self.held + 1 if ok else 0
        if self.held >= self.task.hold_steps: self.ever_success = True
        exec_left = ctrl[:8].astype(np.float32)
        with torch.no_grad(): r = float(self.task.reward_torch(torch.from_numpy(s)[None], torch.from_numpy(exec_left)[None], torch.from_numpy(self.goal)[None], torch.from_numpy(tcp)[None])[0])
        terminated = (not finite) or self.task.failed(ctx); truncated = (not terminated) and self.t >= self.max_steps; self.episode_return += r
        contact = np.array([min(ff, 50.0) / 50.0, nf / 2.0, min(hf, 50.0) / 50.0], np.float32)
        info = dict(success=bool(self.held >= self.task.hold_steps), ever_success=bool(self.ever_success), held=int(self.held), box_max_rise=self.box_max_rise, finger_force_N=ff, finger_contacts=nf, hand_force_N=hf, tcp_speed=tcp_speed, box_speed=speed,
                    n_clipped=int(n_rng), n_slewed=int(n_slew), t=self.t, finite=finite)
        return dict(state=s, reward=r, terminated=bool(terminated), truncated=bool(truncated), goal=self.goal.copy(), tcp=tcp, contact=contact, executed=exec_left, info=info)
    def close(self): pass
