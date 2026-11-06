"""Lazy, read-only MuJoCo scene reconstruction (MODE A frames and MODE B scene view).  Uses the repository's own `lab_scene` model and `CameraRig`.
Only `mj_forward` (kinematics) is ever called: there is no `mj_step`, no controller, no policy and no rollout.  The model is created on the first rendering request (about 12 CPU-s), never at start-up."""
import os, sys, io, threading, collections
from . import security as S
os.environ.setdefault("MUJOCO_GL", "osmesa")
class RenderUnavailable(Exception): pass
W, H = 480, 360
CAMERAS = ("overview", "front", "wrist_left", "wrist_right")
class SceneRenderer:
    def __init__(self):
        self.lock = threading.RLock(); self.ready = False; self.error = None; self.cache = collections.OrderedDict(); self.cache_max = 400; self._m = None
    def _init(self):
        if self.ready: return
        if self.error: raise RenderUnavailable(self.error)
        if os.environ.get("OPENARM_STUDIO_NO_RENDER") == "1": self.error = "rendering disabled by OPENARM_STUDIO_NO_RENDER=1"; raise RenderUnavailable(self.error)
        try:
            sim = str(S.REPO / "sim")
            for p in (sim, str(S.REPO)):
                if p not in sys.path: sys.path.insert(0, p)
            import numpy as np, mujoco, lab_scene
            from cameras import CameraRig
            from control.action_adapter import ActionAdapter
            m, d = lab_scene.compile_lab(); m.light_castshadow[:] = 0; m.mat_reflectance[:] = 0.0; m.vis.quality.offsamples = 0          # same 'fast' profile as LabEnv
            self.np, self.mj, self.lab_scene = np, mujoco, lab_scene; self.m, self.d = m, d; self.adapter = ActionAdapter(m, 20); self.rig = CameraRig(m, W, H); self.free = mujoco.Renderer(m, H, W)
            self.box_adr = int(m.jnt_qposadr[m.joint("blue_box_free").id]); self.tcp_id = m.body("openarm_left_hand_tcp").id
            self.finger2 = {s: int(m.jnt_qposadr[m.joint(f"openarm_{s}_finger_joint2").id]) for s in ("left", "right")}; self.ready = True
        except Exception as e:
            self.error = f"MuJoCo rendering unavailable: {type(e).__name__}: {e}"; raise RenderUnavailable(self.error)
    def status(self): return dict(initialised=self.ready, error=self.error, backend=os.environ.get("MUJOCO_GL"), size=[W, H])
    def _set(self, state, box_pos, box_quat):
        np, mj, m, d = self.np, self.mj, self.m, self.d
        mj.mj_resetData(m, d); d.qpos[self.adapter.qadr] = np.asarray(state, float)
        d.qpos[self.finger2["left"]] = state[7]; d.qpos[self.finger2["right"]] = state[15]                      # finger_joint2 is tied to joint1 by an equality constraint; set it explicitly (no dynamics are run)
        if box_pos is not None: d.qpos[self.box_adr:self.box_adr + 3] = box_pos; d.qpos[self.box_adr + 3:self.box_adr + 7] = box_quat if box_quat is not None else [1, 0, 0, 0]
        mj.mj_forward(m, d)
    def _jpeg(self, img, quality=82):
        from PIL import Image
        b = io.BytesIO(); Image.fromarray(img).save(b, "JPEG", quality=quality); return b.getvalue()
    def render(self, state, box_pos, box_quat, view="overview", az=140.0, el=-25.0, dist=1.9, key=None):
        ck = (key, view, round(az), round(el), round(dist, 2)) if key is not None else None
        with self.lock:
            if ck is not None and ck in self.cache: self.cache.move_to_end(ck); return self.cache[ck]
            self._init(); self._set(state, box_pos, box_quat)
            if view == "free":
                cam = self.mj.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = float(az), float(el), float(dist); cam.lookat[:] = [0.38, 0.02, 0.40]; self.free.update_scene(self.d, camera=cam); img = self.free.render()
            elif view in CAMERAS: img = self.rig.rgb(self.d, view)
            else: raise RenderUnavailable(f"unknown view: {view}")
            jpg = self._jpeg(img)
            if ck is not None:
                self.cache[ck] = jpg
                while len(self.cache) > self.cache_max: self.cache.popitem(last=False)
            return jpg
    def home_state(self):
        with self.lock:
            self._init(); self.lab_scene.reset_home(self.m, self.d); self.mj.mj_forward(self.m, self.d); st = self.d.qpos[self.adapter.qadr].copy(); box = self.d.qpos[self.box_adr:self.box_adr + 3].copy(); q = self.d.qpos[self.box_adr + 3:self.box_adr + 7].copy(); return st, box, q
    def tcp_fk(self, state):
        with self.lock:
            self._init(); self._set(state, None, None); R = self.d.xmat[self.tcp_id].reshape(3, 3).copy(); q = self.np.zeros(4); self.mj.mju_mat2Quat(q, R.ravel()); return self.d.xpos[self.tcp_id].copy().tolist(), q.tolist()
RENDERER = SceneRenderer()
