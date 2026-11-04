#!/usr/bin/env python3
"""G2b per-finger instrumentation and branch-trace container (B1).  DRAFT; not used by any experiment yet.
Logs, per control step (row t = observation BEFORE the action of step t0+t): executed gripper target and measured gripper position, per-finger (left/right) contact point counts and summed NORMAL forces [N], palm (hand mesh) force, other-contact force, robot-obstacle penetration,
TCP pose, box pose/velocity, the box pose in the TCP frame, the net contact force on the box (world frame, audit column) and the executed 16-D command.
CONVENTIONS (verified by eval/test_g2b_trace_contacts.py, not assumed): mj_contactForce(m, d, i, f6) returns the contact force in the CONTACT frame (SI units, newtons); f6[0] is the NORMAL component and is >= 0 for these unilateral contacts; the contact frame rows are [normal, tangent1, tangent2] (contact.frame, row-major) with the
normal pointing from geom1 to geom2; the world-frame force exerted on geom2 (by geom1) is frame.T @ f6[:3] and on geom1 is its negative.  Per-finger force = sum over the contact points between the box collision geom and that finger's collision mesh of |f6[0]|.
It is a measured contact quantity - never inferred from the actuator servo law.  `n*` count contact POINTS (mesh manifolds give ~4 per finger), not fingers.
Units: m, m/s, N, steps; dt = 0.05 s; world frame, z up."""
import os, json, hashlib, tempfile, numpy as np
from pathlib import Path
DT = 0.05
GEOMS = dict(box="blue_box_collision", fingerL="openarm_left_left_finger_collision", fingerR="openarm_left_right_finger_collision", hand="openarm_left_hand_collision")
FIELDS = {  # name -> (shape per row, unit, description)
    "g_exec": ((), "m", "executed left-gripper target applied at this step (after clip->slew); the last row repeats the previous value"), "g_meas": ((), "m", "measured left finger joint position (qpos of openarm_left_finger_joint1)"),
    "nL": ((), "count", "contact POINTS box <-> left-finger mesh"), "nR": ((), "count", "contact POINTS box <-> right-finger mesh"), "nH": ((), "count", "contact POINTS box <-> hand (palm) mesh"), "nO": ((), "count", "contact POINTS box <-> any other geom (table, tray, ...)"),
    "FL": ((), "N", "sum of normal-force magnitudes, left finger"), "FR": ((), "N", "sum of normal-force magnitudes, right finger"), "Fh": ((), "N", "sum of normal-force magnitudes, palm (hand mesh)"), "Fo": ((), "N", "sum of normal-force magnitudes, other contacts"),
    "neg_normal": ((), "count", "number of box contact points with a negative normal component (must be 0)"), "F_on_box_world": ((3,), "N", "net force exerted on the box by all its contacts, world frame (audit column)"), "pen_obst": ((), "m", "max robot-obstacle (desk/tray/room) penetration depth"),
    "tcp_pos": ((3,), "m", "TCP position, world"), "tcp_rot": ((9,), "-", "TCP rotation matrix, row-major, world<-TCP"), "box_pos": ((3,), "m", "box centre position, world"), "box_quat": ((4,), "-", "box orientation quaternion (w,x,y,z)"), "box_vel": ((3,), "m/s", "box linear velocity (free-joint qvel[0:3]), world"),
    "box_in_tcp": ((3,), "m", "box centre in the TCP frame: R_tcp^T (box_pos - tcp_pos)"), "ctrl_exec": ((16,), "rad / m", "executed 16-D absolute actuator targets (left joints 0-6 rad, left finger 7 m, right joints 8-14 rad, right finger 15 m)")}
class ContactMap:
    """geom/body mapping resolved by NAME from the model (fails loud); left-arm fingers/hand only."""
    def __init__(self, m):
        import mujoco; self.mj, self.m = mujoco, m; g = lambda n: m.geom(n).id
        self.box, self.fL, self.fR, self.hand = g(GEOMS["box"]), g(GEOMS["fingerL"]), g(GEOMS["fingerR"]), g(GEOMS["hand"]); self.body = {k: m.body(m.geom_bodyid[v]).name for k, v in (("box", self.box), ("fingerL", self.fL), ("fingerR", self.fR), ("hand", self.hand))}
        assert len({self.fL, self.fR, self.hand, self.box}) == 4 and "left" in self.body["fingerL"] and "left" in self.body["fingerR"] and "left" in self.body["hand"], self.body
        self.finger_qadr = int(m.jnt_qposadr[m.joint("openarm_left_finger_joint1").id]); self.box_dof = int(m.jnt_dofadr[m.joint("blue_box_free").id]); self.box_body = int(m.geom_bodyid[self.box]); self.tcp = m.body("openarm_left_hand_tcp").id
        bn = [m.body(m.geom_bodyid[i]).name for i in range(m.ngeom)]; self.robot_g = np.array([n.startswith("openarm_") for n in bn]); self.obst_g = np.array([n in ("desk", "tray", "room") for n in bn]); assert self.robot_g.sum() > 10 and self.obst_g.sum() >= 3
    def read_contacts(self, d):
        mj = self.mj; f6 = np.zeros(6); r = dict(nL=0, nR=0, nH=0, nO=0, FL=0.0, FR=0.0, Fh=0.0, Fo=0.0, neg_normal=0, F_on_box_world=np.zeros(3), pen_obst=0.0)
        for i in range(d.ncon):
            c = d.contact[i]; g1, g2 = c.geom1, c.geom2
            if (self.robot_g[g1] and self.obst_g[g2]) or (self.robot_g[g2] and self.obst_g[g1]): r["pen_obst"] = max(r["pen_obst"], -float(c.dist))
            if self.box not in (g1, g2): continue
            mj.mj_contactForce(self.m, d, i, f6); fn = float(f6[0]); fw = np.asarray(c.frame, float).reshape(3, 3).T @ f6[:3]; r["F_on_box_world"] += fw if g2 == self.box else -fw; r["neg_normal"] += int(fn < -1e-9); fn = abs(fn); o = g2 if g1 == self.box else g1
            if o == self.fL: r["nL"] += 1; r["FL"] += fn
            elif o == self.fR: r["nR"] += 1; r["FR"] += fn
            elif o == self.hand: r["nH"] += 1; r["Fh"] += fn
            else: r["nO"] += 1; r["Fo"] += fn
        return r
class TraceRecorder:
    def __init__(self, m, meta, refresh=True):
        """refresh=True (r2 default): contacts/forces/xpos/xmat are read from a SCRATCH copy of the data after mj_forward, i.e. they belong to the same instant as qpos/qvel (end of the control step); the live MjData is never modified (mj_forward on the live data could change the solver warm-start).  refresh=False reads the raw post-mj_step arrays (contacts/xpos are from the start of the last substep, 1 ms older than qpos)."""
        import mujoco
        self.cmap = ContactMap(m); self.m = m; self.rows = {k: [] for k in FIELDS}; self.meta = dict(meta); self._last = None; self.refresh = bool(refresh); self._scr = mujoco.MjData(m) if refresh else None
    def record(self, d, g_exec, ctrl_exec):
        c = self.cmap
        if self.refresh:
            self.cmap.mj.mj_copyData(self._scr, self.m, d); self.cmap.mj.mj_forward(self.m, self._scr); src = self._scr
        else: src = d
        r = c.read_contacts(src); tcp = src.xpos[c.tcp].copy(); R = src.xmat[c.tcp].reshape(3, 3).copy(); b = src.xpos[c.box_body].copy(); row = dict(g_exec=float(g_exec), g_meas=float(d.qpos[c.finger_qadr]), box_pos=b, box_quat=src.xquat[c.box_body].copy(), box_vel=d.qvel[c.box_dof:c.box_dof + 3].copy(), tcp_pos=tcp, tcp_rot=R.ravel(), box_in_tcp=R.T @ (b - tcp), ctrl_exec=np.asarray(ctrl_exec, float).copy(), **r)
        for k in FIELDS: self.rows[k].append(row[k])
        self._last = row
    def record_final(self, d):
        """last row of a branch: the observation after the final action; the executed target/command repeat the previous ones"""
        self.record(d, self._last["g_exec"], self._last["ctrl_exec"])
    def arrays(self): return {k: np.asarray(v) for k, v in self.rows.items()}
def to_labeller_trace(arrs, meta):
    """map the recorder arrays onto the schema of eval/g2b_labeller.py"""
    m = dict(meta); m.setdefault("t0", 0); m.setdefault("valid", True); valid = all(np.isfinite(arrs[k]).all() for k in ("g_exec", "nL", "nR", "FL", "FR", "box_pos", "box_vel")) and int(arrs["neg_normal"].sum()) == 0; m["valid"] = bool(m["valid"] and valid)
    return dict(g_exec=arrs["g_exec"], g_meas=arrs["g_meas"], nL=arrs["nL"], nR=arrs["nR"], FL=arrs["FL"], FR=arrs["FR"], box_pos=arrs["box_pos"], box_vel=arrs["box_vel"], Fh=arrs["Fh"], meta=m)       # meta must carry t0, box_z0 and g_prev0 (the gripper target in force before row 0)
def _atomic_write(path, data_bytes):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp"); os.close(fd); Path(tmp).write_bytes(data_bytes); os.replace(tmp, path)
def save_trace(path_base, arrs, meta, label=None, code_files=()):
    """atomic: <base>.npz (arrays) + <base>.json (meta, units, conventions, label with events/margins, code sha256)"""
    import io; base = Path(path_base); buf = io.BytesIO(); np.savez_compressed(buf, **arrs); _atomic_write(base.with_suffix(".npz"), buf.getvalue())
    side = dict(meta=meta, fields={k: dict(shape=list(v[0]), unit=v[1], description=v[2]) for k, v in FIELDS.items()}, dt_s=DT, conventions=__doc__, label=label, code_sha256={str(f): hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in code_files}, n_rows=int(len(arrs["g_exec"])))
    _atomic_write(base.with_suffix(".json"), json.dumps(side, indent=1, default=float).encode())
def load_trace(path_base):
    base = Path(path_base); z = np.load(base.with_suffix(".npz")); return {k: z[k] for k in z.files}, json.loads(base.with_suffix(".json").read_text())
