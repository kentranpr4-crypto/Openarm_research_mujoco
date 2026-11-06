"""Episode discovery and normalised, lazily loaded telemetry for the recorded replay mode (MODE A).  Only existing saved files are used; nothing is simulated or fabricated.
Alignment convention shared by all sources: row t of the state is the observation BEFORE action t; action t is executed after it and first appears in row t+1."""
import json, os, re
from functools import lru_cache
from pathlib import Path
import numpy as np
from . import security as S
DT = 0.05; CHUNK = 20
DATASETS_DIR = "outputs/datasets"; PILOT_DIR = "outputs/closed_loop_pilot/results"; WM1_DIR = "outputs/wm1/pilot/episodes"
JOINTS = ["openarm_left_joint%d" % i for i in range(1, 8)] + ["openarm_left_finger_joint1"] + ["openarm_right_joint%d" % i for i in range(1, 8)] + ["openarm_right_finger_joint1"]
SOURCE_INFO = {
    "dataset": dict(label="Scripted-expert demonstration (saved dataset episode)", has_policy=False, definition="pick = box rise >= 0.05 m while near the TCP; place = box within 0.03 m of the tray centre, at rest height and speed < 0.02 m/s (dataset-generation definition)"),
    "pilot": dict(label="Stage-1 pilot rollout (ACT checkpoint selection experiment, privileged-state observer)", has_policy=True, definition="Stage-1 pilot outcome definitions (pick / place / task) from outputs/closed_loop_pilot"),
    "wm1": dict(label="WM-1 pilot episode (saved state/contact log)", has_policy=False, definition="no outcome label stored in the file")}
def _dataset_dirs():
    base = S.resolve(DATASETS_DIR)
    if not base.is_dir(): return []
    return sorted(d.name for d in base.iterdir() if d.is_dir() and (d / "episodes.jsonl").is_file() and any(d.glob("episode_*.npz")))
@lru_cache(maxsize=1)
def _pilot_meta():
    try: rows = S.read_json(PILOT_DIR + "/pilot_results.json")["episodes"]
    except S.ArtifactUnavailable: return {}
    return {(r["arm"], r["pose_idx"], r["repeat"]): r for r in rows}
def _dataset_meta(name):
    try: p = S.resolve(f"{DATASETS_DIR}/{name}/episodes.jsonl")
    except S.ArtifactUnavailable: return {}
    out = {}
    for line in p.read_text().splitlines():
        try: r = json.loads(line); out[int(r["episode_index"])] = r
        except Exception: continue
    return out
def list_episodes():
    items = []
    for name in _dataset_dirs():
        meta = _dataset_meta(name)
        for p in sorted(S.resolve(f"{DATASETS_DIR}/{name}").glob("episode_*.npz")):
            m = re.match(r"episode_(\d+)\.npz$", p.name)
            if not m: continue
            idx = int(m.group(1)); r = meta.get(idx, {})
            items.append(dict(id=f"ds.{name}.{idx}", source="dataset", group=name, title=f"{name} #{idx}", n_frames=r.get("length"), box_xy=r.get("box_xy"), outcome=("task_success" if r.get("task_success") else "task_fail") if "task_success" in r else None, has_images=True, has_contacts=False))
    pm = _pilot_meta(); pdir = S.resolve(PILOT_DIR + "/traj")
    if pdir.is_dir():
        for p in sorted(pdir.glob("*.npz")):
            m = re.match(r"(A[012])_p(\d+)_r(\d+)\.npz$", p.name)
            if not m: continue
            arm, pose, rep = m.group(1), int(m.group(2)), int(m.group(3)); r = pm.get((arm, pose, rep), {})
            items.append(dict(id=f"pilot.{arm}_p{pose:02d}_r{rep}", source="pilot", group=arm, title=f"Stage 1 {arm} pose {pose} repeat {rep}", n_frames=200, box_xy=[r.get("x"), r.get("y")] if r else None, outcome=r.get("outcome"), has_images=False, has_contacts=True))
    wdir = S.resolve(WM1_DIR)
    if wdir.is_dir():
        for p in sorted(wdir.glob("*.npz")):
            items.append(dict(id=f"wm1.{p.stem}", source="wm1", group="wm1", title=f"WM-1 {p.stem}", n_frames=None, box_xy=None, outcome=None, has_images=False, has_contacts=True))
    return items
def _file_for(eid):
    S.safe_id(eid); kind, _, rest = eid.partition(".")
    if kind == "ds":
        name, _, idx = rest.rpartition(".")
        if name not in _dataset_dirs() or not idx.isdigit(): raise S.ArtifactUnavailable(f"unknown dataset episode: {eid}")
        return "dataset", S.resolve(f"{DATASETS_DIR}/{name}/episode_{int(idx):06d}.npz"), (name, int(idx))
    if kind == "pilot":
        m = re.match(r"(A[012])_p(\d+)_r(\d+)$", rest)
        if not m: raise S.ArtifactUnavailable(f"unknown pilot episode: {eid}")
        return "pilot", S.resolve(f"{PILOT_DIR}/traj/{rest}.npz"), (m.group(1), int(m.group(2)), int(m.group(3)))
    if kind == "wm1":
        if not re.match(r"^[A-Za-z0-9_\-]+$", rest): raise S.ArtifactUnavailable(f"unknown WM-1 episode: {eid}")
        return "wm1", S.resolve(f"{WM1_DIR}/{rest}.npz"), (rest,)
    raise S.ArtifactUnavailable(f"unknown episode id: {eid}")
@lru_cache(maxsize=6)
def _load(path_str):
    """lazy: opens the npz, copies only small arrays, keeps the (large) image block as a lazily decoded handle"""
    try: z = np.load(path_str, allow_pickle=False)
    except Exception as e: raise S.ArtifactUnavailable(f"episode file unreadable or corrupted ({type(e).__name__})")
    return z
def _arr(z, key, ncols=None):
    if key not in z.files: return None
    try: a = np.asarray(z[key])
    except Exception as e: raise S.ArtifactUnavailable(f"array {key} unreadable ({type(e).__name__})")
    return a
def episode_meta(eid):
    kind, path, key = _file_for(eid)
    if not path.is_file(): raise S.ArtifactUnavailable(f"episode file missing: {eid}")
    info = dict(SOURCE_INFO[kind]); info.update(id=eid, source=kind, dt_s=DT, chunk_size=CHUNK if kind == "pilot" else None, alignment="state[t] is the observation before action[t]; action[t] first affects state[t+1]", file_bytes=path.stat().st_size)
    if kind == "dataset": r = _dataset_meta(key[0]).get(key[1], {}); info["recorded_metadata"] = {k: r.get(k) for k in ("seed", "region", "box_xy", "box_yaw", "light_scale", "length", "pick_success", "place_success", "task_success", "max_rise_m", "final_box_to_tray_xy_m") if k in r}
    if kind == "pilot": r = _pilot_meta().get(key, {}); info["recorded_metadata"] = {k: r.get(k) for k in ("arm", "pose_idx", "repeat", "x", "y", "yaw", "outcome", "pick_success", "place_success", "task_success", "palm_contact", "max_rise_m", "final_box_to_tray_xy_m") if k in r}
    return info
def derive_events(gripper_cmd, box_z, rise_thr=0.03, close_thr=0.040, open_rise=0.004):
    """VIEWER HEURISTIC on recorded signals (NOT the G2b labeller): first close command, first 3 cm box rise, first re-opening of the commanded gripper after the lift"""
    ev = []
    if gripper_cmd is not None and len(gripper_cmd):
        i = np.where(gripper_cmd < close_thr)[0]
        if len(i): ev.append(dict(t=int(i[0]), kind="close_command", note="commanded gripper < 0.040 m"))
    if box_z is not None and len(box_z):
        r = box_z - box_z[0]; j = np.where(r >= rise_thr)[0]
        if len(j):
            ev.append(dict(t=int(j[0]), kind="lift", note="box rise >= 0.03 m"))
            if gripper_cmd is not None:
                g = np.asarray(gripper_cmd)[int(j[0]):]; run = np.minimum.accumulate(g); k = np.where(g - run >= open_rise)[0]
                if len(k): ev.append(dict(t=int(j[0] + k[0]), kind="release", note="commanded gripper rises >= 4 mm over its minimum after the lift"))
    return ev
def telemetry(eid):
    kind, path, key = _file_for(eid)
    if not path.is_file(): raise S.ArtifactUnavailable(f"episode file missing: {eid}")
    z = _load(str(path)); state = _arr(z, "observation.state") if kind == "dataset" else _arr(z, "state")
    if state is None or state.ndim != 2 or state.shape[1] != 16: raise S.ArtifactUnavailable("episode file has no valid 16-D joint state; exact replay is not possible")
    n = len(state); action = _arr(z, "action"); lim = []
    if kind == "dataset": box, tcp, quat, yaw = _arr(z, "eval.box_pos"), _arr(z, "eval.tcp_pos"), _arr(z, "eval.box_quat"), None
    elif kind == "pilot": box, tcp, quat, yaw = _arr(z, "box_pos"), _arr(z, "tcp_pos"), None, None
    else: box, tcp, quat, yaw = _arr(z, "box_pos"), _arr(z, "tcp_pos"), None, _arr(z, "box_yaw")
    for nm, a in (("box_pos", box), ("tcp_pos", tcp)):
        if a is None: lim.append(f"{nm} not recorded")
    if box is not None and len(box) < n: raise S.ArtifactUnavailable("box_pos shorter than the state: file inconsistent")
    if quat is None and yaw is None: lim.append("box orientation not recorded: the box is drawn unrotated in the MuJoCo reconstruction")
    contacts = None
    if kind == "wm1": contacts = {k: _arr(z, k) for k in ("finger_force", "finger_contacts", "hand_force", "table_force", "tray_force")}
    if kind == "pilot":
        c = _arr(z, "c_rows")
        if c is not None and c.ndim == 2 and c.shape[1] >= 5: contacts = dict(finger_force=c[:, 0] * 50.0, finger_contacts=c[:, 1], hand_force=c[:, 4]); lim.append("contact columns follow the documented scaling used in eval/test_g2b_trace_contacts.py (finger force = c0 x 50 N, contacts = c1, palm = c4); per-finger forces are not recorded")
    if contacts is None: lim.append("contact forces not recorded in this file")
    n_ok = min(n, len(box)) if box is not None else n
    if kind == "pilot": n_ok = min(n_ok, 200)
    gcmd = action[:, 7] if action is not None and action.ndim == 2 and action.shape[1] == 16 else None
    out = dict(id=eid, source=kind, n_frames=int(n_ok), dt_s=DT, t=[round(i * DT, 4) for i in range(n_ok)], joint_names=JOINTS, state=np.round(state[:n_ok], 5).tolist(), gripper_left_m=np.round(state[:n_ok, 7], 5).tolist(), gripper_right_m=np.round(state[:n_ok, 15], 5).tolist(),
               action=(np.round(action[:n_ok], 5).tolist() if action is not None else None), n_actions=int(len(action)) if action is not None else 0, tcp_pos=(np.round(tcp[:n_ok], 5).tolist() if tcp is not None else None), box_pos=(np.round(box[:n_ok], 5).tolist() if box is not None else None),
               box_quat=(np.round(quat[:n_ok], 5).tolist() if quat is not None else None), box_yaw=(np.round(yaw[:n_ok], 5).tolist() if yaw is not None else None),
               contacts=({k: (np.round(v[:n_ok], 4).tolist() if v is not None else None) for k, v in contacts.items()} if contacts else None), has_images=bool(kind == "dataset" and "observation.images.front" in z.files), cameras=(["front", "wrist_left", "wrist_right"] if kind == "dataset" else []),
               chunk_boundaries=(list(range(0, n_ok, CHUNK)) if kind == "pilot" else []), model_predictions=None, model_predictions_note="no model prediction is stored in this file", limitations=lim)
    phases = None
    if kind == "dataset" and "phase" in z.files:
        names = _dataset_meta(key[0]).get(key[1], {}).get("phase_names"); ph = np.asarray(z["phase"])[:n_ok]
        if names and len(ph):
            phases = []; start = 0
            for i in range(1, len(ph) + 1):
                if i == len(ph) or ph[i] != ph[start]: idx = int(ph[start]); phases.append(dict(name=names[idx] if 0 <= idx < len(names) else str(idx), start=int(start), end=int(i - 1))); start = i
    out["phases"] = phases; out["phases_note"] = "RECORDED by the scripted-expert generator (phase array in the dataset file)" if phases else None
    out["events"] = derive_events(gcmd[:n_ok] if gcmd is not None else None, box[:n_ok, 2] if box is not None else None); out["events_note"] = "DERIVED by the viewer from recorded signals (heuristic); not an outcome label and not the G2b labeller"
    out["outcome"] = episode_meta(eid).get("recorded_metadata"); return out
def recorded_frame(eid, k, cam):
    kind, path, key = _file_for(eid)
    if kind != "dataset": raise S.ArtifactUnavailable("this episode has no stored camera images")
    if cam not in ("front", "wrist_left", "wrist_right"): raise S.ArtifactUnavailable(f"unknown camera: {cam}")
    z = _load(str(path)); name = f"observation.images.{cam}"
    if name not in z.files: raise S.ArtifactUnavailable(f"camera {cam} not stored in this file")
    imgs = z[name]
    if not (0 <= k < len(imgs)): raise S.ArtifactUnavailable(f"frame index {k} out of range 0..{len(imgs) - 1}")
    return np.asarray(imgs[k])
def render_inputs(eid, k):
    """arrays needed to reconstruct frame k in MuJoCo (state, box pose); explicit errors when the file cannot support it"""
    kind, path, key = _file_for(eid); z = _load(str(path)); state = _arr(z, "observation.state") if kind == "dataset" else _arr(z, "state")
    if state is None: raise S.ArtifactUnavailable("no joint state in this file")
    n = len(state)
    if kind == "pilot": n = min(n, 200)
    if not (0 <= k < n): raise S.ArtifactUnavailable(f"frame index {k} out of range 0..{n - 1}")
    box = _arr(z, "eval.box_pos" if kind == "dataset" else "box_pos"); quat = _arr(z, "eval.box_quat") if kind == "dataset" else None; yaw = _arr(z, "box_yaw") if kind == "wm1" else None
    if box is None: raise S.ArtifactUnavailable("no box pose in this file: scene reconstruction impossible")
    q = quat[k] if quat is not None else ([float(np.cos(yaw[k] / 2)), 0.0, 0.0, float(np.sin(yaw[k] / 2))] if yaw is not None else [1.0, 0.0, 0.0, 0.0])
    return np.asarray(state[k], float), np.asarray(box[k], float), np.asarray(q, float)
