#!/usr/bin/env python3
"""STAGE 0.1 CLIP / SLEW AUDIT (inference-only, NO closed loop, NO training, nothing executed on a policy's behalf).
The Stage-0 JSON stores only summaries, so the raw candidate chunks are regenerated with the SAME deterministic code path (12 decision points, 6 frozen xsweep checkpoints, WM not needed) and saved to outputs/closed_loop_design/stage0_1_chunks.npz;
all analysis below is then pure numpy on those arrays + MJCF actuator metadata.  Hard cap 0.04 process CPU-h; on the cap: PARTIAL, never raised.
Run (ACT venv): python eval/act_clip_audit.py"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import sys, json, time, resource, tempfile, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "eval"), str(ROOT / "data")): sys.path.insert(0, p)
torch.set_num_threads(1)
OUTD = ROOT / "outputs/closed_loop_design"; OUTD.mkdir(parents=True, exist_ok=True)
def cpu(): a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN); return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime
C0_ = cpu(); CAP = 0.04 * 3600
class CapReached(Exception): pass
def guard(tag, margin=0.0):
    if cpu() - C0_ + margin > CAP: raise CapReached(tag)
CK = {"c0_s0": "e6b_c0_seed0", "c0_s1": "e0_c0_seed1", "c0_s2": "e0_c0_seed2", "s1_s0": "e6b_s1_seed0", "s1_s1": "e0_s1_seed1", "s1_s2": "e0_s1_seed2"}
POSES = [(0.35, 0.10, 0.0), (0.40, 0.10, 0.0), (0.45, 0.10, 0.0)]
PHASE_GROUP = {"pre_a": "approach", "pregrasp": "approach", "approach": "approach", "close": "grasp", "grip_hold": "grasp", "lift": "lift", "transport": "lift", "lower": "place", "release": "place"}
res = dict(protocol="stage0.1-v1", status="RUNNING")
try:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import OBS_IMAGES
    class ChunkPolicy:
        def __init__(self, name):
            path = ROOT / "outputs/checkpoints" / CK[name] / "last"; self.policy = ACTPolicy.from_pretrained(str(path)); self.policy.eval()
            self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=str(path)); self.cams = [k.split(".")[-1] for k in self.policy.config.image_features]
        @torch.no_grad()
        def chunk(self, state, images):
            batch = {"observation.state": torch.as_tensor(state, dtype=torch.float32)[None]}
            for c in self.cams: batch[f"observation.images.{c}"] = torch.as_tensor(images[c]).permute(2, 0, 1).float().div(255.0)[None]
            b = dict(self.pre(batch)); b[OBS_IMAGES] = [b[k] for k in self.policy.config.image_features]
            self.policy.model.eval(); a = self.policy.model(b)[0]
            return torch.stack([torch.as_tensor(self.post(a[:, k]).reshape(-1)) for k in range(a.shape[1])]).cpu().numpy()
    policies = {}
    for k in CK: guard(f"load {k}"); policies[k] = ChunkPolicy(k)
    import mujoco
    from lab_env import LabEnv
    from cameras import POLICY_CAMERAS
    from policies.baseline import ScriptedExpert
    from control.safety_limits import SafetyLimits
    env = LabEnv(cameras=POLICY_CAMERAS); lim = SafetyLimits.from_model(env.m); expert = ScriptedExpert(env.m, control_hz=20)
    names_act = [env.m.actuator(i).name for i in range(env.m.nu)]
    RAW, PREV, STATE, META = [], [], [], []
    for (x, y, yaw) in POSES:
        guard(f"pose {x}", margin=10.0)
        plan = expert.plan((x, y), yaw); A = np.array(plan["ctrl"], float); Ex = np.zeros_like(A)
        env.set_perturbation(); env.reset((x, y), yaw); prev = env.prev_ctrl.copy()
        for t in range(len(A)): Ex[t], _, _ = lim.clip_and_slew(A[t], prev, 0.05); prev = Ex[t]
        nm = np.array(plan["phase_names"])[np.array(plan["phase"])]; times = {}
        for g in ("approach", "grasp", "lift", "place"):
            idxs = [t for t in range(4, len(A) - 21) if PHASE_GROUP.get(str(nm[t])) == g]
            if idxs: times[g] = int(idxs[len(idxs) // 2])
        env.set_perturbation(); env.reset((x, y), yaw); obs_at = {}
        for t in range(max(times.values()) + 1):
            if t in times.values(): obs_at[t] = env.obs()
            env.d.ctrl[:] = Ex[t]; env.prev_ctrl = Ex[t].copy()
            for _ in range(env.substeps): mujoco.mj_step(env.m, env.d)
            env.t += 1.0 / env.hz
        for g, t in times.items():
            guard(f"point {x} {g}", margin=5.0); o = obs_at[t]
            RAW.append(np.stack([policies[k].chunk(o["state"], o["images"]) for k in CK])); PREV.append(Ex[t - 1]); STATE.append(o["state"]); META.append(dict(pose=[x, y, yaw], t=t, phase_group=g, phase_name=str(nm[t])))
    RAW = np.array(RAW); PREV = np.array(PREV); STATE = np.array(STATE)
    np.savez(OUTD / "stage0_1_chunks.npz", raw=RAW, prev_exec=PREV, state=STATE, lo=lim.lo, hi=lim.hi, max_speed=lim.max_speed)
    json.dump(dict(meta=META, ckpts=list(CK), actuators=names_act), open(OUTD / "stage0_1_chunks_meta.json", "w"), indent=1)
    res["status"] = "CHUNKS_SAVED"; res["n_points"] = len(META)
except CapReached as e:
    res["status"] = "PARTIAL"; res["stop_reason"] = f"cap at {e}"
except Exception as e:
    import traceback; res["status"] = "BLOCKED"; res["stop_reason"] = f"{type(e).__name__}: {e}"; res["traceback"] = traceback.format_exc()
finally:
    res["cpu_s"] = cpu() - C0_; (OUTD / "stage0_1_run.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1), flush=True)
