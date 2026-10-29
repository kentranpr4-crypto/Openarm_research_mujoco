"""Normalisation contract for the OpenArm ACT policies (E6-B, stage 0-1). Run inside the ACT venv (needs numpy, pyarrow, lerobot).

Contract (per dimension d of `observation.state` and `action`; images keep per-channel MEAN_STD):
    sigma_eff[d] = max(sigma_train[d], floor[d])            # floor[d] = 0 means "no floor" (V1 behaviour)
    x_n[d]       = (x[d] - mu_train[d]) / (sigma_eff[d] + EPS)        EPS = 1e-8  (LeRobot 0.6.1 normalize_processor.py:359)
    x[d]         = x_n[d] * sigma_eff[d] + mu_train[d]                (de-normalisation AS IMPLEMENTED by LeRobot, line 360: NO eps)
ASYMMETRY (found by test TB): forward divides by sigma_eff+EPS, inverse multiplies by sigma_eff. The round trip is the identity only up to a
relative error EPS/sigma_eff: negligible for every floored or live dimension (sigma_eff >= 2e-3), but for an UNFLOORED idle dimension with
sigma << EPS (action dims 8-15 under V1/C0/S1) the inverse collapses any deviation to ~sigma/(sigma+EPS) of its size, i.e. the decoded idle
command is ~mu. That is a property of the library we keep (no patching), documented and tested, not a bug in the contract.
Statistics come from the TRAIN episodes only (aggregated from the per-episode stats LeRobot stores in meta/episodes), or - for the
legacy V1 behaviour - from the dataset-wide stats.json (`source="dataset"`).
LeRobot's processors store the (already floored) std/mean inside the checkpoint, so inference uses exactly the training numbers;
`norm_contract.json` next to the weights records provenance and lets `verify_checkpoint` fail loud on any mismatch.
Floors are HYPOTHESES to test (0.02 rad, 0.002 m), not hardware specifications."""
import json, hashlib, subprocess
from pathlib import Path
import numpy as np

EPS = 1e-8
STATE, ACTION = "observation.state", "action"
GRIPPER_IDX = (7, 15)                       # left / right finger slide (metres, v1); every other dim is a joint angle (rad)
EXPECTED_BINDING = tuple(range(8, 16))      # right arm + right gripper: the idle arm is the only thing a floor may bind on
CONTRACT_FILE = "norm_contract.json"
ROOT = Path(__file__).resolve().parent.parent

class NormContractError(RuntimeError):
    pass

def floors_vector(arm_floor, gripper_floor, n=16):
    """(n,) per-dimension floor in the dimension's own unit (rad for joints, m for grippers)."""
    f = np.full(n, float(arm_floor)); f[list(GRIPPER_IDX)] = float(gripper_floor); return f

# ---------------------------------------------------------------- statistics
def _episode_stats_table(dataset_root):
    import pyarrow.parquet as pq
    files = sorted(Path(dataset_root).glob("meta/episodes/**/*.parquet")); rows = []
    for f in files: rows += pq.read_table(f).to_pylist()
    rows.sort(key=lambda r: r["episode_index"]); return rows

def episode_stats(dataset_root):
    """list (indexed by LeRobot episode index) of {feature: {stat: np.ndarray}} read from meta/episodes/*.parquet."""
    out = []
    for r in _episode_stats_table(dataset_root):
        d = {}
        for k, v in r.items():
            if k.startswith("stats/"):
                _, feat, stat = k.split("/", 2); d.setdefault(feat, {})[stat] = np.asarray(v)
        out.append(d)
    return out

def aggregate(dataset_root, episodes, features=None):
    """Weighted aggregation (LeRobot's own aggregate_stats) over the given LeRobot episode indices."""
    from lerobot.datasets.compute_stats import aggregate_stats
    es = episode_stats(dataset_root); chosen = [es[i] for i in episodes]
    if features is not None: chosen = [{k: v for k, v in e.items() if k in features} for e in chosen]
    return aggregate_stats(chosen)

def dataset_wide_stats(dataset_root):
    """What V1 used: LeRobotDatasetMetadata.stats (stats.json, ALL episodes, train+val)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    return LeRobotDatasetMetadata("local/openarm_lab_pickplace", root=dataset_root).stats

def frame_count(stats):
    return int(np.asarray(stats[ACTION]["count"]).ravel()[0])

# ---------------------------------------------------------------- floors
def apply_floor(stats, state_floor=None, action_floor=None, expected_binding=EXPECTED_BINDING):
    """Return a COPY of `stats` whose state/action std is max(std, floor). Fail loud if a floor binds on a dimension outside
    `expected_binding` (a live dimension would be silently re-scaled) or if the shapes are wrong."""
    out = {k: {s: np.array(v, copy=True) for s, v in d.items()} for k, d in stats.items()}; binding = {}
    for key, floor in ((STATE, state_floor), (ACTION, action_floor)):
        if floor is None: continue
        floor = np.asarray(floor, float); std = out[key]["std"].astype(float)
        if floor.shape != std.shape: raise NormContractError(f"{key}: floor shape {floor.shape} != std shape {std.shape}")
        bound = [int(i) for i in np.nonzero(std < floor)[0]]; extra = sorted(set(bound) - set(expected_binding))
        if extra: raise NormContractError(f"{key}: floor would re-scale live dimension(s) {extra} (std {std[extra]} < floor {floor[extra]})")
        out[key]["std"] = np.maximum(std, floor).astype(out[key]["std"].dtype); binding[key] = bound
    return out, binding

def digest(stats):
    h = hashlib.sha256()
    for key in sorted(stats):
        for s in ("mean", "std"):
            if s in stats[key]: h.update(key.encode() + s.encode() + np.ascontiguousarray(np.asarray(stats[key][s], np.float64)).tobytes())
    return h.hexdigest()

# ---------------------------------------------------------------- normalise / de-normalise (reference implementation used by tests)
def normalize(x, stats, key):
    return (np.asarray(x, np.float64) - stats[key]["mean"]) / (np.asarray(stats[key]["std"], np.float64) + EPS)

def denormalize(xn, stats, key):
    """LeRobot 0.6.1 inverse: x_n * sigma + mu (no EPS), see module docstring."""
    return np.asarray(xn, np.float64) * np.asarray(stats[key]["std"], np.float64) + stats[key]["mean"]

# ---------------------------------------------------------------- contract file
def build_contract(source, train_episodes, stats_train, stats_final, state_floor, action_floor, binding, dataset_root):
    sha = lambda: subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return dict(version=1, source=source, dataset=str(dataset_root), train_episodes=[int(i) for i in train_episodes], n_frames=frame_count(stats_final),
                eps=EPS, state_floor=None if state_floor is None else np.asarray(state_floor).tolist(), action_floor=None if action_floor is None else np.asarray(action_floor).tolist(),
                binding=binding, expected_binding=list(EXPECTED_BINDING), code_commit=sha(),
                state=dict(mean=stats_final[STATE]["mean"].tolist(), std_train=np.asarray(stats_train[STATE]["std"]).tolist(), std_eff=np.asarray(stats_final[STATE]["std"]).tolist()),
                action=dict(mean=stats_final[ACTION]["mean"].tolist(), std_train=np.asarray(stats_train[ACTION]["std"]).tolist(), std_eff=np.asarray(stats_final[ACTION]["std"]).tolist()),
                stats_sha256=digest(stats_final))

def write_contract(path, contract):
    Path(path).write_text(json.dumps(contract, indent=1))

def read_contract(path):
    c = json.loads(Path(path).read_text())
    if c.get("version") != 1: raise NormContractError(f"unsupported contract version {c.get('version')}")
    return c

def _safetensors_stats(ckpt_dir):
    from safetensors.numpy import load_file
    pre = load_file(str(Path(ckpt_dir) / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
    post = load_file(str(Path(ckpt_dir) / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    return pre, post

def verify_checkpoint(ckpt_dir, rtol=1e-6):
    """None for legacy (V1) checkpoints without a contract. Otherwise raises NormContractError unless the std/mean stored in
    the checkpoint's pre/post processors equal the contract's (state in the preprocessor, action in both)."""
    p = Path(ckpt_dir) / CONTRACT_FILE
    if not p.exists(): return None
    c = read_contract(p); pre, post = _safetensors_stats(ckpt_dir)
    for key, name, tensors in ((STATE, "state", (pre,)), (ACTION, "action", (pre, post))):
        for t in tensors:
            for stat, ref in (("mean", c[name]["mean"]), ("std", c[name]["std_eff"])):
                got = np.asarray(t[f"{key}.{stat}"], np.float64); ref = np.asarray(ref, np.float64)
                if not np.allclose(got, ref, rtol=rtol, atol=1e-12): raise NormContractError(f"{ckpt_dir}: {key}.{stat} in processors differs from {CONTRACT_FILE}")
    return c
