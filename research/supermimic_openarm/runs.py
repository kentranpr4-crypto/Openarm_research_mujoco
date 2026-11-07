"""Run-directory management: never overwrite a completed experiment, refuse silently reusing a non-empty directory, record reproducibility metadata."""
import json
from pathlib import Path
from . import utils
from .config import Config
def run_dir_for(cfg: Config, run_name=None):
    root = Path(cfg.experiment.output_dir); root = root if root.is_absolute() else utils.REPO / root
    return root / (run_name or cfg.experiment.name) / f"seed{cfg.experiment.seed}"
def prepare_run_dir(cfg, run_dir, resume=False):
    d = Path(run_dir)
    if (d / "COMPLETED").is_file(): raise FileExistsError(f"experiment already COMPLETED, refusing to touch it: {d} (choose a new --run-name or seed)")
    if d.exists() and any(d.iterdir()) and not resume: raise FileExistsError(f"run directory exists and is not empty: {d}. Use --resume to continue it or a different --run-name")
    d.mkdir(parents=True, exist_ok=True); return d
def write_run_files(cfg, run_dir, env_manifest=None, resume=False):
    d = Path(run_dir); utils.atomic_write_json(d / ("config.resolved.json" if not resume else "config.resolved.json"), cfg.to_dict())
    md = utils.repro_metadata(dict(experiment=cfg.experiment.name, seed=cfg.experiment.seed, algo=cfg.experiment.algo)); path = d / "metadata.json"
    if resume and path.is_file(): hist = json.loads(path.read_text()); hist.setdefault("resumes", []).append(md); md = hist
    utils.atomic_write_json(path, md)
    if env_manifest is not None: utils.atomic_write_json(d / "env_manifest.json", env_manifest)
def build_env_manifest(cfg, model_info=None):
    from .envs.tasks import make_task; from .envs import layout as L
    t = make_task(cfg.env.task, **cfg.env.task_params); up = utils.REPO / "sim" / "openarm_mujoco"; head = None
    try: head = (up / ".git" / "HEAD").read_text().strip(); 
    except Exception: pass
    sha = None
    try:
        import subprocess; sha = subprocess.run(["git", "-C", str(up), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception: pass
    return dict(task=t.name, observation_spec=t.observation_spec(), action_spec=t.action_spec(), reward_spec=t.reward_spec(), termination_spec=t.termination_spec(), success_spec=t.success_spec(), randomization_spec=t.randomization_spec(), max_steps=t.max_steps,
                state_layout=dict(dim=L.STATE_DIM, q="0:8", dq="8:16", prev_ctrl="16:24", box_pos="24:27", box_quat="27:31", box_vel="31:34"), model=model_info, upstream_openarm_mujoco_commit=sha, privileged_state=True, simulator="MuJoCo CPU",
                scene="sim/lab_scene.py (MjSpec on top of the unmodified upstream XML)", protected_seeds_used=False)
