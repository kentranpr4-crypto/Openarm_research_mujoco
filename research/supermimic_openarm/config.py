"""Typed, validated configuration (YAML).  `load_config(path, overrides)` -> Config; `Config.validate()` raises ValueError with the reason."""
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
import copy
from .utils import assert_seeds_allowed
TASKS = ("reaching", "pushing", "pick_place")
HORIZONS = (8, 16, 32)
@dataclass
class ExperimentCfg: name: str = "debug"; seed: int = 1000; output_dir: str = "outputs/supermimic_openarm"; algo: str = "supermimic"          # supermimic | ppo | bc
@dataclass
class EnvCfg: task: str = "reaching"; num_envs: int = 4; backend: str = "sync"; control_hz: int = 20; max_steps: int = 0; sim_backend: str = "mujoco_cpu"; task_params: dict = field(default_factory=dict)     # max_steps 0 -> task default
@dataclass
class ReplayCfg: capacity_per_env: int = 2000
@dataclass
class WorldModelCfg: hidden: int = 256; layers: int = 3; window: int = 4; batch_size: int = 256; updates_per_iter: int = 4; lr: float = 3e-4; grad_clip: float = 1.0; term_weight: float = 1.0; contact_head: bool = True; reward_head: bool = False; contact_weight: float = 0.5; reward_weight: float = 0.5
@dataclass
class ActorCfg: hidden: int = 256; layers: int = 3; lr: float = 3e-4; action_param: str = "delta"; action_scale: float = 1.0; noise_std: float = 0.1        # action_param: delta | absolute
@dataclass
class CriticCfg: hidden: int = 256; layers: int = 2; lr: float = 3e-4; epochs: int = 2; minibatches: int = 2
@dataclass
class ImaginationCfg: horizon: int = 8; batch_size: int = 512; chunk_size: int = 512; gamma: float = 0.99; lam: float = 0.95; grad_clip: float = 1.0; grad_clip_elementwise: float = 0.1
@dataclass
class TrainCfg: iterations: int = 20; steps_per_iter: int = 8; warmup_steps: int = 32; eval_interval: int = 10; eval_episodes: int = 8; checkpoint_interval: int = 10; log_interval: int = 1; save_replay: bool = False; eval_seed_base: int = 9_000_000; train_seed_base: int = 1_000_000
@dataclass
class SystemCfg: device: str = "cpu"; precision: str = "fp32"; tf32: bool = True; max_vram_gb: float = 20.0; torch_threads: int = 0                    # precision: fp32 | bf16
@dataclass
class PPOCfg: rollout_steps: int = 64; epochs: int = 4; minibatches: int = 4; clip: float = 0.2; lr: float = 3e-4; gamma: float = 0.99; lam: float = 0.95; ent_coef: float = 0.0; vf_coef: float = 0.5; init_log_std: float = -1.0; hidden: int = 256
@dataclass
class BCCfg: demos: int = 64; epochs: int = 200; lr: float = 1e-3; batch_size: int = 256; hidden: int = 256
@dataclass
class Config:
    experiment: ExperimentCfg = field(default_factory=ExperimentCfg); env: EnvCfg = field(default_factory=EnvCfg); replay: ReplayCfg = field(default_factory=ReplayCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg); actor: ActorCfg = field(default_factory=ActorCfg); critic: CriticCfg = field(default_factory=CriticCfg)
    imagination: ImaginationCfg = field(default_factory=ImaginationCfg); train: TrainCfg = field(default_factory=TrainCfg); system: SystemCfg = field(default_factory=SystemCfg)
    ppo: PPOCfg = field(default_factory=PPOCfg); bc: BCCfg = field(default_factory=BCCfg)
    def validate(self):
        e, w, im, t, s = self.env, self.world_model, self.imagination, self.train, self.system
        def req(c, msg):
            if not c: raise ValueError("invalid config: " + msg)
        req(e.task in TASKS, f"env.task must be one of {TASKS}, got {e.task!r}")
        req(self.experiment.algo in ("supermimic", "ppo", "bc"), "experiment.algo must be supermimic|ppo|bc")
        req(e.sim_backend == "mujoco_cpu", "env.sim_backend: only 'mujoco_cpu' is implemented. A MuJoCo-Warp/mjlab backend is NOT verified for the OpenArm scene (docs/reproduction_audit.md)")
        req(e.backend in ("sync", "subproc"), "env.backend must be sync|subproc"); req(e.num_envs >= 1, "env.num_envs >= 1"); req(e.control_hz == 20, "env.control_hz must be 20 (existing action-adapter contract)")
        req(im.horizon in HORIZONS, f"imagination.horizon must be one of {HORIZONS}"); req(0 < im.gamma <= 1 and 0 <= im.lam <= 1, "gamma in (0,1], lam in [0,1]")
        req(im.chunk_size >= 1 and im.batch_size >= 1, "imagination batch/chunk sizes >= 1"); req(w.window >= 1 and w.batch_size >= 1, "world_model.window/batch_size >= 1")
        req(self.actor.action_param in ("delta", "absolute"), "actor.action_param must be delta|absolute"); req(self.actor.action_scale > 0, "actor.action_scale > 0")
        req(s.precision in ("fp32", "bf16"), "system.precision must be fp32|bf16"); req(s.device == "auto" or s.device == "cpu" or s.device.startswith("cuda"), "system.device must be cpu|cuda[:i]|auto")
        req(self.replay.capacity_per_env >= (w.window + 2) * 2, "replay.capacity_per_env too small for the world-model window")
        req(t.iterations >= 1 and t.steps_per_iter >= 1 and t.eval_episodes >= 1, "train.iterations/steps_per_iter/eval_episodes >= 1")
        req(t.checkpoint_interval >= 1 and t.eval_interval >= 1, "checkpoint/eval intervals >= 1"); req(s.max_vram_gb > 0, "system.max_vram_gb > 0")
        req(t.warmup_steps * e.num_envs >= w.batch_size // 8 or t.warmup_steps >= w.window + 2, "train.warmup_steps too small to fill one world-model window")
        assert_seeds_allowed(self.experiment.seed, t.eval_seed_base, t.train_seed_base)
        return self
    def to_dict(self): return asdict(self)
def _merge(obj, d, path=""):
    for k, v in d.items():
        names = {f.name: f for f in fields(obj)}
        if k not in names: raise ValueError(f"unknown config key: {path}{k}")
        cur = getattr(obj, k)
        if is_dataclass(cur): 
            if not isinstance(v, dict): raise ValueError(f"{path}{k} must be a mapping")
            _merge(cur, v, path + k + ".")
        else:
            if isinstance(cur, bool) and not isinstance(v, bool): raise ValueError(f"{path}{k} must be a boolean")
            if isinstance(cur, (int, float)) and not isinstance(cur, bool) and not isinstance(v, (int, float)): raise ValueError(f"{path}{k} must be numeric")
            setattr(obj, k, type(cur)(v) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else v)
def from_dict(d):
    c = Config(); _merge(c, d or {}); return c.validate()
def load_config(path, overrides=()):
    import yaml
    p = Path(path)
    if not p.is_file(): raise FileNotFoundError(f"config file not found: {path}")
    d = yaml.safe_load(p.read_text()) or {}
    for ov in overrides:
        if "=" not in ov: raise ValueError(f"override must be key.path=value, got {ov!r}")
        k, v = ov.split("=", 1); cur = d
        parts = k.split(".")
        for q in parts[:-1]: cur = cur.setdefault(q, {})
        cur[parts[-1]] = yaml.safe_load(v)
    return from_dict(d)
