"""ACT inference wrapper (run in the ACT venv). The ONLY inputs are the proprioceptive state (16) and camera images - the
interface takes nothing else, so simulator object coordinates cannot reach the policy. Output = safe ctrl (16) after
clip + slew limiting (control/action_adapter.py). Latency is measured around the whole act() call (pre-process, model,
post-process, safety)."""
import sys, json, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

class ACTRunner:
    def __init__(self, ckpt_dir, adapter, n_action_steps=None, threads=1, temporal_ensemble=None):
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        torch.set_num_threads(threads)
        self.policy = ACTPolicy.from_pretrained(str(ckpt_dir)); self.policy.eval()
        if n_action_steps: self.policy.config.n_action_steps = n_action_steps
        if temporal_ensemble:     # inference-only: ACT temporal ensembling (needs a chunk every step, so n_action_steps=1)
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
            self.policy.config.temporal_ensemble_coeff = temporal_ensemble; self.policy.config.n_action_steps = 1
            self.policy.temporal_ensembler = ACTTemporalEnsembler(temporal_ensemble, self.policy.config.chunk_size)
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=str(ckpt_dir))
        self.adapter, self.cams = adapter, [k.split(".")[-1] for k in self.policy.config.image_features]
        self.prev_ctrl = None; self.latency_ms = []; self.query_ms = []; self.clipped = 0; self.slewed = 0; self.raw = []
        self.input_keys = sorted(self.policy.config.input_features)
        assert all(k == "observation.state" or k.startswith("observation.images.") for k in self.input_keys), self.input_keys

    def reset(self, ctrl0):
        self.policy.reset(); self.prev_ctrl = np.asarray(ctrl0, float).copy(); self.raw = []

    @torch.no_grad()
    def act(self, state, images):
        """state (16,) float, images {cam: HxWx3 uint8}  ->  ctrl (16,)."""
        t0 = time.perf_counter(); q = len(self.policy._action_queue) == 0
        batch = {"observation.state": torch.as_tensor(state, dtype=torch.float32)[None]}
        for c in self.cams: batch[f"observation.images.{c}"] = torch.as_tensor(images[c]).permute(2, 0, 1).float().div(255.0)[None]
        a = self.post(self.policy.select_action(self.pre(batch)))
        a = a.squeeze(0).cpu().numpy() if hasattr(a, "cpu") else np.asarray(a).reshape(-1)
        self.raw.append(np.asarray(a, np.float32).copy())      # policy output BEFORE clip/slew (diagnostics)
        ctrl, nr, ns = self.adapter.to_ctrl(a, self.prev_ctrl); self.prev_ctrl = ctrl; self.clipped += nr; self.slewed += ns
        ms = (time.perf_counter() - t0) * 1e3; self.latency_ms.append(ms)
        if q or self.policy.config.temporal_ensemble_coeff is not None: self.query_ms.append(ms)
        return ctrl
