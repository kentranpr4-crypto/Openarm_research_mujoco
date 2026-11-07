# SuperMimic-OpenArm (isolated research package)

SuperMimic-**style** policy optimisation for the OpenArm v1 MuJoCo scene: a differentiable world model is fit from replayed executed transitions, and the actor is optimised by back-propagating a short imagined return **through the frozen world model**.
**Not an exact reproduction** of the humanoid SuperMimic paper (see `docs/reproduction_audit.md`). **Nothing has been trained at scale and no task is claimed to be solved**: tasks are implemented and tested, not proven trainable.
Privileged-state simulation research only (not Sim2Real). Everything lives under `research/supermimic_openarm/`; outputs go to the git-ignored `outputs/supermimic_openarm/`.

## Tasks (`envs/tasks.py`; each exposes observation/action/reward/termination/success/randomization specs)
| Task | Goal | Success (deterministic, simulator truth) | Status |
|---|---|---|---|
| `reaching` (P0) | TCP to randomised 3-D target | dist < 2 cm and speed < 0.1 m/s | implemented + tested; actor input has no privileged data |
| `pushing` (P1) | push the blue box into a goal region | box within 3 cm, at rest, never lifted > 1 cm | implemented + tested; **not proven trainable** |
| `pick_place` (P2) | grasp, lift, carry, release, settle in the tray | lifted earlier, in tray region, at rest, **fingers open and not touching**, 10 steps | implemented + tested; **not proven trainable** |

## Install on the RTX 4090 workstation (Ubuntu, no sudo)
```bash
git clone https://github.com/kentranpr4-crypto/openarm_research_mujoco.git && cd openarm_research_mujoco
git checkout <branch-with-this-package>
research/supermimic_openarm/scripts/setup_ubuntu.sh        # private .venv-supermimic + pinned deps + pinned upstream model; prints GPU report
research/supermimic_openarm/scripts/check_gpu.sh           # exit 0 only if PyTorch can really use the GPU
research/supermimic_openarm/scripts/smoke_test.sh          # unit tests + CPU mini-training (add --gpu for the 1000-step GPU benchmark)
```
Requirements you must provide: `git`, Python 3.10-3.13 with `venv` (`sudo apt install python3-venv` if missing), a working NVIDIA driver, about 8 GB disk for the wheels. The scripts never install drivers and never use sudo.

**Driver vs CUDA runtime vs PyTorch build vs toolkit**
| Piece | What it is | Who provides it |
|---|---|---|
| NVIDIA driver | kernel module + `nvidia-smi`; sets the highest CUDA version the machine can run | you / your admin (never these scripts) |
| CUDA runtime | the `libcudart`/cuBLAS/cuDNN libraries | **bundled inside the PyTorch wheel** (`nvidia-*-cu12` packages) |
| PyTorch CUDA build | the wheel flavour. `torch==2.9.1` from PyPI on Linux x86_64 is built for **CUDA 12.8** (verified from the PyPI metadata: `nvidia-cuda-runtime-cu12==12.8.90`) | `setup_ubuntu.sh` |
| CUDA toolkit (`nvcc`) | compiler | **not needed** |
Minimum driver for the CUDA 12.8 wheel: a recent R570-series driver is the safe choice (NOT tested on your machine; `check_gpu.sh` tells you if it fails). Do not invent a different combination: if `check_gpu.sh` fails, send its output.

## Commands (run from the repository root, venv `.venv-supermimic`)
```bash
python -m research.supermimic_openarm.train     --config research/supermimic_openarm/configs/reaching_4090.yaml [--override imagination.horizon=32] [--run-name NAME] [--max-iterations N] [--dry-run]
python -m research.supermimic_openarm.train     --resume outputs/supermimic_openarm/reaching_4090/seed1000          # continue an interrupted run
python -m research.supermimic_openarm.evaluate  --checkpoint outputs/supermimic_openarm/reaching_4090/seed1000/checkpoints/best.pt --task reaching --episodes 64
python -m research.supermimic_openarm.benchmark --config research/supermimic_openarm/configs/reaching_4090.yaml --steps 1000
```
Wrappers: `scripts/train_reaching.sh | train_pushing.sh | train_pick_place.sh | evaluate.sh | smoke_test.sh`. The pushing / pick-place scripts refuse to start until the previous task has a COMPLETED run (`ALLOW_SKIP_GATE=1` overrides, knowingly).
All commands print the resolved configuration, validate it, fail with a clear message on missing config/dependencies/assets, record the git commit + library versions (`metadata.json`), and never overwrite a COMPLETED run.
Baselines: `experiment.algo: ppo` (model-free) and `bc` (reaching only, scripted Jacobian demonstrations on the NEW task) share the environment, observations, action adapter, goal seeds and evaluation protocol.

## Configurations
`configs/reaching_4090.yaml`, `pushing_4090.yaml`, `pick_place_4090.yaml`, `debug_cpu.yaml`. Everything is configurable: batch sizes, replay capacity, imagination horizon (8/16/32), `env.num_envs` + `env.backend` (`sync` | `subproc`), `system.device`, `system.precision` (`fp32` | `bf16`, bf16 NOT tested), checkpoint/eval frequency, `system.max_vram_gb` (hard per-process cap via `torch.cuda.set_per_process_memory_fraction`).
Simulation runs on **CPU MuJoCo** (process-parallel); only the neural networks use CUDA. A MuJoCo-Warp/mjlab GPU backend is **not implemented / not verified** for this scene and 4,096 parallel environments are **not assumed**. Set `env.num_envs` to at most (physical cores - 2). Run `benchmark` before choosing values.

## Memory and time: ESTIMATES (no GPU was available when this was written)
* VRAM, derived from saved-tensor bytes measured on CPU for the shipped shapes: imagination chunk (1024 samples) 0.41 GB at H=16, 0.82 GB at H=32; world-model window batch 0.21 GB; weights + Adam < 0.02 GB. With CUDA context and allocator overhead expect **roughly 2-4 GB**, far below the 20 GB cap. `benchmark.py` measures the real peak and returns NO-GO above the cap.
* Environment throughput on a CPU core measured in the validation report (see `docs/validation_report.md`); the wall-clock of a run is dominated by `num_envs x per-core step rate`, not by the GPU. Training duration for any task is **unknown** until a pilot is run; no convergence claim is made.

## Outputs (`outputs/supermimic_openarm/<experiment>/seed<k>/`, git-ignored)
`checkpoints/{latest.pt,best.pt,manifest.json}` (weights, optimizers, RNG, config; sha256 verified on load), `config.resolved.json`, `metadata.json` (commit, versions, GPU), `env_manifest.json` (all task specs, layout, model info, upstream commit), `metrics.jsonl`, `eval/*.json`, `trajectories/*.npz` (actual states, executed actions, WM one-step and imagined-rollout predictions), optional `replay_latest.npz`, `COMPLETED`.
Resume restores networks, optimisers, RNG and (if `train.save_replay: true`) the replay buffer; the environment episode stream restarts with new deterministic seeds. Multiple seeds: change `experiment.seed` (31-33 are rejected).

## RTX 4090 deployment checklist (do not skip steps; no automatic progression to long training)
1. `setup_ubuntu.sh`, `check_gpu.sh` (GPU usable?). 2. `smoke_test.sh` (CPU). 3. `benchmark --config reaching_4090.yaml --steps 1000` (records VRAM, throughput, time/step). 4. Short reaching pilot: `train --max-iterations 200`, then `evaluate` on held-out goals. 5. Only then pushing, then pick-and-place.
**GO** requires: imports/dependencies verified; env steps correctly; actor gradients through the world model non-zero and finite (`bad_grads == 0`, `actor_grad_norm > 0` in `metrics.jsonl`); checkpoints resume; peak VRAM within `max_vram_gb`.
**NO-GO**: silent environment incompatibility, action-semantics mismatch, NaN gradients, GPU OOM, checkpoint corruption, privileged data leaking into a deployment-mode observation.

## Research Studio integration (optional)
`studio_adapter.py` (stdlib + numpy; no torch/mujoco) serves task list, runs, checkpoints, curves, evaluations, trajectories (actual vs predicted next states, imagined rollouts, WM error) and status under `/api/supermimic/*`. `studio/app_hook.patch` is the proposed 6-line hook for `app/backend/server.py` (**not applied**; outside this directory, needs separate approval). Without it the app is unchanged.

## Safety rules
No ACT checkpoint, dataset, G2a/G2b file or protected pose is read; ACT test seeds 31-33 are rejected; upstream OpenArm XML is never modified; nothing outside `outputs/` is written; large outputs are never committed. Do not claim SuperMimic outperforms ACT: no such comparison exists.
