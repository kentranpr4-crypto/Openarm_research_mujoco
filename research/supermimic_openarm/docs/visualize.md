# Visual evaluation of trained reaching checkpoints (`visualize.py`)

Read-only: loads `best.pt` (weights only, `model.eval()`, `torch.no_grad()`), never trains, never updates parameters, never modifies a checkpoint, never overwrites an output directory (`--out` must be new/empty).

## Run (on the machine that has the checkpoints, e.g. the workstation)
```bash
cd ~/openarm_research_mujoco && source .venv-supermimic/bin/activate
pip install matplotlib imageio-ffmpeg                      # only for this tool
python -m research.supermimic_openarm.visualize \
  --run-root outputs/supermimic_openarm --seeds 1000 1001 1002 --episodes 10 --gl egl \
  --out outputs/supermimic_viz/reaching_final
```
`--gl egl` (GPU, headless) or `--gl osmesa` (CPU software; needs `libosmesa6`). `--no-render` gives metrics and plots only. `--width/--height/--fps` change the video size.
The 10 held-out goals come from ONE fixed seed list, 9,800,000 + i (i = 0..9), identical for every model (verified: the tool aborts if goals differ). They are disjoint from the checkpoint-selection seeds (9,000,000+), the earlier evaluations (9,500,000+, 9,700,000+) and from 31-33.

## What it checks before running
`best.pt` exists, SHA-256 equals the run's `manifest.json`, `config.resolved.json` is a SuperMimic reaching run, `metadata.json` records training commit `3ee510b595` (the SuperMimic source is byte-identical between that commit and the current HEAD: `git diff 3ee510b HEAD -- research/supermimic_openarm` is empty). `--allow-commit-mismatch` overrides.

## Outputs (`--out`)
`viewer.html` (open in a browser; plain local page, no server needed) | `videos/<seed>_goalNN.mp4` (actual rendered MuJoCo rollouts) | `comparison_videos/compare_goalNN.mp4` (seeds side by side, same goal) | `plot_distance_to_goal.png`, `plot_tcp_speed.png`, `plot_success_dwell_wm.png` | `results.json` (per-episode TCP trajectory, error, speed, success, dwell, world-model error, goals, provenance, criteria).

## Actual success criterion (`research/supermimic_openarm/envs/tasks.py`, `Reaching`)
Instantaneous: TCP distance to the goal < 0.02 m AND TCP speed < 0.10 m/s. Success = that holds for 5 consecutive control steps (0.25 s at 20 Hz), at any time within 100 steps (5 s). TCP speed = finite difference of the measured TCP between control steps x 20 Hz. The tool recomputes success from the recorded arrays and reports any disagreement with the simulator's own flag (`inconsistent_success_flags`, expected 0).

## Overlays (drawn on the real simulation, nothing synthesised)
Red sphere = goal with the 2 cm success radius; cyan dots = measured TCP trail; yellow = current TCP; text = measured TCP error (mm), TCP speed (m/s), hold counter and SUCCESS status.

## Limitations
Simulation only (privileged-state-trained, not Sim2Real); 10 goals per model is a visual check, not a statistical benchmark (the 64-100 goal evaluations are); videos show one camera angle; the world-model error is the open-loop normalised RMSE along the executed episode, comparable only between models of the same configuration; MP4 encoding needs `imageio-ffmpeg` (falls back to the system `ffmpeg`, then to GIF).
