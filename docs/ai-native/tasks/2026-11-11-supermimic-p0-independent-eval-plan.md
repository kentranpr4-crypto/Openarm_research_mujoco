# SuperMimic P0 (reaching): read-only audit and 100-goal-per-seed independent evaluation plan

Status: PLAN (nothing executed on the checkpoints). No new training, no G2a/G2b execution, no protected seeds 31-33. Evidence labels: `[verified]` read/computed from files or code, `[reported]` from the user's pasted logs, `[inference]`, `[estimate]`.

## 1. Read-only audit

### 1.1 What is established
| Item | Result | Label |
|---|---|---|
| Checkpoints | seeds 1000/1001/1002: `best.pt` (iterations 3000 / 1900 / 2600) and `latest.pt` (3000); SHA-256 equals `manifest.json` for all six; `COMPLETED` present; 289,536 env steps each | `[reported]` |
| Source identity | trained at commit `3ee510b595`; `git diff 3ee510b HEAD -- research/supermimic_openarm` is empty (SuperMimic source unchanged) | `[verified]` + `[reported]` |
| Earlier results | 64 goals/seed (seeds 9,500,000+): 64/64 each, final error 0.6-1.2 mm. Visual check, 10 fixed goals (9,800,000+): 30/30, longest hold 95-97 steps, first entry into the goal region 0.25-0.28 s, world-model rollout RMSE 0.017-0.027 (normalised) | `[reported]` |
| Success criterion | TCP < 2 cm and TCP speed < 0.10 m/s for 5 consecutive steps (0.25 s), within 100 steps; code: `Reaching` in `research/supermimic_openarm/envs/tasks.py`, `hold_steps = 5` | `[verified]` |

### 1.2 Findings that limit what the existing results can mean
1. **The three runs share the same training goals.** `train_seed_base` (default 1,000,000) is identical in all three runs; the training episode seed is `SeedSequence([train_seed_base, env_idx, episode_idx])` and does not involve `experiment.seed` (`trainer.py`, `_make_vec`; the overnight script overrides only `experiment.seed`). Seeds 1000-1002 therefore differ in network initialisation, action noise and minibatch sampling, NOT in the training goal sequence. Between-seed spread understates data-sampling variability and the three seeds are not three independent replicates of the whole procedure. `[verified in code; to be confirmed on the runs' config.resolved.json, step 2a below]`
2. **Checkpoint selection used a fixed 32-goal set** (seeds 9,000,000-9,000,031, re-evaluated every 100 iterations) and chose `best.pt` by mean RETURN, not success. A small optimistic bias is possible for those 32 goals; it does not carry to the disjoint 9,500,000+, 9,800,000+ and planned 9,900,000+ lists. `[verified in code]`
3. **Return does not separate "arrived and still" from "arrived and shaking".** In the training curves several intermediate checkpoints had success 0.00 with return 190+ and error 6-11 mm (TCP not settled: speed or hold criterion failed); seed 1000 dropped to 0.69 at iteration 2500 after reaching 1.00. The three selected checkpoints do not show this, but late-training instability is real. `[reported]`
4. **In-distribution only.** All evaluation goals come from the training goal box (x 0.32-0.50, y 0.05-0.25, z 0.44-0.58 m). No out-of-distribution goals, no observation noise, no action delay, no perturbation. `[verified]`
5. **Aggressive motion.** The arm enters the goal region within about 5 control steps, i.e. at the placeholder slew limit (2 rad/s per joint); this limit is a simulation value, not a motor specification. Relevant only for any hardware discussion. `[reported]`
6. **Seed hygiene (computed):** the planned lists 9,900,000-9,900,099 and 9,800,000-9,800,099 have 0 collisions with the (upper-bound 7,200) training episode seeds, with the selection list, and with each other (except that the 10 visual-check seeds are the first 10 of the 9,800,000 list), and contain none of 31-33. `[verified]`

### 1.3 Read-only commands for the audit (run on the machine holding the runs; they write nothing)
```bash
cd ~/openarm_research_mujoco/outputs/supermimic_openarm
for S in 1000 1001 1002; do                      # 2a: do the runs really share train_seed_base? seeds differ only in experiment.seed?
  python3 -c "import json;c=json.load(open('reaching_long_s$S/seed$S/config.resolved.json'));print($S,'experiment.seed',c['experiment']['seed'],'train_seed_base',c['train']['train_seed_base'],'eval_seed_base',c['train']['eval_seed_base'],'eval_episodes',c['train']['eval_episodes'])"
done
sha256sum reaching_long_s*/seed*/checkpoints/*.pt      # compare by eye with checkpoints/manifest.json (the visualiser already verified best.pt)
```

## 2. Independent evaluation plan (100 goals per seed)

### 2.1 Design
* **Models:** 3 seeds x {`best.pt`, `latest.pt`} = 6 checkpoints. `best.pt` is the primary object (it is what was selected); `latest.pt` is secondary. No re-selection after seeing results.
* **Goals:** primary list = seeds **9,900,000 + i, i = 0..99**, identical for all 6 checkpoints (deterministic goal from the seed; the tool verifies identical goals). Secondary dwell/speed metrics come from `visualize --no-render --episodes 100`, which uses 9,800,000 + i (i = 0..99); its first 10 goals were already viewed in the visual check (viewing is not training or selection; disclosed).
* **Isolation:** copy the three run directories to a new scratch directory (about 140 MB) and evaluate the COPIES, so the original run directories never receive new files. Nothing is trained, no parameter is updated, the originals are never modified.
* **Evaluation protocol:** deterministic actor, privileged-state reaching input as in training, same environment and original action adapter, 100-step horizon, same success criterion as above.

### 2.2 Pre-registered endpoints and decision rule (fixed now, before any run; not to be changed after results)
| Id | Endpoint | Threshold |
|---|---|---|
| P1 | success of each `best.pt` on the 100 goals | >= 95/100 for every seed |
| P2 | pooled `best.pt` success (300 episodes) | >= 0.97 |
| S1 | median final TCP error of each `best.pt` | <= 5 mm |
| S2 | fraction of episodes with longest in-criterion run >= 50 steps (each model, from `visualize`) | >= 0.95 |
| S3 | mean fraction of steps in criterion after the first hold (each model) | >= 0.95 |
| F | `latest.pt` success below 0.90 | reported as a flag, not a pass criterion |
Verdict: **PASS** if P1, P2, S1, S2, S3 all hold; **PARTIAL** if P1 and P2 hold but any S fails; **FAIL** otherwise; **INCOMPLETE** if a file is missing. Intervals: Wilson 95% per checkpoint; the pooled interval is reported with the caveat of finding 1. A FAIL or PARTIAL triggers analysis and a request for a separate decision; it never triggers automatic retraining.

### 2.3 Commands (run on the machine with the checkpoints; every command writes only into new directories)
```bash
cd ~/openarm_research_mujoco && source .venv-supermimic/bin/activate
mkdir -p ~/p0_independent && cp -r outputs/supermimic_openarm/reaching_long_s1000 outputs/supermimic_openarm/reaching_long_s1001 outputs/supermimic_openarm/reaching_long_s1002 ~/p0_independent/
# 1) success / error / world-model error / latency, 100 NEW goals, best and latest (about 1 min each; CPU, GPUs untouched)
for S in 1000 1001 1002; do for C in best latest; do
  python -m research.supermimic_openarm.evaluate --checkpoint ~/p0_independent/reaching_long_s$S/seed$S/checkpoints/$C.pt --task reaching --episodes 100 --seed-base 9900000 --device cpu --save-traj 0 | grep -E '"success_rate"|"mean_final_goal_error_m"'
done; done
# 2) dwell / stay metrics, 100 goals, no video (best.pt)
python -m research.supermimic_openarm.visualize --run-dir ~/p0_independent/reaching_long_s1000/seed1000 --run-dir ~/p0_independent/reaching_long_s1001/seed1001 --run-dir ~/p0_independent/reaching_long_s1002/seed1002 \
  --labels "seed 1000" "seed 1001" "seed 1002" --episodes 100 --no-render --out ~/p0_independent/viz_best_100
# 3) apply the pre-registered rule
python3 docs/ai-native/tasks/2026-11-11-supermimic-p0-aggregate.py --runs ~/p0_independent/reaching_long_s1000/seed1000 ~/p0_independent/reaching_long_s1001/seed1001 ~/p0_independent/reaching_long_s1002/seed1002 \
  --viz-json ~/p0_independent/viz_best_100/results.json --out ~/p0_independent/p0_report.json
```
(The aggregation script and the repo-relative path `docs/ai-native/tasks/...` exist only in the cloud checkout; copy the script to the workstation first, or run it from there once it is in git.)

### 2.4 Cost and limits
`[estimate]` about 6-10 CPU-minutes in total (6 evaluations of 100 episodes, plus 3 x 100 episodes without rendering); no GPU use. The evaluation cannot show: behaviour on goals outside the training box, robustness to sensing or actuation noise, hardware readiness, or anything about pushing, pick-and-place or WAM. Tested here only on the untrained debug checkpoints (script logic, not results).

### 2.5 Deliverables after execution
Per-checkpoint table (success, Wilson 95%, median/max error, failing seed ids, WM rollout RMSE, inference latency), pooled `best.pt` result with the caveat, S2/S3 table, verdict, provenance (checkpoint SHA-256, commit), and any failing episodes listed by seed for follow-up (e.g. with the `visualize` tool on those seeds).
