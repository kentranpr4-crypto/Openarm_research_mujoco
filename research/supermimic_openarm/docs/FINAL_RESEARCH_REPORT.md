# SuperMimic-OpenArm — Final research report (archive v1.0)

Scope: simulation-only study of model-based actor optimisation through a differentiable learned world model on an OpenArm v1 MuJoCo scene, with privileged simulator state. This report states what was **established**, what was **measured only on proxies**, and what **remains unverified**. No claim here exceeds the evidence cited.

**Evidence labels.** **MEASURED** — produced by a tool in this repository (the tool, its flags and its seeds are documented; the raw result files are not archived). **OWNER-REPORTED** — from the project owner's runs on their own workstation, taken from pasted logs recorded in the development reports; the artifacts are not in the repository. **EXECUTED NOW** — re-run while preparing this archive. Source reports (Vietnamese, with the same labels) are under `docs/ai-native/tasks/` and are cited per section.

## 1. Research question

> Can a differentiable learned dynamics model support model-based actor optimisation for OpenArm manipulation?

Operationally: (a) can an actor trained only by back-propagating imagined returns through a frozen learned model solve a task when evaluated in the real simulator, and (b) where does the learned model fail the optimiser?

## 2. Methods

**Environment interaction.** `OpenArmTaskEnv` wraps the repository's lab scene (`sim/lab_scene.py`) built on the unmodified upstream OpenArm XML (pinned commit, cloned by the setup script). Control runs at 20 Hz with 50 MuJoCo substeps per step. Every command passes through the repository's original `ActionAdapter` (clip to control range, then slew limit); the right arm is held. CPU MuJoCo, synchronous or spawned worker processes; episode seeds are `SeedSequence([train_seed_base, env_idx, episode_idx])`.

**State-based world model.** A residual MLP `s' = s + denorm(MLP(norm(s), norm(a_exec)))` on the 34-D privileged state (`q`, `dq`, `prev_ctrl`, box position/quaternion/velocity), conditioned on the *executed* control (so slew/clip effects are part of the data), with `prev_ctrl' := a_exec` set structurally. Heads: termination, contact. Fit with a k-step (8) autoregressive window loss on replayed real transitions. (`models.py`, `trainer.py`)

**Actor–critic learning.** A tanh MLP actor (delta joint-target parameterisation, exploration noise only in collection) and a goal-conditioned critic trained on λ-returns of imagined trajectories (γ = 0.99, λ = 0.95).

**Differentiable imagination.** From replay start states, the actor acts on predicted states for H = 16 steps; the frozen world model predicts transitions; a differentiable FK layer gives the TCP; rewards are analytic; the objective is the survival-weighted return plus a critic bootstrap, back-propagated through the dynamics (global gradient-norm clip 1.0, element-wise clamp 0.1). Imagined rollouts are **model predictions, never simulator trajectories**. (`imagination.py`, `kinematics.py`)

**Contact-rich task development (P1).** A pushing task with a 5×4×4 cm box, physically validated randomisation, a strict success criterion (box within 3 cm, at rest, never lifted > 1 cm, 5 held steps), a scripted expert as a reference controller, regime-labelled diagnostic datasets (free-space, first contact, sustained contact, contact loss), controlled world-model ablations, and pre-registered monitoring gates for a future actor pilot (`gates/p1_pilot_gates.yaml`, SHA-256 `a7dc91ca66f6350d4798602aa86258a4c01777d9d123f2e724519499d24a8dca`).

## 3. P0 — reaching

| Item | Value | Label | Source |
|---|---|---|---|
| Training runs | 3 (`experiment.seed` 1000/1001/1002) | OWNER-REPORTED | `docs/ai-native/tasks/2026-11-11-supermimic-p0-independent-eval-plan.md` §1.1 |
| Iterations per run | 3,000 | OWNER-REPORTED | same |
| Environment steps | 289,536 per run; **868,608 in total** (= 3 × (3,000 × 4 × 24 + 64 × 24)) | OWNER-REPORTED; arithmetic checked | same + `configs/reaching_4090.yaml` (`steps_per_iter`, `warmup_steps`) |
| Completion / integrity | all `COMPLETED`; `best.pt` (iterations 3000 / 1900 / 2600) and `latest.pt` (3000) SHA-256 equal `manifest.json` | OWNER-REPORTED | same |
| Checkpoint provenance | trained at commit `3ee510b595`; the SuperMimic source was byte-identical between that commit and the commit used for evaluation | OWNER-REPORTED + code diff | `docs/visualize.md` |
| Held-out evaluation | **30/30** successes on **10 shared goals** (seeds 9,800,000 + i); mean final TCP error per run **0.7–1.5 mm** | OWNER-REPORTED | `docs/visualize.md`; evaluation tool `visualize.py` |
| Earlier evaluation | 64/64 per run on 64 goals (seeds 9,500,000+), error 0.6–1.2 mm | OWNER-REPORTED | plan §1.1 |
| Success criterion | TCP < 2 cm and TCP speed < 0.10 m/s for 5 consecutive steps within 100 steps | code | `envs/tasks.py` (`Reaching`) |

**Generalisation limits (verified from the code).** (1) The three runs share `train_seed_base`, hence one training-goal sequence; `experiment.seed` only changes initialisation, noise, warm-up actions and replay sampling — the runs are *partially* independent (`p0_seed_audit.py` documents and checks this). (2) The 30/30 are ten common goals, not thirty independent configurations. (3) `best.pt` was chosen by mean return on a fixed 32-goal set (seeds 9,000,000+), and return does not separate "arrived and still" from "arrived and shaking". (4) All goals come from the training goal box. (5) Motion reaches the goal region at the slew limit (a simulation placeholder). (6) The independent 100-goal-per-seed evaluation (`p0_final_eval.py`, goals 9,900,000–9,900,099, pre-registered thresholds) was **prepared and tested only on a debug checkpoint; it has NOT been completed**. No real-robot validation.

## 4. P1 — pushing

### 4.1 Simulation repair (MEASURED; `p1_validate_env`, 11/11 PASS, 2026-11-11)
Defects found in the original pushing environment and their measured repair:

| Defect | Evidence before | Repair | Evidence after |
|---|---|---|---|
| Friction scale ≤ 1.0 had no effect | MuJoCo uses the larger of the two geoms' friction (desk 0.8 vs box 0.8 × scale); sliding distance identical for scales 0.7/0.85/1.0 | box geoms get contact `priority = 1` in `friction_mode="pair_exact"` | effective box–desk μ read from real contacts = `0.8 × scale` (max error 0.0 over 300 resets); sliding distance 22.8 / 18.9 / 16.2 / 14.1 / 12.3 mm vs Coulomb `v²/(2μg)` within 2 % |
| Constant mass | one value, 0.05 kg | scale 0.6–1.6 with principal inertia scaled consistently (uniform cuboid) and `mj_setConst` | mass 0.031–0.080 kg (300 distinct); inertia error 2e-16; acceleration under 0.6 N 14.5 / 4.6 / 0.057 m/s² |
| Invalid goals (≈ 2 % inside the tray; goals silently clipped) | 6/300 | rejection sampling: workspace, tray/cube/cylinder footprints, blocked straight path | 0 invalid goals; 35.7 % of first draws re-sampled; goal distance exactly 0.07–0.13 m |
| Expert lifted the box (19 of 28 failures) | box hit at 0.3–0.6 m/s | speed-capped phase-machine expert (below) | 1 lift in 300 episodes |

The strict success criterion was **not** changed. Other checks: rest stability 2.6 µm drift; executed controls equal an independent clip→slew reference to 1.1e-7 with zero violations; recomputed success equals the simulator flag (0 mismatches in 100 episodes); deterministic reset and **bit-exact snapshot replay** (difference 0.0); minimum box–robot penetration −0.56 mm (pre-registered limit −2 mm). Reaching and pick-and-place trajectories are **bit-identical to the pre-change commit** (test `test_p0_compat…`). Sources: `docs/ai-native/tasks/2026-11-11-supermimic-p1-env-repair-report.md`, `tests/test_p1_env.py`.

### 4.2 Scripted pushing expert (MEASURED)
Parameters were chosen on tuning seeds only (24,000,000–24,000,047: 42/48). Strict success on three fresh development sets, never used for tuning: **83/100** (Wilson 74.5–89.1 %), **89/100** (81.4–93.7 %), **88/100** (80.2–93.0 %); **pooled 260/300 = 86.7 %** (Wilson 95 % 82.4–90.1 %). Remaining failures are mostly overshoot past the goal. Before the repair the strict success was 20/48 = 41.7 %. This is a **scripted reference controller**, not a learned policy.

### 4.3 World-model diagnostics (MEASURED, **proxy** model and data)
Setup (`p1_wm_eval`): 360 training / 120 validation episodes (54,000 / 18,000 transitions) from the scripted pusher with noise, a fast "poke", and random actions; regimes labelled from finger/hand contact flags (free-space; first contact; sustained contact; contact loss). A 6,000-update model on CPU. **Not the world model of a P1 actor run.**

Box-position error after 16 open-loop steps (mm; same executed actions as the simulator):

| Regime | n windows | World model | Stationary | Constant velocity |
|---|---:|---:|---:|---:|
| free space | 5,857 | 4.09 | **0.42** | 0.98 |
| first contact | 1,381 | **13.0** | 39.0 | 40.8 |
| sustained contact | 305 | **15.0** | 38.9 | 33.9 |
| contact loss | 474 | **14.6** | 35.7 | 55.7 |

Findings: the model is far better than both baselines in contact, but worse than "the box stays" in free space (ghost motion ≈ 0.39 mm/step, systematic, independent of the pusher's distance), and it does **not** improve box orientation over the stationary baseline in contact (13.3° vs 12.4° at 16 steps, first contact). The first-contact and contact-loss cells are small (about 200 validation transitions each), so their intervals are wide.

### 4.4 Model exploitation (MEASURED, proxy)
* **Matched trajectories** (same start state, same executed actions, H = 16, 8,160 windows): imagined and real returns correlate at 0.99; mean optimism gap **+0.08** (95 % CI +0.05…+0.12); largest in sustained contact (+0.38, CI −0.002…+0.76), mainly through the near-goal bonus term.
* **After action optimisation** (40 Adam steps on the imagined return of 16 actions, warm-started from recorded actions, then executed in MuJoCo from the same snapshot; 24 start states): the gap rises to **+8.4 … +9.0** while the real-return gain is not significant (e.g. +2.7, CI −1.7…+7.0; an earlier model: −0.55). The optimiser exploits the model.
* **Gradient fidelity vs finite differences of the real return** (192 perturbations per model): real return is higher along +gradient than along −gradient in 83–88 % of states (chance 50 %), sign agreement 0.71–0.78 where the response is measurable, correlation 0.28–0.41; the model *overestimates* the slope (≈ 2.5–10×), which is the mechanism of the optimism gap.

### 4.5 Mitigation ablations (MEASURED, proxy; one factor changed per arm; baseline seed spread from three seeds)
| Variant | Result |
|---|---|
| Contact-gated box dynamics | free-space error 4.3 → **0.79 mm** (−82 %); orientation 3.1° → 0.7°; contact error +1–1.5 mm; **no reduction of exploitation** (gap +11.2) |
| Multi-step window K | K = 1 worse everywhere; K = 16 no better than K = 8 and 2.2× slower |
| Contact-balanced replay | no contact gain beyond seed spread; free-space worse |
| Shorter imagination horizon (8, 4) | **no help** (gap +11.4, +7.1; real gain negative) |
| Action trust region ±0.1 | gap +2.8 (CI +1.2…+4.6); real gain +1.7 (CI +0.6…+3.0) |
| Ensemble-disagreement penalty (3 members, λ = 3) | gap +2.4; real gain +3.2 (CI +1.5…+4.8); member disagreement 11.5 → 0.7 |
| Combined (h8, 20 steps, trust ±0.1, λ = 1) | gap +1.2 (CI −0.2…+2.7); real gain +1.3 |

Optimised actions mostly stay inside the training action distribution (only 6 % of steps beyond the 99th-percentile Mahalanobis distance) while ensemble disagreement rises 3.6×: the exploitation happens in unseen **state–action combinations**, so action-marginal checks are insufficient. All of this is on proxy models with 24 start states and one seed per non-baseline arm; it **motivates** the opt-in variants B1/B2/B3 (`p1_variants.py`) but is **not** confirmation that they help a trained actor.

### 4.6 Actor experiment status — NOT COMPLETED
No P1 actor was trained. A registered pilot protocol exists (B0 original trainer, 1 seed, ≤ 2 h, seven pre-registered gates with a checker, `gates/p1_pilot_gates.yaml`; A/B protocol `experiments/p1_actor_ab_protocol.yaml`, SHA-256 `126266d5f4235249a1cfde2070fee2b86d89f1efd09c0d332abc1b335d5611f2`, draft and unauthorised). **Colab Tesla T4 benchmark (run once by the owner, 2026-11-12; OWNER-REPORTED from the terminal log): NO-GO.**

| Condition | Measured on the T4 VM | Required | Result |
|---|---|---|---|
| Host RAM for the registered config (24 spawned workers) | 1,732.7 MiB per worker → predicted **41.9 GiB** vs 11.3 GiB available | fits with 15 % headroom | **fail** |
| Peak VRAM (diagnostic run, registered batch sizes) | 811 MiB of 15,360 MiB | ≤ 80 % | pass |
| Iteration time | 2.14 s → 2,441 iterations in 2 h incl. checkpoints, evaluation and gate cost | ≥ 4,000 | **fail** |

The registered config was not altered; the pilot did not run; artifacts were verified and the session released. This says nothing about an RTX 5090. The driver's control flow was validated against a fake CLI (which exposed and fixed four defects) — it proves nothing about Colab itself.

## 5. P2 — pick-and-place (status only)
Implemented: environment, task, reward (carry/place/release terms), strict success (lifted earlier, in the tray region, at rest, fingers open and not touching, 10 steps), configuration, interface tests. **Not completed:** no training run, no evaluation, no success claim. **Identified, not repaired:** a read-only audit on real contacts (`p1_pickplace_audit.py`) found that box–desk/tray/obstacle friction uses `max(box, other)` so scales ≤ 1.0 (half of the 0.8–1.2 range) have no effect, and the finger–box (grasp) friction is fixed at 1.0 (finger geoms have priority 1) so it is **not randomised at all**. Options and acceptance criteria are recorded in `docs/ai-native/tasks/2026-11-11-pickplace-friction-issue.md`. The pushing repair was deliberately **not** copied into PickPlace.

## 6. Scientific limitations
1. **No real-robot or Sim2Real validation**; privileged state (box pose/velocity) for P1/P2.
2. **No completed P1 or P2 actor training**; no evidence that SuperMimic pushing reaches the scripted expert's 86.7 %.
3. **Limited independent seed diversity** for P0 (shared training-goal sequence; selection on a fixed goal set); 100-goal independent evaluation incomplete.
4. **Proxy-only world-model results** (small datasets, one seed per ablation arm, 24 optimisation start states).
5. **No demonstrated advantage over matched ACT/PPO baselines**; baselines exist (PPO, BC) but were not compared.
6. **Environment mismatch of the archival tests** (Python 3.13 / PyTorch 2.11 CPU vs the pinned Python 3.10 / PyTorch 2.9.1); GPU numerics (TF32/bf16) untested.
7. The original paper's compute-scale and humanoid results are **not** reproduced; this is a SuperMimic-style adaptation.

## 7. Conclusion
**Established (simulation):** (i) a SuperMimic-style loop — collect, fit a differentiable world model on executed controls, optimise an actor by back-propagation through frozen imagined rollouts — is implemented, tested at unit and pipeline level, and, per the owner's reports, trained actors that reach held-out goals on the simple reaching task; (ii) a contact-rich pushing environment was repaired and validated physically, with a scripted expert at 86.7 % strict success; (iii) on proxy models, a learned world model predicts contact far better than stationary or constant-velocity baselines yet drifts in free space, and an optimiser *exploits* it (optimism gap ≈ +8–9), with action trust regions and ensemble-disagreement penalties reducing the gap to ≈ +2.4–2.8 and contact gating fixing free-space drift but not exploitation.

**Unverified / not done:** whether SuperMimic can train a pushing or pick-and-place actor on this scene; whether the proposed mitigations help in actual training; independent 100-goal P0 generalisation; any comparison with ACT or PPO; any GPU benchmark on the intended workstation; any real-robot behaviour.

## 8. Reproducibility record (this archive)
* Repository tests (`python -m pytest research/supermimic_openarm/tests -q`): **80 passed, 0 failed, 0 skipped (118 s; Python 3.13, PyTorch 2.11, CPU)** (EXECUTED NOW; Python 3.13 / PyTorch 2.11 CPU). Skipped/not run: `tests/run_t4_driver_selftest.py` (slow, manual; passed on 2026-11-12), GPU-only paths, full P0/P1 training.
* `scripts/smoke_test.sh`: **exit 0: 44 unittest tests OK, 5/5 config dry-runs, CPU mini-training (4 iterations, `bad_grads=0`), evaluation of the smoke checkpoint, CPU benchmark verdict `NOT_TESTED_NO_GPU`** (EXECUTED NOW).
* All five shipped configs validate (`train --dry-run`, 5/5).
* Pre-registered hashes: gates `a7dc91ca66f6350d4798602aa86258a4c01777d9d123f2e724519499d24a8dca`; A/B protocol `126266d5f4235249a1cfde2070fee2b86d89f1efd09c0d332abc1b335d5611f2`; validation dataset (not archived) `92b9e5185a313e1095e49729c5a974bb6cf45fba4e6be975bad11d2561bf23c6`.
* Not archived by design: checkpoints, datasets, run outputs, Colab result archives, OpenArm-WAM, G2a/G2b data, credentials.
