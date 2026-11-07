# SuperMimic reproduction audit (2026-11-06)

**Verdict: this package is a SuperMimic-STYLE adaptation to OpenArm manipulation. It is NOT an exact reproduction of the humanoid paper.**
Evidence labels: `[fetched]` = read from the live source today; `[repo-claim]` = stated by a reference repository, not independently reproduced; `[mine]` = my design decision; `[not verified]`.

## 1. Sources (exact)
| Source | Identifier | Status |
|---|---|---|
| Paper | arXiv:2610.28480, "SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model", Liao, Shao, Long, Zakka, Horowitz, Abbeel, Sreenath; submitted 20 Oct 2026; "Accepted at CoRL 2026 (Oral)" | `[fetched]` abstract page only. **The full PDF/TeX was not read**, so appendix hyperparameter tables are `[not verified]` against the paper |
| Project page | https://supermimic.github.io/ | `[fetched]` (3-step loop description, compute claims, G1 hardware) |
| Reference A (manipulation) | github.com/emoPointer/supermimic_manipulation, commit `eda7da480c6d92a68cc41c9628b7efdae08cae23`, **MIT**, 780 tracked files | `[fetched]` README, pyproject, tree |
| Reference B (humanoid, mjlab) | github.com/YHFone/supermimic-mjlab, commit `9aab0a588c6941bf9862a2d444b711fd73e2b832`, **Apache-2.0**, 43 tracked files | `[fetched]` README, pyproject, `algo/config.py`, `algo/trainer.py` (collect / world-model fit / imagination update) |

The pages were unreachable at first (network allow-list); `arxiv.org` and `supermimic.github.io` were then allowed in the user's cloud environment so the audit could read them.

## 2. What the sources say (summary)
* Paper/project page `[fetched]`: one differentiable world model fit from logged transitions of a standard rigid-body simulator; world model and policy train together in one loop (collect -> fit -> imagine); backpropagation through time over **short imagined rollouts** gives the policy analytic gradients; gradients never pass through the simulator; humanoid (Unitree G1) whole-body motion tracking over 129,514 motions; claims 3 h on one GPU surpass a 7-day/128-GPU baseline (H100 for SuperMimic).
* Reference B `[fetched]`: Algorithm A.1 loop `collect -> fit k-step autoregressive windows (L_WM) -> improve policy with frozen f_w over H imagined steps (BPTT + GAE critic)`. Defaults claimed to follow Table A.5: horizon 32, window 8, world-model MLP 2048x3, lr 2e-4, gamma 0.99, lambda 0.95, element-wise gradient clip 0.1, action noise 0.1, 4096 envs on MuJoCo Warp, observation history 10. Requires mjlab>=1.6, torch>=2.7, mujoco>=3.11, Python>=3.10; **GPU simulator needed for training**.
* Reference A `[fetched]`: the same idea on a simulated ARX5 arm, five manipulation tasks, OmniReset-style reset banks, mjlab 1.5.3 + torch 2.9.1 (cu128), **Python >=3.12,<3.14**. Its README task-success numbers are `[repo-claim]`.

## 3. Reuse: legal and technical
| Question | Answer |
|---|---|
| Legal | MIT and Apache-2.0 both permit reuse with the licence notice preserved. **No source code was copied.** The design (loop structure, k-step window loss, termination BCE, frozen world model in the actor phase, survival-weighted return + critic bootstrap, element-wise gradient clipping) was re-implemented independently from the described algorithm; the structure is credited to the sources above. If code is ever copied, the Apache-2.0 NOTICE / MIT copyright lines must be added. |
| Technical | Not drop-in: both repos need **mjlab / MuJoCo Warp** (CUDA simulator), different robots (G1, ARX5), different tasks/rewards, and Reference A needs Python 3.12+ while the target workstation default is Python 3.10 (Ubuntu 22.04). The OpenArm scene is built with MjSpec from the upstream XML and has **not** been ported to mjlab. |
| Therefore | Simulation = standard MuJoCo on CPU (existing `sim/lab_scene.py`, original `ActionAdapter` / `SafetyLimits`); neural networks on CUDA. A MuJoCo-Warp backend is **not implemented and not verified** (config validation rejects `env.sim_backend != mujoco_cpu`). |

## 4. Algorithmic differences (why it is "style", not "exact")
| Component | Reference (B) | This package `[mine]` |
|---|---|---|
| Task | humanoid whole-body motion tracking | OpenArm reaching / pushing / pick-and-place |
| Simulator / scale | MuJoCo Warp, 4096 GPU envs | MuJoCo CPU, <=64 envs (process-parallel) |
| World model | MLP 2048x3, goal encoder, shared decoder, latent 32 | residual MLP 512x3 on the executed action; `prev_ctrl` set structurally; no goal encoder / latent |
| Reward in imagination | analytic tracking reward from differentiable FK of all bodies | analytic task rewards; differentiable FK of the left-arm TCP only; optional learned reward head not used in the objective |
| Actor | encoder/decoder, 10-frame observation history, sensor noise | plain MLP, no history, action noise only |
| Reset sampling | return-based reset sampler | uniform randomisation (no OmniReset banks) |
| Termination | learned soft termination weights imagined return | same idea; terminal = box fell / non-finite state |
| State | proprioception + motion goal | **privileged simulator state** (box pose/velocity for push and pick-place); reaching actor uses no privileged input |
| Action | whole-body joint targets | left arm + gripper (8-D) through the existing clip -> slew adapter; right arm held; delta (default) or absolute parameterisation |
| Hyperparameters | Table A.5 (`[repo-claim]`) | smaller (hidden 512, batch 2048-4096) for a 24 GB card and CPU env throughput |
| Compute claims | 3 h on one H100 | **none**; nothing was trained at scale |

## 5. Existing-repository constraints respected
Original `control/action_adapter.py` + `safety_limits.py` are used by the environment for every step; a torch mirror (tested for exact parity) is used only inside imagination. Upstream `sim/openarm_mujoco` unchanged. No ACT checkpoint, dataset, G2a/G2b file, protected pose or ACT test seed 31-33 is read (`utils.assert_seeds_allowed`, static scan in tests). WM3-B is not reused: it is an offline, non-differentiable-through-ridge model on logged ACT windows, different interface.

## 6. Open items `[not verified]`
Full paper text and appendix; whether reference hyperparameters transfer to OpenArm; MuJoCo-Warp compatibility of the OpenArm scene; CUDA behaviour on the RTX 4090 (no GPU in this environment).
