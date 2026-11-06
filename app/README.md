# OpenArm Research Studio

A **read-only** research visualisation and experiment-inspection application for the OpenArm v1 + MuJoCo + ACT + World Model project.
It does **not** run experiments, policies, training or simulator rollouts; it only reads saved artifacts.

## What it shows
| Workspace | Content | Data |
|---|---|---|
| Overview | research stages with data-derived status labels | `/api/research/status` |
| Simulation | **Mode A** recorded replay (824 dataset episodes with stored camera images, 120 Stage-1 pilot trajectories, WM-1 episodes) and **Mode B** MuJoCo scene view with orbit. **Mode C** (live simulation) is disabled | saved `npz` files + the repository's own `sim/lab_scene.py` model |
| Experiments | ACT V1, WM3-B, Stage 1, branching replay, G2a, G2b (design) with metrics, provenance, validity | read from `outputs/**.json` when a card is opened |
| ACT Policy / World Model | detail views, calibration caveat, error-vs-horizon chart | `outputs/audit_v1`, `outputs/e6b`, `outputs/indep` |
| Architecture | node-and-edge explorer with three separate layers: CURRENT IMPLEMENTATION, HISTORICAL (Stage 1), G2a/G2b grasp correction | `app/data/architecture_current.json` + file-existence checks |
| Future WAM | design-only lab: flow, model inspector, candidate cards, synthetic-illustration timeline, Sim2Real readiness | `app/data/architecture_future.json` |
| Research Status | labels, blockers (confirmed in the design documents), modes | derived |

Status labels (IMPLEMENTED / VALIDATED / EXPERIMENTAL / DESIGN ONLY / NOT TRAINED / NOT EVALUATED) are **computed**: from file existence, from the artifacts' own `label` / `status` fields and from gate verdicts (e.g. WM3-B is not shown as VALIDATED because gate G3 FAILs in the saved independent evaluation). G2a's grasp-quality outcomes are invalidated and are not displayed as a success metric. G2b has no results and shows none.

## Quick start on Ubuntu 22.04 (no outputs/ needed)
Needs Python 3.10+ (the Ubuntu 22.04 default), `git`; the dashboard itself needs only `numpy` and `Pillow` (about 40 MB RAM).
```bash
sudo apt update && sudo apt install -y git python3 python3-venv python3-pip
git clone https://github.com/kentranpr4-crypto/openarm_research_mujoco.git && cd openarm_research_mujoco
git checkout main-f9g0uh                                   # branch that contains app/
python3 -m venv .venv-studio && . .venv-studio/bin/activate
pip install -r app/requirements.txt                         # numpy + Pillow only
app/run_studio.sh --no-render                               # or: python -m app.backend.server --no-render
# open http://localhost:8765
```
`outputs/` (datasets, checkpoints, logs, about 12 GB) is **not** in git. Without it the app starts, shows a "DATA NOTE" banner, and the **Architecture explorer, Future-WAM design lab, Research Status** work. To see recorded episodes and historical results, use the **demo bundle** (about 5 MB, built from existing saved artifacts only, SHA-256 manifest included):
```bash
tar -xzf studio_demo_bundle.tar.gz                          # gives ./studio_demo_bundle/
app/run_studio.sh --data-root studio_demo_bundle --no-render
```
Bundle content: 3 recorded episodes (`ds.demo_train.0` with 3 camera streams, `pilot.A0_p00_r0`, `wm1.p00_v00`), the historical result JSONs of ACT V1, WM3-B, Stage 1, branching replay and G2a, and `MANIFEST.json` (per-file SHA-256 + original source path). It contains **no** checkpoint, weights, training data beyond one episode, pending G2b report or pose file. Experiment cards whose source file is not in the bundle show `UNAVAILABLE` with the reason (for example the G2b integration gates).
Rebuild it on a research machine that has `outputs/`: `python -m app.tools.make_demo_bundle --out outputs/studio_demo_bundle --tar` (refuses to overwrite).

### Options
| Option | Env var | Meaning |
|---|---|---|
| `--port N` | `OPENARM_STUDIO_PORT` | default 8765 |
| `--host H` | `OPENARM_STUDIO_HOST` | default 127.0.0.1 (local only) |
| `--data-root DIR` | `OPENARM_STUDIO_DATA_ROOT` | read-only artifact directory (same layout as the repo, e.g. `DIR/outputs/...`), searched before the checkout |
| `--bundle-only` | `OPENARM_STUDIO_EXCLUSIVE=1` | read `outputs/` only from `--data-root` (preview the bundle on a research machine) |
| `--no-render` | `OPENARM_STUDIO_NO_RENDER=1` | never create a MuJoCo scene |

### Optional: MuJoCo rendering (Mode B, Stage-1 / WM-1 trajectories, free orbit)
```bash
pip install -r app/requirements-render.txt                  # mujoco>=3.6
sudo apt install -y libosmesa6 libgl1 libegl1               # software (CPU) OpenGL
git clone https://github.com/enactic/openarm_mujoco.git sim/openarm_mujoco && git -C sim/openarm_mujoco checkout a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c   # pinned upstream model (same as sim/setup.sh); never modified
app/run_studio.sh --data-root studio_demo_bundle            # render on demand
```
The scene is created **lazily** on the first MuJoCo frame (about 12 CPU-seconds), later frames about 0.25 s (480x360, `mj_forward` only). `MUJOCO_GL` defaults to `osmesa`.

### Troubleshooting
| Symptom | Fix |
|---|---|
| `render unavailable` / `mujoco` import error | Use `--no-render`, or install `requirements-render.txt` and the upstream model above. The rest of the UI keeps working. |
| `OSMesa` / `libOSMesa.so` not found | `sudo apt install libosmesa6` and keep `MUJOCO_GL=osmesa` (default). |
| GPU/desktop available and faster rendering wanted | `MUJOCO_GL=egl app/run_studio.sh ...` (needs EGL drivers); if it fails go back to `osmesa`. |
| `glfw` / `GLXBadContext` / black frames | Never use `MUJOCO_GL=glfw` for this headless server; use `osmesa` or `egl`. |
| Very slow first frame | Normal with software rendering (about 12 CPU-s once); subsequent frames are cached. |
| `Address already in use` | `app/run_studio.sh --port 8766` (or `OPENARM_STUDIO_PORT`). |
| Empty episode list + DATA NOTE banner | `outputs/` is absent: pass `--data-root <bundle>`. |
| Browser cannot reach the app from another machine | The server binds 127.0.0.1 on purpose (read-only, unauthenticated). Use an SSH tunnel: `ssh -L 8765:localhost:8765 host`. |

## API (GET only; every other method returns 405)
`/api/health`, `/api/modes`, `/api/research/status`, `/api/experiments`, `/api/experiments/{id}`, `/api/episodes`, `/api/episodes/{id}/metadata`, `/api/episodes/{id}/telemetry`, `/api/episodes/{id}/frame/{k}?view=recorded:<cam>|mujoco:<overview|front|wrist_left|wrist_right|free>&az=&el=&dist=`, `/api/episodes/{id}/fk/{k}`, `/api/scene/frame?view=...`, `/api/render/probe`, `/api/architecture/current`, `/api/architecture/future`.
Security: IDs are validated and mapped to files by the catalog (never used as paths); artifact reads go through `app/backend/security.py` (relative, allow-listed top directories, no `..`, no symlink escape); there are no file writes, no subprocesses, no eval/exec, no simulator stepping.

## Data semantics and limits (read before trusting a plot)
- **Alignment:** `state[t]` is the observation before `action[t]`; the action's effect first appears in `state[t+1]` (tested: the measured gripper follows the command with a one-row lag).
- **Reconstruction:** MuJoCo frames are rebuilt from the recorded 16-D joint state and box pose with `mj_forward`. The forward-kinematics TCP matches the recorded TCP to 0.07 mm on checked frames. Stage-1 pilot files have **no box orientation**, so the box is drawn unrotated (shown in the limitations line).
- **Events** drawn on the timeline are a **viewer heuristic** on recorded signals (close command, 3 cm lift, re-opening); they are not outcome labels and not the G2b labeller. Dataset episodes also show the **recorded** scripted-expert phases.
- **Contacts:** WM-1 files record finger / palm / table / tray forces; pilot files only the documented aggregate columns; dataset files none. Missing data is displayed as "not recorded", never filled in.
- **Metrics** come from saved artifacts and carry their source path; historical report values (ACT V1 30/30 and 18/30; WM3-B 3.20 mm vs WM2-C0 3.83 mm) are additionally confirmed against the report text.
- **Illustrative only:** the Future WAM timeline is stamped "SYNTHETIC ILLUSTRATION - NOT MODEL OUTPUT"; its tensor shapes are proposals. No fake predictions exist anywhere.

## Tests
```bash
app/tests/run_all.sh        # python syntax, backend/API + portability tests, node syntax, frontend tests, eslint (needs node >= 18 for the frontend tests)
```
Covered: API schema, artifact discovery and provenance, missing/corrupted trajectories, frame indexing, action-observation alignment, status labelling, history-vs-future separation, absence of simulator stepping / policy loading / writes in the backend, read-only protection, frontend structure (no external scripts, no mutating requests). There is no TypeScript layer; the "type check" is `node --check` + eslint (`no-undef`).

## Layout
`app/backend/` (server, security, catalog, render, experiments) - `app/data/` (architecture definitions) - `app/frontend/` (index.html, styles.css, js modules) - `app/tests/` - `app/docs/screenshots/`.
