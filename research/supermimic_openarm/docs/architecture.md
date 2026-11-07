# Architecture (SuperMimic-style, OpenArm)

```mermaid
flowchart LR
  subgraph SIM["CPU MuJoCo (envs/): sim/lab_scene.py, UNMODIFIED upstream XML"]
    ENV["OpenArmTaskEnv x N (sync | spawn-subproc)"]
    AD["original ActionAdapter.to_ctrl\nclip ctrlrange -> slew limit"]
  end
  subgraph CUDA["GPU (torch)"]
    ACT["Actor MLP\nobs -> tanh a in [-1,1]^8"]
    WM["World model f(s, executed a)\nresidual dynamics, prev_ctrl := a\n+ termination head, contact head, optional reward head"]
    CR["Critic V(s, goal)"]
    FK["TorchFK: q -> TCP"]
    TAD["TorchAdapter (parity-tested mirror)"]
  end
  RB[("Replay ring buffer\n(state, EXECUTED ctrl, next_state,\nreward, term, trunc, contact)")]
  ENV -- "state, goal" --> ACT
  ACT -- "a + noise -> target" --> AD --> ENV
  ENV -- "transitions" --> RB
  RB -- "k-step windows (never cross episodes)" --> WM
  RB -- "start states" --> IMG
  subgraph IMG["Imagination (H = 8 | 16 | 32), WM + critic FROZEN"]
    direction LR
    S0["s0"] --> A1["actor"] --> TAD --> F["f_w"] --> R["analytic reward(s', FK)"] --> J["J = sum g^t S_t r_t + g^H S_H V(s_H)"]
  end
  J -- "backprop through dynamics (BPTT)" --> ACT
  IMG -- "lambda-returns" --> CR
```

Training loop per iteration: **collect** (policy + noise; warmup random) -> **fit world model** (k-step autoregressive window loss on executed actions + termination BCE + optional contact/reward) -> **imagine** (actor update with finite-gradient and element-wise clipping; WM and critic frozen) -> **critic** fit to lambda-returns -> periodic **evaluation** on held-out goal seeds -> **checkpoint** (latest/best + manifest sha256).

State vector (34, PRIVILEGED): `q(8) dq(8) prev_ctrl(8) box_pos(3) box_quat(4) box_vel(3)`. Reaching actor input = proprioception + FK TCP + goal (deployable); pushing / pick-place actor input additionally contains the box state (needs perception on a real robot).

Studio integration: `studio_adapter.py` (stdlib + numpy, read-only) -> optional `/api/supermimic/*` hook (`studio/app_hook.patch`, not applied).
