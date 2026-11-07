"""Optional, READ-ONLY adapter exposing SuperMimic-OpenArm runs to OpenArm Research Studio.
Standard library + numpy only: it never imports torch or mujoco, never loads a checkpoint (it reads json / jsonl / npz files and the checkpoint manifest).
`handle(path, query)` -> (http_status, json_obj) serves the routes under /api/supermimic/ ; see studio/app_hook.patch for the 6-line (UNAPPLIED) hook in app/backend/server.py.
The app keeps working when this package, its runs or its dependencies are absent: the hook returns a 'not installed' message instead of an error."""
import json, re
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; DEFAULT_ROOT = "outputs/supermimic_openarm"
SAFE = re.compile(r"^[A-Za-z0-9_.\-]{1,96}$")
TASKS = {"reaching": "P0 - reach randomised TCP targets (no object contact)", "pushing": "P1 - push a block into a goal region (contact-aware)", "pick_place": "P2 - grasp, lift, carry, release and settle in the tray"}
def _root(root=None):
    r = Path(root or DEFAULT_ROOT); return r if r.is_absolute() else REPO / r
def _safe(x):
    if not isinstance(x, str) or not SAFE.match(x): raise KeyError(f"invalid identifier: {x!r}")
    return x
def _json(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None
def _run_dirs(root):
    r = _root(root)
    if not r.is_dir(): return []
    return sorted(p for p in r.glob("*/seed*") if p.is_dir() and not p.parent.name.startswith(("benchmarks", "_")))
def run_id(d): return f"{d.parent.name}.{d.name}"
def _resolve(rid, root):
    for d in _run_dirs(root):
        if run_id(d) == _safe(rid): return d
    raise KeyError(f"unknown run: {rid}")
def list_runs(root=None):
    out = []
    for d in _run_dirs(root):
        cfg = _json(d / "config.resolved.json") or {}; man = _json(d / "checkpoints" / "manifest.json") or {}; last = None
        for ln in (d / "metrics.jsonl").read_text().splitlines()[-1:] if (d / "metrics.jsonl").is_file() else []:
            try: last = json.loads(ln)
            except Exception: pass
        evals = sorted((d / "eval").glob("eval_it*.json")) if (d / "eval").is_dir() else []; ev = _json(evals[-1]) if evals else (_json(d / "eval" / "eval_final.json") if (d / "eval" / "eval_final.json").is_file() else None)
        out.append(dict(id=run_id(d), name=d.parent.name, seed=cfg.get("experiment", {}).get("seed"), algo=cfg.get("experiment", {}).get("algo"), task=cfg.get("env", {}).get("task"), completed=(d / "COMPLETED").is_file(), iteration=(last or {}).get("iteration"), env_steps=(last or {}).get("env_steps"),
                        checkpoints={k: dict(bytes=v.get("bytes"), sha256=v.get("sha256"), iteration=v.get("iteration")) for k, v in man.items() if k.endswith(".pt")}, last_eval=(dict(success_rate=ev.get("success_rate"), mean_return=ev.get("mean_return"), n_episodes=ev.get("n_episodes")) if ev else None),
                        privileged_state=True, label=("TRAINED (completed run)" if (d / "COMPLETED").is_file() else "PARTIAL / INTERRUPTED")))
    return out
def tasks(root=None):
    runs = list_runs(root); res = []
    for k, desc in TASKS.items():
        rr = [r for r in runs if r["task"] == k]; done = [r for r in rr if r["completed"]]
        res.append(dict(id=k, description=desc, implemented=True, runs=len(rr), completed_runs=len(done), labels=["IMPLEMENTED"] + (["TRAINED (smoke/pilot runs exist)"] if done else ["NOT TRAINED"]) + ([] if any(r["last_eval"] for r in rr) else ["NOT EVALUATED"]),
                        note="readiness is NOT claimed: see the validation report"))
    return res
def curves(rid, root=None):
    d = _resolve(rid, root); rows = []
    p = d / "metrics.jsonl"
    if p.is_file():
        for ln in p.read_text().splitlines():
            try: rows.append(json.loads(ln))
            except Exception: pass
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and k not in ("iteration",)}); return dict(id=rid, iterations=[r.get("iteration") for r in rows], series={k: [r.get(k) for r in rows] for k in keys})
def evaluations(rid, root=None):
    d = _resolve(rid, root); out = []
    for p in sorted((d / "eval").glob("*.json")) if (d / "eval").is_dir() else []:
        j = _json(p)
        if j: out.append(dict(file=p.name, **{k: v for k, v in j.items() if k != "episodes"}, n_episode_rows=len(j.get("episodes", []))))
    return dict(id=rid, evaluations=out, note="privileged-state simulation evaluation on held-out goals; not Sim2Real")
def trajectories(rid, root=None):
    d = _resolve(rid, root); t = d / "trajectories"; return dict(id=rid, files=[p.name for p in sorted(t.glob("*.npz"))] if t.is_dir() else [])
def trajectory(rid, name, root=None):
    d = _resolve(rid, root); name = _safe(name); p = d / "trajectories" / (name if name.endswith(".npz") else name + ".npz")
    if not p.is_file(): raise KeyError(f"trajectory not found: {name}")
    z = np.load(p, allow_pickle=False); S = z["states"]; out = {k: (z[k].tolist() if z[k].dtype == bool else np.round(z[k].astype(float), 5).tolist()) for k in z.files if k in ("states", "executed", "tcp", "reward", "wm_one_step", "wm_rollout", "goal", "contact", "success")}
    if "wm_one_step" in z.files:
        out["one_step_error"] = np.round(np.linalg.norm(z["wm_one_step"] - S[1:], axis=1), 6).tolist(); out["rollout_error"] = np.round(np.linalg.norm(z["wm_rollout"] - S[1:], axis=1), 6).tolist()
    out.update(id=rid, name=p.name, n_steps=int(len(S) - 1), note="actual vs predicted next states: wm_one_step = f(actual state_t, executed action_t); wm_rollout = open-loop imagined rollout from the first state with the executed actions (state vector layout: q(8) dq(8) prev_ctrl(8) box_pos(3) box_quat(4) box_vel(3))"); return out
def handle(path, query=None, root=None):
    """path without the '/api/supermimic/' prefix"""
    parts = [x for x in path.strip("/").split("/") if x]
    try:
        if not parts or parts == ["status"]: return 200, dict(installed=True, root=str(_root(root)), tasks=tasks(root), runs=len(list_runs(root)), note="optional SuperMimic-OpenArm integration; read-only")
        if parts == ["tasks"]: return 200, tasks(root)
        if parts == ["runs"]: return 200, list_runs(root)
        if len(parts) == 3 and parts[0] == "runs":
            fn = dict(curves=curves, evaluations=evaluations, trajectories=trajectories).get(parts[2])
            if fn: return 200, fn(parts[1], root)
        if len(parts) == 4 and parts[0] == "runs" and parts[2] == "trajectory": return 200, trajectory(parts[1], parts[3], root)
    except KeyError as e: return 404, dict(error="unavailable", reason=str(e))
    return 404, dict(error="not_found", reason=f"unknown route: /api/supermimic/{path}")
