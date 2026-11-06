"""OpenArm Research Studio - read-only HTTP API + static frontend (Python standard library only).
Run (from the repository root):  python -m app.backend.server [--port 8765] [--data-root DIR] [--no-render]
Only GET is accepted.  No endpoint writes files, runs a policy, steps the simulator, executes a command or accepts a file path."""
import argparse, io, json, mimetypes, os, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from . import security as S, catalog as C, experiments as X
from .render import RENDERER, RenderUnavailable
APP = Path(__file__).resolve().parents[1]; FRONT = APP / "frontend"; DATA = APP / "data"; VERSION = "0.1.0"
def _json_file(name): return json.loads((DATA / name).read_text())
def head_commit():
    try:
        h = (S.REPO / ".git" / "HEAD").read_text().strip()
        if h.startswith("ref:"): return (S.REPO / ".git" / h.split(" ", 1)[1]).read_text().strip()[:12]
        return h[:12]
    except Exception: return None
def modes():
    return [dict(id="A", name="Recorded replay", enabled=True, text="Replays saved trajectories; the scene is reconstructed with mj_forward only."), dict(id="B", name="MuJoCo scene view", enabled=True, text="Loads the OpenArm v1 lab scene read-only; no evaluation experiment is run."),
            dict(id="C", name="Live simulation preview", enabled=False, text="DISABLED: requires a separately authorised, safe simulation entry point. Not implemented in this application.")]
def architecture(which):
    d = _json_file("architecture_current.json" if which == "current" else "architecture_future.json")
    for n in d["nodes"]:
        n["files_status"] = [dict(path=f, exists=S.exists(f)) for f in n.get("files", [])]
    d["generated_checks"] = dict(all_files_exist=all(f["exists"] for n in d["nodes"] for f in n["files_status"])); return d
def research_status():
    ex = X.summary(); r3 = "docs/ai-native/tasks/2026-11-05-g2b-outcome-contract-r3-freeze-candidate.md"; ann = "docs/ai-native/tasks/2026-11-05-g2b-blind-annotation-plan.md"
    wam_ckpt = S.resolve("outputs/checkpoints").is_dir() and any(p.name.lower().startswith(("wam", "wm_grasp")) for p in S.resolve("outputs/checkpoints").iterdir())
    wam = dict(id="wam", title="Grasp-focused World Action Model (proposed)", labels=[x for x in ("DESIGN ONLY", "NOT TRAINED" if not wam_ckpt else None, "NOT EVALUATED") if x], validity="DESIGN_ONLY", purpose="Educational design preview; there are no trained artifacts.")
    blockers = [dict(id="annotator", text="Independent blind annotator and adjudicator are UNASSIGNED", verified_in_doc=(S.exists(ann) and S.doc_contains(ann, "UNASSIGNED")), doc_missing=not S.exists(ann), source=ann),
                dict(id="r3", text="Outcome Contract r3 is a freeze candidate, NOT FROZEN", verified_in_doc=(S.exists(r3) and S.doc_contains(r3, "NOT FROZEN")), doc_missing=not S.exists(r3), source=r3),
                dict(id="reserve", text="Sealed reserve poses and the 40 prospective development poses are not generated", verified_in_doc=S.exists("docs/ai-native/tasks/2026-11-05-g2b-compute-budget-and-pose-plan.md") and S.doc_contains("docs/ai-native/tasks/2026-11-05-g2b-compute-budget-and-pose-plan.md", "no pose is generated"), doc_missing=not S.exists("docs/ai-native/tasks/2026-11-05-g2b-compute-budget-and-pose-plan.md"), source="docs/ai-native/tasks/2026-11-05-g2b-compute-budget-and-pose-plan.md")]
    return dict(experiments=ex + [wam], blockers=blockers, modes=modes(), evidence_ladder=["DESIGN ONLY", "IMPLEMENTED", "EXPERIMENTAL", "VALIDATED"], rules=["Historical results are shown with their artifact provenance.", "No G2b treatment result exists.", "G2a grasp-quality outcomes are invalidated and not shown as success."], generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
def data_status():
    """what artifacts are reachable, with a human-readable hint when nothing is (a missing outputs/ directory is a normal state, not an error)"""
    try: n_ep = len(C.list_episodes())
    except Exception: n_ep = 0
    ex = [S.exists(p) for p in ("outputs/audit_v1/audit_results.json", "outputs/indep/evaluation_independent.json", "outputs/closed_loop_pilot/results/pilot_analysis.json", "outputs/branching_replay/results/branching_analysis.json")]
    present = sum(ex); msg = None
    if n_ep == 0 and present == 0: msg = "No research artifacts found (outputs/ is not in git). The architecture explorer and the future-design lab work; to see recorded episodes and historical results, download the demo bundle and start the server with --data-root <bundle directory> (see app/README.md)."
    elif n_ep == 0: msg = "Historical result files were found but no recorded episodes. Replay is empty."
    elif present < len(ex): msg = "Partial artifact set (demo bundle?): experiment cards whose source file is absent show UNAVAILABLE with the reason."
    return dict(data_roots=[str(r) for r in S.ROOTS], n_episodes=n_ep, result_files_present=present, result_files_checked=len(ex), message=msg)
def health():
    return dict(status="ok", app="OpenArm Research Studio", version=VERSION, read_only=True, repo_head=head_commit(), render=RENDERER.status(), modes=[dict(id=m["id"], enabled=m["enabled"]) for m in modes()], artifact_roots=list(S.ALLOWED_TOP), data=data_status())
class Handler(BaseHTTPRequestHandler):
    server_version = "OpenArmStudio/" + VERSION
    def log_message(self, *a): pass
    def _send(self, code, body, ctype, extra=None):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)
    def _json(self, obj, code=200): self._send(code, json.dumps(obj, default=float).encode(), "application/json; charset=utf-8", {"Cache-Control": "no-store"})
    def _err(self, code, error, reason): self._json(dict(error=error, reason=reason), code)
    def do_POST(self): self._err(405, "read_only", "this API is read-only: only GET is supported")
    do_PUT = do_DELETE = do_PATCH = do_POST
    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}; p = u.path
        try:
            if p.startswith("/api/"): return self._api(p[5:].strip("/"), q)
            return self._static(p)
        except S.ArtifactUnavailable as e: return self._err(404, "unavailable", str(e))
        except RenderUnavailable as e: return self._err(503, "render_unavailable", str(e))
        except Exception as e: return self._err(500, "internal_error", f"{type(e).__name__}: {e}")
    def _static(self, p):
        rel = "index.html" if p in ("", "/") else p.lstrip("/")
        if rel.startswith("static/"): rel = rel[7:]
        f = (FRONT / rel).resolve()
        if FRONT.resolve() not in f.parents or not f.is_file(): return self._err(404, "not_found", "no such static file")
        self._send(200, f.read_bytes(), mimetypes.guess_type(str(f))[0] or "application/octet-stream", {"Cache-Control": "no-cache"})
    def _api(self, r, q):
        parts = r.split("/")
        if r == "health": return self._json(health())
        if r == "modes": return self._json(modes())
        if r == "research/status": return self._json(research_status())
        if r == "experiments": return self._json(X.summary())
        if len(parts) == 2 and parts[0] == "experiments": return self._json(X.detail(parts[1]))
        if r == "episodes": return self._json(C.list_episodes())
        if len(parts) == 3 and parts[0] == "episodes" and parts[2] == "metadata": return self._json(C.episode_meta(parts[1]))
        if len(parts) == 3 and parts[0] == "episodes" and parts[2] == "telemetry": return self._json(C.telemetry(parts[1]))
        if len(parts) == 4 and parts[0] == "episodes" and parts[2] == "frame": return self._frame(parts[1], parts[3], q)
        if len(parts) == 4 and parts[0] == "episodes" and parts[2] == "fk": return self._fk(parts[1], parts[3])
        if r == "scene/frame": return self._scene(q)
        if r == "render/probe": RENDERER._init(); return self._json(RENDERER.status())
        if r == "architecture/current": return self._json(architecture("current"))
        if r == "architecture/future": return self._json(architecture("future"))
        return self._err(404, "not_found", f"unknown endpoint: /api/{r}")
    def _fparams(self, q):
        try: return dict(az=float(q.get("az", 140)), el=float(q.get("el", -25)), dist=float(q.get("dist", 1.9)))
        except ValueError: raise S.ArtifactUnavailable("az / el / dist must be numbers")
    def _frame(self, eid, k, q):
        if not k.isdigit(): raise S.ArtifactUnavailable("frame id must be a non-negative integer")
        k = int(k); view = q.get("view", "mujoco:overview"); src, _, name = view.partition(":")
        if src == "recorded":
            from PIL import Image
            img = C.recorded_frame(eid, k, name); b = io.BytesIO(); Image.fromarray(img).save(b, "JPEG", quality=90); return self._send(200, b.getvalue(), "image/jpeg", {"Cache-Control": "max-age=300"})
        if src == "mujoco":
            st, box, quat = C.render_inputs(eid, k); jpg = RENDERER.render(st, box, quat, view=name, key=(eid, k), **self._fparams(q)); return self._send(200, jpg, "image/jpeg", {"Cache-Control": "max-age=300"})
        raise S.ArtifactUnavailable("view must be recorded:<camera> or mujoco:<view>")
    def _fk(self, eid, k):
        if not k.isdigit(): raise S.ArtifactUnavailable("frame id must be a non-negative integer")
        st, box, quat = C.render_inputs(eid, int(k)); pos, qt = RENDERER.tcp_fk(st); rec = None
        try:
            tel = C.telemetry(eid); rec = tel["tcp_pos"][int(k)] if tel.get("tcp_pos") else None
        except S.ArtifactUnavailable: pass
        import math
        return self._json(dict(frame=int(k), tcp_pos_fk=pos, tcp_quat_fk_wxyz=qt, tcp_pos_recorded=rec, fk_vs_recorded_m=(math.dist(pos, rec) if rec else None), note="forward kinematics of the recorded joint state in the repository's own MuJoCo model (mj_forward only)"))
    def _scene(self, q):
        name = q.get("view", "overview"); st, box, quat = RENDERER.home_state(); jpg = RENDERER.render(st, box, quat, view=name, key=("scene",), **self._fparams(q)); return self._send(200, jpg, "image/jpeg", {"Cache-Control": "max-age=300"})
def make_server(host="127.0.0.1", port=8765): return ThreadingHTTPServer((host, port), Handler)
def main():
    ap = argparse.ArgumentParser(prog="python -m app.backend.server", description="OpenArm Research Studio (read-only research viewer)")
    ap.add_argument("--host", default=os.environ.get("OPENARM_STUDIO_HOST", "127.0.0.1")); ap.add_argument("--port", type=int, default=int(os.environ.get("OPENARM_STUDIO_PORT", "8765")))
    ap.add_argument("--data-root", default=None, help="read-only directory with the artifact layout (outputs/..., e.g. the unpacked demo bundle); searched before the repository. Env: OPENARM_STUDIO_DATA_ROOT")
    ap.add_argument("--bundle-only", action="store_true", help="read outputs/ ONLY from --data-root (ignore the checkout's own outputs/); use it to preview the demo bundle on a research machine. Env: OPENARM_STUDIO_EXCLUSIVE=1")
    ap.add_argument("--no-render", action="store_true", help="never create the MuJoCo scene (recorded camera frames, telemetry, experiments and architecture still work). Env: OPENARM_STUDIO_NO_RENDER=1")
    a = ap.parse_args()
    if a.data_root: S.set_data_root(a.data_root, exclusive=a.bundle_only or os.environ.get("OPENARM_STUDIO_EXCLUSIVE") == "1")
    if a.no_render: os.environ["OPENARM_STUDIO_NO_RENDER"] = "1"
    srv = make_server(a.host, a.port); ds = data_status()
    print(f"OpenArm Research Studio on http://{'localhost' if a.host == '127.0.0.1' else a.host}:{a.port}  (read-only; MuJoCo scene is created lazily on the first render request; render={'OFF' if os.environ.get('OPENARM_STUDIO_NO_RENDER') == '1' else 'on demand'})", flush=True)
    print(f"artifact roots: {', '.join(ds['data_roots'])} | episodes: {ds['n_episodes']} | result files: {ds['result_files_present']}/{ds['result_files_checked']}", flush=True)
    if ds["message"]: print("NOTE: " + ds["message"], flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
if __name__ == "__main__": main()
