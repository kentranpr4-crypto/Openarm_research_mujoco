#!/usr/bin/env python3
"""Export a SINGLE-FILE static snapshot of the running Studio (for hosts that cannot reach the Python server, e.g. a Claude Artifact).
It queries a running server (default http://127.0.0.1:8765), embeds every API answer and a few pre-rendered episodes, inlines the CSS and a dependency-ordered bundle of the ES modules, and replaces `fetch('/api/...')` by a read-only lookup.  Nothing is simulated.
Usage:  python3 app/tools/export_static.py --out app/dist/openarm_studio_static.html"""
import argparse, base64, io, json, re, sys, urllib.request, urllib.error
from pathlib import Path
APP = Path(__file__).resolve().parents[1]; FRONT = APP / "frontend"
def get(base, path, raw=False):
    try:
        with urllib.request.urlopen(base + path, timeout=180) as r: b = r.read(); return r.status, (b if raw else json.loads(b))
    except urllib.error.HTTPError as e: b = e.read(); return e.code, (b if raw else json.loads(b))
def recompress(jpg, q=72):
    from PIL import Image
    im = Image.open(io.BytesIO(jpg)); b = io.BytesIO(); im.save(b, "JPEG", quality=q, optimize=True); return b.getvalue()
def bundle():
    order = ["api", "ui", "arch", "sim", "main"]; out = []
    for m in order:
        src = (FRONT / "js" / f"{m}.js").read_text(); exports = []
        def imp(mo): names = mo.group(1).replace(" as ", ": "); return f"const {{{names}}} = __m_{mo.group(2)};"
        src = re.sub(r"^import \{([^}]*)\} from '\./(\w+)\.js';?\s*$", imp, src, flags=re.M)
        for mo in re.finditer(r"^export (?:async )?(?:function|const|let) (\w+)", src, flags=re.M): exports.append(mo.group(1))
        for mo in re.finditer(r"^export \{([^}]*)\};?\s*$", src, flags=re.M): exports += [x.strip() for x in mo.group(1).split(",") if x.strip()]
        src = re.sub(r"^export \{[^}]*\};?\s*$", "", src, flags=re.M); src = re.sub(r"^export (async )?(function|const|let) ", r"\1\2 ", src, flags=re.M)
        out.append(f"const __m_{m} = (() => {{\n{src}\nreturn {{{', '.join(sorted(set(exports)))}}};\n}})();")
    return "\n".join(out)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="http://127.0.0.1:8765"); ap.add_argument("--out", required=True); ap.add_argument("--max-mb", type=float, default=11.0); a = ap.parse_args(); base = a.base
    D = {}; frames = {}
    def api(path):
        c, d = get(base, path); assert c == 200, (path, c, d); D[path] = d; return d
    health = api("/api/health"); health["static_snapshot"] = True; health["render"] = dict(initialised=True, error=None, backend="pre-rendered", size=[480, 360]); D["/api/health"] = health
    api("/api/modes"); api("/api/research/status"); ex = api("/api/experiments")
    for e in ex: api(f"/api/experiments/{e['id']}")
    api("/api/architecture/current"); api("/api/architecture/future")
    eps = get(base, "/api/episodes")[1]; pick = [next(e for e in eps if e["id"] == "ds.demo_train.0"), next(e for e in eps if e["id"] == "pilot.A0_p00_r0"), next(e for e in eps if e["source"] == "wm1")]
    D["/api/episodes"] = pick
    for e in pick:
        api(f"/api/episodes/{e['id']}/metadata"); t = api(f"/api/episodes/{e['id']}/telemetry"); n = t["n_frames"]
        views = ["recorded:front"] if e["source"] == "dataset" else ["mujoco:overview"]
        for v in views:
            for k in range(n):
                c, b = get(base, f"/api/episodes/{e['id']}/frame/{k}?view={v}", raw=True); assert c == 200, (e["id"], k, v, b[:200]); frames[f"{e['id']}|{k}|{v}"] = base64.b64encode(recompress(b)).decode()
        print("embedded", e["id"], n, "frames", flush=True)
    c, b = get(base, "/api/scene/frame?view=overview", raw=True); assert c == 200; frames["scene|overview"] = base64.b64encode(recompress(b)).decode()
    payload = json.dumps(dict(api=D, frames=frames), separators=(",", ":"))
    css = (FRONT / "styles.css").read_text().replace("height:100vh}", "height:100%}"); js = bundle()
    shim = """
const __DATA = %s;
const __json = (o, s = 200) => new Response(JSON.stringify(o), { status: s, headers: { 'content-type': 'application/json' } });
window.fetch = async (input) => {
  const u = new URL(typeof input === 'string' ? input : input.url, location.href); const p = u.pathname; const q = u.searchParams;
  if (p === '/api/scene/frame') { const f = __DATA.frames['scene|' + (q.get('view') || 'overview')]; return f ? __img(f) : __json({ error: 'render_unavailable', reason: 'static snapshot: only the overview scene view is embedded; start the real server for orbit and other cameras' }, 503); }
  let m = p.match(/^\\/api\\/episodes\\/([^/]+)\\/frame\\/(\\d+)$/);
  if (m) { const f = __DATA.frames[decodeURIComponent(m[1]) + '|' + m[2] + '|' + q.get('view')]; return f ? __img(f) : __json({ error: 'render_unavailable', reason: 'static snapshot: this view is not embedded for this episode (embedded: dataset episode = recorded front camera; Stage-1 and WM-1 episodes = MuJoCo overview)' }, 503); }
  if (/^\\/api\\/episodes\\/[^/]+\\/fk\\//.test(p)) return __json({ error: 'unavailable', reason: 'static snapshot: forward kinematics are computed by the live server' }, 404);
  const hit = __DATA.api[p]; if (hit !== undefined) return __json(hit); return __json({ error: 'unavailable', reason: 'static snapshot: endpoint not embedded: ' + p }, 404);
};
function __img(b64) { const bin = atob(b64); const a = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i); return new Response(new Blob([a], { type: 'image/jpeg' }), { status: 200, headers: { 'content-type': 'image/jpeg' } }); }
""" % payload
    html = f"""<title>OpenArm Research Studio</title>
<style>
{css}
</style>
<div id="app" data-view="overview">
  <aside id="nav"><div class="brand"><span class="logo">&#9672;</span><div><b>OpenArm</b><small>Research Studio</small></div></div><nav id="navlinks"></nav><div class="navfoot"><span class="pill ro">STATIC SNAPSHOT</span><small id="navhead"></small></div></aside>
  <header id="top"><div class="tcell"><label>Project</label><b>OpenArm v1 &middot; MuJoCo &middot; ACT &middot; World Model</b></div><div class="tcell"><label>Simulation</label><b id="t-sim">checking&hellip;</b></div><div class="tcell"><label>Active experiment</label><b id="t-exp">&mdash;</b></div><div class="tcell"><label>Model / checkpoint</label><b id="t-ckpt">&mdash;</b></div><div class="tcell"><label>Evidence status</label><span id="t-ev">&mdash;</span></div><div class="tcell"><label>Data source</label><b id="t-src">&mdash;</b></div></header>
  <main id="main"></main><aside id="inspector"></aside><footer id="timeline"><canvas id="tl" height="120"></canvas></footer>
</div>
<script>
{shim}
{js}
</script>
"""
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(html); mb = len(html.encode()) / 2**20; print(f"wrote {a.out}: {mb:.2f} MB", flush=True); assert mb < a.max_mb, "snapshot too large"
if __name__ == "__main__": main()
