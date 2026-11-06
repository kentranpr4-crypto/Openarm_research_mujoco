"""Build the small OpenArm Research Studio demo bundle from EXISTING saved artifacts (nothing is generated, simulated or evaluated).
The bundle mirrors the repository-relative layout (outputs/...), so the app uses it directly:  python -m app.backend.server --data-root <bundle dir>
Only an explicit allow-list is copied.  No checkpoint / weights / training artifact / pending G2b report / sealed or reserved pose is ever included.
Usage (research checkout that has outputs/):  python -m app.tools.make_demo_bundle --out outputs/studio_demo_bundle [--tar]"""
import argparse, hashlib, json, sys, tarfile, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
EPISODE_DATASET = ("demo_train", 0)                                   # scripted-expert demonstration with stored camera frames (training region)
PILOT_EPISODE = "A0_p00_r0"                                           # Stage-1 pilot trajectory (development pose 0, arm A0, repeat 0)
WM1_EPISODE = "p00_v00"
COPY = [   # (relative path, role)
    ("outputs/datasets/demo_train/episode_%06d.npz" % EPISODE_DATASET[1], "recorded episode (state, action, 3 camera streams)"),
    ("outputs/datasets/demo_train/episode_%06d.json" % EPISODE_DATASET[1], "episode sidecar"),
    (f"outputs/closed_loop_pilot/results/traj/{PILOT_EPISODE}.npz", "recorded Stage-1 pilot trajectory (state, box, TCP, contact rows)"),
    (f"outputs/wm1/pilot/episodes/{WM1_EPISODE}.npz", "recorded WM-1 pilot episode (state, box, contacts)"),
    ("outputs/audit_v1/audit_results.json", "historical result: ACT V1 audit"), ("outputs/e6b/stage1_results.json", "historical result: Stage 0/1 boundary"),
    ("outputs/indep/evaluation_independent.json", "historical result: WM3-B independent evaluation"), ("outputs/wm3/evaluation_v2.json", "historical result: WM3 development evaluation"),
    ("outputs/wm3/model_config_wm3b.json", "frozen WM3-B model spec (config only, no weights)"), ("outputs/checkpoints/act_full_v1/train_info.json", "ACT V1 training metadata (no weights)"),
    ("outputs/closed_loop_pilot/protocol_freeze.json", "Stage-1 protocol freeze"), ("outputs/closed_loop_pilot/results/pilot_analysis.json", "historical result: Stage-1 analysis"), ("outputs/closed_loop_pilot/results/pilot_results.json", "Stage-1 per-episode outcomes"),
    ("outputs/branching_replay/protocol_freeze.json", "branching replay protocol freeze"), ("outputs/branching_replay/results/branching_analysis.json", "historical result: branching replay"),
    ("outputs/g2a_act_corrections/protocol_freeze.json", "G2a protocol freeze"), ("outputs/g2a_act_corrections/results/g2a_analysis.json", "historical result: G2a (validity only)")]
FORBIDDEN_SUFFIX = (".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".engine", ".mp4")
FORBIDDEN_PARTS = ("g2b_", "g2b/", "reserve", "sealed", "checkpoints/act_full_v1/model", "pose_plan")
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
def build(out, tar=False):
    out = Path(out).resolve(); entries = []
    if out.exists() and any(out.iterdir()): raise SystemExit(f"refusing to overwrite a non-empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    def put(rel, role, data=None, derived_from=None, note=None):
        low = rel.lower()
        assert not low.endswith(FORBIDDEN_SUFFIX) and not any(x in low for x in FORBIDDEN_PARTS), f"forbidden artifact: {rel}"
        src = REPO / (derived_from or rel); dst = out / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        if data is None: dst.write_bytes(src.read_bytes())
        else: dst.write_bytes(data)
        e = dict(path=rel, role=role, bytes=dst.stat().st_size, sha256=sha256(dst), original_source=(derived_from or rel), original_sha256=sha256(src), identical_to_original=(sha256(dst) == sha256(src)))
        if note: e["note"] = note
        entries.append(e)
    for rel, role in COPY: put(rel, role)
    # episodes.jsonl is filtered to the single bundled episode (the original lists every episode of the dataset)
    name, idx = EPISODE_DATASET; src_rel = f"outputs/datasets/{name}/episodes.jsonl"
    lines = [ln for ln in (REPO / src_rel).read_text().splitlines() if ln.strip() and json.loads(ln).get("episode_index") == idx]
    put(src_rel, "episode index (filtered)", data=("\n".join(lines) + "\n").encode(), derived_from=src_rel, note=f"derived: only the line of episode_index {idx} is kept")
    man = dict(bundle="openarm_studio_demo_bundle", format=1, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), purpose="Minimal read-only data for OpenArm Research Studio. Only existing saved artifacts; no new rollouts, training or evaluation.",
               episodes=[f"ds.{EPISODE_DATASET[0]}.{EPISODE_DATASET[1]}", f"pilot.{PILOT_EPISODE}", f"wm1.{WM1_EPISODE}"], excluded=["checkpoints / model weights (*.pt)", "datasets other than one episode", "outputs/g2b_* (pending, unreviewed G2b reports)", "sealed-reserve or prospective pose files (not generated)", "videos"],
               files=entries, total_bytes=sum(e["bytes"] for e in entries))
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
    (out / "README.txt").write_text("OpenArm Research Studio demo bundle (read-only data).\nUse:  python -m app.backend.server --data-root <this directory>\nSee MANIFEST.json for per-file SHA-256 and original source paths.\nHistorical results only: nothing here is a new evaluation.\n")
    res = dict(dir=str(out), n_files=len(entries) + 2, bytes=man["total_bytes"])
    if tar:
        tp = out.with_suffix(".tar.gz") if out.suffix == "" else out.parent / (out.name + ".tar.gz")
        with tarfile.open(tp, "w:gz") as t: t.add(out, arcname=out.name)
        res.update(tar=str(tp), tar_bytes=tp.stat().st_size, tar_sha256=sha256(tp))
    return res
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(REPO / "outputs" / "studio_demo_bundle")); ap.add_argument("--tar", action="store_true"); a = ap.parse_args()
    print(json.dumps(build(a.out, a.tar), indent=1))
