"""Portability tests: configurable artifact roots, missing outputs/, demo bundle, no cloud paths, no torch.  No simulator, no model.
Run from the repository root:  python -m unittest discover -s app/tests -v     (bundle tests run when the bundle exists: OPENARM_STUDIO_BUNDLE or outputs/studio_demo_bundle)"""
import os, sys, json, re, tempfile, threading, unittest, urllib.request, hashlib
from pathlib import Path
os.environ["OPENARM_STUDIO_NO_RENDER"] = "1"
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from app.backend import security as S, catalog as C, server as SV
BUNDLE = Path(os.environ.get("OPENARM_STUDIO_BUNDLE", ROOT / "outputs" / "studio_demo_bundle"))
def serve():
    h = SV.make_server("127.0.0.1", 0); threading.Thread(target=h.serve_forever, daemon=True).start(); return h
def get(h, path):
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{h.server_address[1]}{path}", timeout=60); return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read())
class Base(unittest.TestCase):
    def tearDown(self): S.set_data_root(None); C._pilot_meta.cache_clear(); C._load.cache_clear()
class TestEmptyRoot(Base):
    def test_missing_outputs_is_reported_not_a_crash(self):
        with tempfile.TemporaryDirectory() as d:
            S.set_data_root(d, exclusive=True)          # empty root, repository outputs/ ignored
            h = serve(); self.addCleanup(h.shutdown)
            c, hl = get(h, "/api/health"); self.assertEqual(c, 200); self.assertEqual(hl["data"]["n_episodes"], 0); self.assertIn("demo bundle", hl["data"]["message"])
            c, eps = get(h, "/api/episodes"); self.assertEqual((c, eps), (200, []))
            c, ex = get(h, "/api/experiments"); self.assertEqual(c, 200); self.assertEqual(len(ex), 6)
            for e in ex:
                c, x = get(h, "/api/experiments/" + e["id"]); self.assertEqual(c, 200)
                self.assertTrue(all(m["status"] in ("UNAVAILABLE", "NOT_EVALUATED") for m in x["metrics"] if m["id"] != "integration"), e["id"])
            c, st = get(h, "/api/research/status"); self.assertEqual(c, 200)
            c, a = get(h, "/api/architecture/future"); self.assertEqual(c, 200)
            c, e = get(h, "/api/episodes/ds.demo_train.0/telemetry"); self.assertEqual(c, 404); self.assertEqual(e["error"], "unavailable")
    def test_root_escape_blocked(self):
        for bad in ("../x", "outputs/../../etc/passwd", "/etc/passwd", "secret/x"):
            with self.assertRaises(S.ArtifactUnavailable): S.resolve(bad)
    def test_symlink_escape_blocked(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as out:
            (Path(d) / "outputs").mkdir(); (Path(out) / "s.json").write_text("{}"); os.symlink(out, Path(d) / "outputs" / "link")
            S.set_data_root(d)
            with self.assertRaises(S.ArtifactUnavailable): S.resolve("outputs/link/s.json")
class TestSourceHygiene(unittest.TestCase):
    def test_no_hardcoded_cloud_paths(self):
        bad = re.compile(r"/home/user|/tmp/claude|\.venvs/act|/root/")
        for p in (ROOT / "app").rglob("*"):
            if p.suffix in (".py", ".js", ".mjs", ".sh", ".md", ".html", ".css", ".json", ".txt") and "docs" not in p.parts and "dist" not in p.parts and "__pycache__" not in p.parts and p.name != "test_portable.py":
                self.assertIsNone(bad.search(p.read_text(errors="replace")), f"hard-coded path in {p}")
    def test_no_torch_or_checkpoint_loading(self):
        self.assertNotIn("torch", sys.modules); self.assertNotIn("mujoco", sys.modules, "MuJoCo must not be imported at start-up")
        for p in (ROOT / "app" / "backend").glob("*.py"): self.assertNotRegex(p.read_text(), r"\bimport torch\b|torch\.load|load_state_dict")
    def test_port_configurable(self):
        h = SV.make_server("127.0.0.1", 0); self.addCleanup(h.server_close); self.assertGreater(h.server_address[1], 0)
        src = (ROOT / "app/backend/server.py").read_text(); self.assertIn("OPENARM_STUDIO_PORT", src); self.assertIn('"8765"', src)
@unittest.skipUnless((BUNDLE / "MANIFEST.json").is_file(), "demo bundle not built")
class TestBundle(Base):
    def setUp(self): S.set_data_root(BUNDLE, exclusive=True); C._pilot_meta.cache_clear(); C._load.cache_clear()
    def test_manifest_hashes_and_exclusions(self):
        m = json.loads((BUNDLE / "MANIFEST.json").read_text()); self.assertEqual(m["total_bytes"], sum(f["bytes"] for f in m["files"]))
        for f in m["files"]:
            p = BUNDLE / f["path"]; self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), f["sha256"], f["path"]); self.assertTrue(f["original_source"].startswith("outputs/")); self.assertEqual(len(f["original_sha256"]), 64)
            self.assertFalse(f["path"].endswith((".pt", ".pth", ".ckpt", ".mp4", ".safetensors"))); self.assertNotIn("g2b", f["path"].lower())
        listed = {f["path"] for f in m["files"]} | {"MANIFEST.json", "README.txt"}; actual = {str(p.relative_to(BUNDLE)) for p in BUNDLE.rglob("*") if p.is_file()}; self.assertEqual(listed, actual, "unlisted or missing bundle files")
        self.assertLess(m["total_bytes"], 100 * 2**20)
    def test_episodes_and_alignment(self):
        eps = C.list_episodes(); self.assertEqual(sorted(e["id"] for e in eps), ["ds.demo_train.0", "pilot.A0_p00_r0", "wm1.p00_v00"])
        for e in eps:
            t = C.telemetry(e["id"]); n = t["n_frames"]; self.assertEqual(len(t["t"]), n); self.assertEqual(len(t["state"]), n); self.assertEqual(len(t["box_pos"]), n); self.assertEqual(len(t["tcp_pos"]), n)
            self.assertAlmostEqual(t["t"][1] - t["t"][0], C.DT)
            if t["action"] is not None: self.assertGreaterEqual(len(t["action"]), n - 1)
        d = C.telemetry("ds.demo_train.0"); self.assertTrue(d["has_images"])
        for cam in ("front", "wrist_left", "wrist_right"):
            self.assertEqual(C.recorded_frame("ds.demo_train.0", d["n_frames"] - 1, cam).shape[2], 3)
            with self.assertRaises(S.ArtifactUnavailable): C.recorded_frame("ds.demo_train.0", d["n_frames"], cam)
    def test_api_in_bundle_mode(self):
        h = serve(); self.addCleanup(h.shutdown)
        c, hl = get(h, "/api/health"); self.assertEqual(hl["data"]["n_episodes"], 3); self.assertEqual(hl["data"]["result_files_present"], 4)
        for e in get(h, "/api/experiments")[1]:
            c, x = get(h, "/api/experiments/" + e["id"]); self.assertEqual(c, 200)
            if e["id"] != "g2b": self.assertTrue(all(m["status"] == "READ_FROM_ARTIFACT" for m in x["metrics"]), (e["id"], [m["id"] for m in x["metrics"] if m["status"] != "READ_FROM_ARTIFACT"]))
        c, g = get(h, "/api/experiments/g2b"); self.assertIn("NOT EVALUATED", g["labels"]); self.assertTrue(all(x["result"] == "UNAVAILABLE" for x in g["integration_gates"]), "pending G2b reports are not in the bundle")
        c, r = get(h, "/api/episodes/pilot.A0_p00_r0/frame/3?view=mujoco:overview"); self.assertEqual((c, r["error"]), (503, "render_unavailable"))
        c, t = get(h, "/api/episodes/ds.demo_train.0/telemetry"); self.assertEqual(c, 200)
        r = urllib.request.urlopen(f"http://127.0.0.1:{h.server_address[1]}/api/episodes/ds.demo_train.0/frame/10?view=recorded:front"); self.assertEqual(r.headers["Content-Type"], "image/jpeg")
if __name__ == "__main__": unittest.main()
