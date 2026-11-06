"""Backend tests for OpenArm Research Studio (standard library unittest; no simulator rollout, no model, no GPU).
Run from the repository root:  python -m unittest discover -s app/tests -v"""
import os, re, sys, json, threading, tempfile, unittest, urllib.request, urllib.error
from pathlib import Path
from unittest import mock
os.environ["OPENARM_STUDIO_NO_RENDER"] = "1"                      # the MuJoCo scene is never created in these tests
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
from app.backend import security as S, catalog as C, experiments as X, server as SV
from app.backend.render import RENDERER
FULL = S.exists("outputs/audit_v1/audit_results.json") and S.exists("outputs/datasets/demo_train/episodes.jsonl") and len(C.list_episodes()) > 100            # the complete research outputs/ (not in git)
needs_full = unittest.skipUnless(FULL, "needs the complete research outputs/ directory (not stored in git); portability tests run in test_portable.py")
class Srv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = SV.make_server("127.0.0.1", 0); cls.port = cls.httpd.server_address[1]; cls.t = threading.Thread(target=cls.httpd.serve_forever, daemon=True); cls.t.start()
    @classmethod
    def tearDownClass(cls): cls.httpd.shutdown(); cls.httpd.server_close()
    def get(self, path, raw=False):
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=60); b = r.read(); return r.status, (b if raw else json.loads(b)), r.headers
        except urllib.error.HTTPError as e:
            b = e.read(); return e.code, (b if raw else json.loads(b)), e.headers
    def req(self, method, path):
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", method=method, data=b"{}" if method != "DELETE" else None)
        try: return urllib.request.urlopen(r, timeout=30).status
        except urllib.error.HTTPError as e: return e.code
# ------------------------------------------------------------------------------------------------------------------------ 1. API schema
class TestApi(Srv):
    def test_health(self):
        c, d, _ = self.get("/api/health"); self.assertEqual(c, 200); self.assertTrue(d["read_only"]); self.assertEqual(d["status"], "ok"); self.assertIn("render", d)
        self.assertFalse(d["render"]["initialised"], "the MuJoCo scene must not be created at start-up or by a health check")
        self.assertEqual([m["id"] for m in d["modes"]], ["A", "B", "C"]); self.assertFalse([m for m in d["modes"] if m["id"] == "C"][0]["enabled"])
    def test_experiments_schema(self):
        c, d, _ = self.get("/api/experiments"); self.assertEqual(c, 200); self.assertEqual([e["id"] for e in d], ["act_v1", "wm3b", "stage1", "branching", "g2a", "g2b"])
        for e in d: self.assertTrue({"id", "title", "labels", "validity", "purpose"} <= set(e)); self.assertTrue(set(e["labels"]) <= set(X.LABELS))
        for e in d:
            c, x, _ = self.get(f"/api/experiments/{e['id']}"); self.assertEqual(c, 200)
            for k in ("purpose", "dataset", "candidates", "method", "metrics", "failure_modes", "outcome_validity", "artifacts", "labels", "caveats"): self.assertIn(k, x, f"{e['id']} lacks {k}")
            for m in x["metrics"]: self.assertIn(m["status"], ("READ_FROM_ARTIFACT", "UNAVAILABLE", "NOT_EVALUATED"))
    @needs_full
    def test_episodes_and_metadata(self):
        c, d, _ = self.get("/api/episodes"); self.assertEqual(c, 200); self.assertGreater(len(d), 100)
        for e in d[:3] + d[-3:]: self.assertTrue({"id", "source", "title", "has_images", "has_contacts"} <= set(e))
        src = {e["source"] for e in d}; self.assertTrue({"dataset", "pilot", "wm1"} <= src)
        c, m, _ = self.get("/api/episodes/pilot.A0_p00_r0/metadata"); self.assertEqual(c, 200); self.assertEqual(m["chunk_size"], 20); self.assertIn("alignment", m)
    def test_unknown_endpoint_and_ids(self):
        self.assertEqual(self.get("/api/nope")[0], 404); self.assertEqual(self.get("/api/experiments/not_real")[0], 404)
        for bad in ("..%2f..%2fetc", "ds.demo_train.9999", "pilot.A0_p99_r9", "zz.1", "wm1.%00"): self.assertEqual(self.get(f"/api/episodes/{bad}/telemetry")[0], 404, bad)
    def test_static_files_only_from_frontend(self):
        self.assertEqual(self.get("/", raw=True)[0], 200); self.assertEqual(self.get("/static/js/main.js", raw=True)[0], 200)
        for p in ("/static/../backend/server.py", "/static/%2e%2e/backend/server.py", "/../app/backend/security.py"): self.assertEqual(self.get(p, raw=True)[0], 404, p)
# ------------------------------------------------------------------------------------------------------------------------ 2. artifacts + provenance
class TestProvenance(unittest.TestCase):
    @needs_full
    def test_numbers_are_read_from_artifacts(self):
        a = {m["id"]: m for m in X.detail("act_v1")["metrics"]}; self.assertEqual((a["nominal"]["value"]["k"], a["nominal"]["value"]["n"]), (30, 30)); self.assertEqual((a["unseen"]["value"]["k"], a["unseen"]["value"]["n"]), (18, 30))
        s = {m["id"]: m for m in X.detail("stage1")["metrics"]}; self.assertEqual((s["a0"]["value"]["k"], s["a0"]["value"]["n"]), (18, 24)); self.assertEqual((s["a2"]["value"]["k"], s["a2"]["value"]["n"]), (17, 24)); self.assertEqual((s["a1"]["value"]["k"], s["a1"]["value"]["n"]), (49, 72)); self.assertEqual(s["a2_a1"]["value"]["verdict"], "INCONCLUSIVE")
        b = {m["id"]: m for m in X.detail("branching")["metrics"]}; self.assertEqual((b["states"]["value"], b["branches"]["value"]), (216, 648)); self.assertLess(b["j20"]["value"]["lo"], b["j20"]["value"]["value"]); self.assertLess(b["j20"]["value"]["value"], b["j20"]["value"]["hi"])
        w = {m["id"]: m for m in X.detail("wm3b")["metrics"]}; self.assertAlmostEqual(w["wm3b_box_k20"]["value"], 3.20, places=2); self.assertAlmostEqual(w["wm2c0_box_k20"]["value"], 3.83, places=2)
        g = {m["id"]: m for m in X.detail("g2a")["metrics"]}; self.assertEqual((g["poses"]["value"], g["cands"]["value"], g["branches"]["value"]), (48, 9, 432))
    def test_historical_numbers_confirmed_by_reports(self):
        for e in ("act_v1", "wm3b"):
            for m in X.detail(e)["metrics"]:
                if "doc_check" in m: self.assertTrue(m["doc_check"]["found"], f"{e}/{m['id']}: the report text does not contain {m['doc_check']['contains']!r}")
    def test_every_metric_has_provenance(self):
        for e in X.ORDER:
            d = X.detail(e); self.assertTrue(d["artifacts"]); 
            for m in d["metrics"]:
                if m["status"] == "READ_FROM_ARTIFACT" and m["id"] != "integration": self.assertTrue(m["source"] and m["path"] != "" or m["source"], f"{e}/{m['id']} has no source")
            for a in d["artifacts"]: self.assertIn("exists", a)
    def test_unavailable_artifact_is_explicit_not_fabricated(self):
        m = X.M("x", "missing", "outputs/does_not_exist.json", ("a",)); self.assertEqual(m["status"], "UNAVAILABLE"); self.assertIsNone(m["value"]); self.assertIn("not found", m["reason"])
        m = X.M("x", "bad path", "outputs/audit_v1/audit_results.json", ("no_such_key",)); self.assertEqual(m["status"], "UNAVAILABLE"); self.assertIsNone(m["value"])
    def test_wilson_interval(self):
        lo, hi = X.wilson(30, 30); self.assertAlmostEqual(hi, 1.0, places=6); self.assertAlmostEqual(lo, 0.886, places=2); lo, hi = X.wilson(18, 30); self.assertAlmostEqual(lo, 0.423, places=2); self.assertAlmostEqual(hi, 0.754, places=2)
# ------------------------------------------------------------------------------------------------------------------------ 3. missing / corrupted trajectories + 4/5 indexing and alignment
@needs_full
class TestTelemetry(unittest.TestCase):
    def _patched(self, arrays=None, raw_bytes=None, kind="pilot"):
        d = tempfile.TemporaryDirectory(); p = Path(d.name) / "x.npz"
        if raw_bytes is not None: p.write_bytes(raw_bytes)
        else: np.savez(p, **arrays)
        C._load.cache_clear(); self.addCleanup(d.cleanup); self.addCleanup(C._load.cache_clear)
        return mock.patch.object(C, "_file_for", return_value=(kind, p, ("A0", 0, 0)))
    def test_corrupted_file(self):
        with self._patched(raw_bytes=b"not an npz at all"):
            with self.assertRaises(S.ArtifactUnavailable) as e: C.telemetry("pilot.A0_p00_r0")
            self.assertIn("corrupted", str(e.exception))
    def test_missing_state(self):
        with self._patched(dict(box_pos=np.zeros((5, 3)))):
            with self.assertRaises(S.ArtifactUnavailable) as e: C.telemetry("pilot.A0_p00_r0")
            self.assertIn("16-D joint state", str(e.exception))
    def test_wrong_state_shape(self):
        with self._patched(dict(state=np.zeros((5, 7)), box_pos=np.zeros((5, 3)))):
            with self.assertRaises(S.ArtifactUnavailable): C.telemetry("pilot.A0_p00_r0")
    def test_box_shorter_than_state(self):
        with self._patched(dict(state=np.zeros((10, 16)), box_pos=np.zeros((4, 3)))):
            with self.assertRaises(S.ArtifactUnavailable) as e: C.telemetry("pilot.A0_p00_r0")
            self.assertIn("inconsistent", str(e.exception))
    def test_missing_optional_data_is_reported_not_invented(self):
        with self._patched(dict(state=np.zeros((10, 16), np.float32), box_pos=np.zeros((10, 3)), tcp_pos=np.zeros((10, 3)), action=np.zeros((10, 16)))):
            t = C.telemetry("pilot.A0_p00_r0"); self.assertIsNone(t["contacts"]); self.assertIsNone(t["box_quat"]); self.assertIsNone(t["model_predictions"])
            self.assertTrue(any("contact forces not recorded" in x for x in t["limitations"])); self.assertTrue(any("box orientation not recorded" in x for x in t["limitations"]))
    def test_no_box_means_no_reconstruction(self):
        with self._patched(dict(state=np.zeros((10, 16), np.float32))):
            with self.assertRaises(S.ArtifactUnavailable): C.render_inputs("pilot.A0_p00_r0", 3)
    def test_frame_index_range(self):
        for k in (-1, 200, 10 ** 6):
            with self.assertRaises(S.ArtifactUnavailable): C.render_inputs("pilot.A0_p00_r0", k)
        st, box, q = C.render_inputs("pilot.A0_p00_r0", 199); self.assertEqual(st.shape, (16,)); self.assertEqual(box.shape, (3,)); self.assertEqual(q.tolist(), [1.0, 0.0, 0.0, 0.0])
        with self.assertRaises(S.ArtifactUnavailable): C.recorded_frame("ds.demo_train.0", 138, "front")
        self.assertEqual(C.recorded_frame("ds.demo_train.0", 137, "front").shape, (120, 160, 3))
        with self.assertRaises(S.ArtifactUnavailable): C.recorded_frame("pilot.A0_p00_r0", 0, "front")
    def test_observation_action_alignment_conventions(self):
        for eid, n_act_minus in (("pilot.A0_p00_r0", 0), ("ds.demo_train.0", 0)):
            t = C.telemetry(eid); self.assertEqual(len(t["state"]), t["n_frames"]); self.assertEqual(t["n_actions"], t["n_frames"] + n_act_minus)
        w = C.telemetry([e["id"] for e in C.list_episodes() if e["source"] == "wm1"][0]); self.assertEqual(w["n_actions"], w["n_frames"] - 1, "WM-1 logs T+1 observations for T actions")
    def test_action_precedes_its_physical_effect_by_one_row(self):
        """a[t] is issued after observation t, so the measured gripper follows a[t] at row t+1 (lag-1 error much smaller than lag-0) while the gripper is closing"""
        t = C.telemetry("ds.demo_train.0"); g = np.array(t["gripper_left_m"]); a = np.array([x[7] for x in t["action"]]); n = len(a)
        idx = [i for i in range(1, n - 1) if abs(a[i] - a[i - 1]) > 1e-4 and 0.021 < g[i + 1]]
        self.assertGreater(len(idx), 3); lag1 = np.median([abs(g[i + 1] - a[i]) for i in idx]); lag0 = np.median([abs(g[i] - a[i]) for i in idx]); self.assertLess(lag1, 0.5 * lag0)
    def test_pilot_chunk_boundaries_and_events_label(self):
        t = C.telemetry("pilot.A0_p00_r0"); self.assertEqual(t["chunk_boundaries"], list(range(0, 200, 20))); self.assertIn("DERIVED", t["events_note"]); self.assertEqual(t["n_frames"], 200)
        t = C.telemetry("ds.demo_train.0"); self.assertEqual(t["chunk_boundaries"], []); self.assertEqual(t["phases"][0]["name"], "settle"); self.assertIn("RECORDED", t["phases_note"])
    def test_derive_events_unit(self):
        g = np.r_[np.full(10, 0.044), np.full(20, 0.01), np.linspace(0.01, 0.044, 10)]; z = np.r_[np.zeros(15), np.linspace(0, 0.05, 10), np.full(15, 0.05)]; ev = {e["kind"]: e["t"] for e in C.derive_events(g, z)}
        self.assertEqual(ev["close_command"], 10); self.assertTrue(15 <= ev["lift"] <= 25); self.assertGreater(ev["release"], ev["lift"])
# ------------------------------------------------------------------------------------------------------------------------ 6/7 labels, history vs future
class TestLabels(unittest.TestCase):
    def test_derive_labels_rules(self):
        self.assertEqual(X.derive_labels(impl_files=["eval/g2b_labeller.py"], artifact_label=None, design_only=True, results_exist=False), ["IMPLEMENTED", "DESIGN ONLY", "NOT EVALUATED"])
        self.assertEqual(X.derive_labels([], None, design_only=True, results_exist=False, trained=False), ["DESIGN ONLY", "NOT TRAINED", "NOT EVALUATED"])
        self.assertEqual(X.derive_labels(["eval/g2b_labeller.py"], "DEVELOPMENT run"), ["IMPLEMENTED", "EXPERIMENTAL"])
        self.assertEqual(X.derive_labels(["eval/g2b_labeller.py"], "INDEPENDENT", independent=True, all_gates_pass=True), ["IMPLEMENTED", "VALIDATED"])
        self.assertEqual(X.derive_labels(["eval/g2b_labeller.py"], "INDEPENDENT", independent=True, all_gates_pass=False), ["IMPLEMENTED", "EXPERIMENTAL"], "a failed gate must not be shown as VALIDATED")
        self.assertEqual(X.derive_labels(["eval/does_not_exist.py"], "DEVELOPMENT"), ["EXPERIMENTAL"])
    @needs_full
    def test_experiment_labels_from_artifacts(self):
        L = {e["id"]: e for e in X.summary()}; self.assertNotIn("VALIDATED", L["wm3b"]["labels"], "WM3-B has a failed independent gate (G3) in the saved evaluation"); self.assertIn("EXPERIMENTAL", L["wm3b"]["labels"])
        self.assertEqual(L["g2b"]["labels"], ["IMPLEMENTED", "DESIGN ONLY", "NOT EVALUATED"]); self.assertEqual(L["g2a"]["validity"], "INVALID_FOR_GRASP_QUALITY")
    @needs_full
    def test_g2b_has_no_results_and_no_success_rates(self):
        g = X.detail("g2b"); eff = [m for m in g["metrics"] if m["id"] == "effects"][0]; self.assertEqual(eff["status"], "NOT_EVALUATED"); self.assertIsNone(eff["value"])
        txt = json.dumps(g); self.assertNotRegex(txt, r"success_rate|\"rate\""); self.assertEqual([c["id"] for c in g["candidate_defs"]], ["c0", "O1", "O3", "D1", "D2", "H1", "S0"])
        self.assertEqual({x["gate"]: x["result"] for x in g["integration_gates"]}, {"I1": "PASS", "I2": "PASS", "I3": "PASS", "I4": "PASS"})
    def test_g2a_hides_oracle_success(self):
        g = X.detail("g2a"); ids = {m["id"] for m in g["metrics"]}; self.assertFalse({m for m in ids if "success" in m or "oracle" in m or "headroom" in m}); self.assertEqual(g["outcome_validity"]["status"], "INVALID_FOR_GRASP_QUALITY")
    def test_stage1_does_not_claim_wm_improvement(self):
        s = X.detail("stage1"); self.assertIn("INCONCLUSIVE", json.dumps(s)); self.assertIn("did NOT show", s["outcome_validity"]["text"])
    def test_history_and_future_are_separate(self):
        cur = SV.architecture("current"); fut = SV.architecture("future")
        for n in cur["nodes"]: self.assertNotIn("future", n["layers"]); self.assertTrue(set(n["layers"]) <= {"current", "stage1", "g2b"})
        self.assertTrue({l["id"] for l in cur["layers"]} == {"current", "stage1", "g2b"}); self.assertIn("NOT TRAINED", fut["banner"])
        gwm = [n for n in fut["nodes"] if n["id"] == "gwm"][0]; self.assertEqual(gwm["status"], "NOT TRAINED"); self.assertEqual(gwm["files"], [])
        self.assertIn("SYNTHETIC ILLUSTRATION", fut["timeline_illustration"]["label"]); self.assertNotRegex(json.dumps(fut), r"probab[a-z]*\"?\s*:\s*[0-9]|success_rate")
        self.assertTrue(all(f["exists"] for n in cur["nodes"] for f in n["files_status"] if not f["path"].startswith("sim/openarm_mujoco/")), "every implementation file named by the architecture must exist (the pinned upstream model is optional)")
    def test_research_status_blockers_verified(self):
        st = SV.research_status(); self.assertTrue(all(b["verified_in_doc"] or b["doc_missing"] for b in st["blockers"])); self.assertEqual([m["enabled"] for m in st["modes"]], [True, True, False])
        self.assertEqual([e for e in st["experiments"] if e["id"] == "wam"][0]["labels"], ["DESIGN ONLY", "NOT TRAINED", "NOT EVALUATED"])
# ------------------------------------------------------------------------------------------------------------------------ 8/9 no rollout, read-only
class TestSafety(Srv):
    def test_no_simulator_rollout_or_policy_in_backend_source(self):
        bad = re.compile(r"mj_step|\.step\(|load_chunk_policies|import torch|from torch|ACTPolicy|subprocess|os\.system|\beval\(|\bexec\(|__import__|\bpickle\b|allow_pickle=True|LabEnv|make_env")
        for f in (ROOT / "app/backend").glob("*.py"):
            import ast; tree = ast.parse(f.read_text())                                                         # comments and docstrings are not code: inspect the parsed code only
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) and isinstance(node.body[0].value.value, str): node.body = node.body[1:] or [ast.Pass()]
            code = ast.unparse(tree); self.assertIsNone(bad.search(code), f"{f.name} contains a forbidden call: {bad.search(code) and bad.search(code).group(0)}")
        self.assertIn("mj_forward", (ROOT / "app/backend/render.py").read_text())
    def test_no_file_writes_in_backend(self):
        for f in (ROOT / "app/backend").glob("*.py"):
            code = f.read_text(); self.assertNotRegex(code, r"open\([^)]*['\"][wax]b?['\"]"); self.assertNotIn("write_text", code); self.assertNotIn("write_bytes", code); self.assertNotIn("os.remove", code); self.assertNotIn("shutil", code)
    def test_modules_do_not_import_torch(self): self.assertNotIn("torch", sys.modules)
    def test_mutating_methods_rejected(self):
        for m in ("POST", "PUT", "DELETE", "PATCH"): self.assertEqual(self.req(m, "/api/health"), 405, m)
    @needs_full
    def test_security_resolve(self):
        for bad in ("/etc/passwd", "../etc/passwd", "outputs/../../etc/passwd", "secret/x", ".git/config", "outputs\x00x", "", None): 
            with self.assertRaises(S.ArtifactUnavailable): S.resolve(bad)
        self.assertTrue(S.resolve("outputs/e6b/stage1_results.json").is_file())
        with tempfile.TemporaryDirectory() as td:
            link = ROOT / "outputs" / ".studio_test_link"
            try: os.symlink(td, link)
            except OSError: self.skipTest("symlinks unavailable"); 
            try:
                with self.assertRaises(S.ArtifactUnavailable): S.resolve("outputs/.studio_test_link")
            finally:
                if link.is_symlink(): link.unlink()
    @needs_full
    def test_artifacts_unchanged_by_requests(self):
        f = ROOT / "outputs/e6b/stage1_results.json"; before = (f.stat().st_mtime_ns, f.stat().st_size)
        for p in ("/api/experiments/act_v1", "/api/research/status", "/api/episodes/pilot.A0_p00_r0/telemetry"): self.get(p)
        self.assertEqual(before, (f.stat().st_mtime_ns, f.stat().st_size))
    @needs_full
    def test_render_unavailable_is_explicit(self):
        c, d, _ = self.get("/api/episodes/pilot.A0_p00_r0/frame/5?view=mujoco:overview"); self.assertEqual(c, 503); self.assertEqual(d["error"], "render_unavailable"); self.assertIn("disabled", d["reason"])
        c, d, _ = self.get("/api/episodes/ds.demo_train.0/frame/5?view=recorded:front", raw=True); self.assertEqual(c, 200)
        self.assertEqual(self.get("/api/episodes/ds.demo_train.0/frame/abc?view=recorded:front")[0], 404); self.assertEqual(self.get("/api/episodes/ds.demo_train.0/frame/500?view=recorded:front")[0], 404)
        self.assertEqual(self.get("/api/episodes/pilot.A0_p00_r0/frame/5?view=recorded:front")[0], 404)
if __name__ == "__main__": unittest.main()
