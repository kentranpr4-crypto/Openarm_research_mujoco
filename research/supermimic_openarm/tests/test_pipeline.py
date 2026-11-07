"""Pipeline tests (CPU, tiny, seconds-minutes): mini-training, checkpoints/resume/corruption, run-dir safety, evaluation, benchmark, baselines, vector envs, Studio adapter, isolation and data-safety scans."""
import io, json, os, re, subprocess, sys, tempfile, unittest, contextlib, shutil
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT)); PKG = ROOT / "research" / "supermimic_openarm"
from research.supermimic_openarm import config as C, utils, runs, studio_adapter as SA
from research.supermimic_openarm.checkpoint import load_checkpoint, verify_checkpoint, CheckpointError
try:
    from research.supermimic_openarm.envs.openarm_env import _load_sim; _load_sim(); SIM_OK, WHY = True, ""
except Exception as e: SIM_OK, WHY = False, str(e)
needs_sim = unittest.skipUnless(SIM_OK, f"simulation dependencies / upstream model missing: {WHY}")
def cfg_for(tmp, **over):
    c = C.load_config(str(PKG / "configs/debug_cpu.yaml"), [f"experiment.output_dir={tmp}"] + [f"{k}={v}" for k, v in over.items()]); return c
@needs_sim
class TestMiniTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from research.supermimic_openarm.trainer import Trainer
        cls.tmp = tempfile.mkdtemp(); cls.cfg = cfg_for(cls.tmp, **{"train.iterations": 4, "train.checkpoint_interval": 2, "train.eval_interval": 2, "train.save_replay": "true"}); cls.rd = runs.run_dir_for(cls.cfg); runs.prepare_run_dir(cls.cfg, cls.rd)
        cls.tr = Trainer(cls.cfg, cls.rd, log=lambda *a: None); runs.write_run_files(cls.cfg, cls.rd, runs.build_env_manifest(cls.cfg, cls.tr.model_info)); cls.out = cls.tr.train(max_iterations=2)
    @classmethod
    def tearDownClass(cls): shutil.rmtree(cls.tmp, ignore_errors=True)
    def test_01_losses_and_gradients_finite(self):
        rows = [json.loads(l) for l in (self.rd / "metrics.jsonl").read_text().splitlines()]; self.assertEqual(len(rows), 2)
        for r in rows:
            for k in ("wm_loss", "critic_loss", "imag_objective", "actor_grad_norm"): self.assertTrue(np.isfinite(r[k]), k)
            self.assertEqual(r["bad_grads"], 0); self.assertGreater(r["actor_grad_norm"], 0.0, "actor receives no gradient through the world model")
    def test_02_files_and_integrity(self):
        for f in ("config.resolved.json", "metadata.json", "env_manifest.json", "metrics.jsonl", "checkpoints/latest.pt", "checkpoints/manifest.json", "checkpoints/best.pt"): self.assertTrue((self.rd / f).is_file(), f)
        md = json.loads((self.rd / "metadata.json").read_text()); self.assertIn("commit", md["git"]); self.assertIn("torch", md); self.assertTrue((self.rd / "eval").is_dir() and list((self.rd / "trajectories").glob("*.npz")))
        verify_checkpoint(self.rd / "checkpoints/latest.pt"); em = json.loads((self.rd / "env_manifest.json").read_text()); self.assertTrue(em["privileged_state"]); self.assertFalse(em["protected_seeds_used"])
    def test_03_corruption_is_detected(self):
        d = Path(tempfile.mkdtemp()); shutil.copytree(self.rd / "checkpoints", d / "c"); p = d / "c" / "latest.pt"; b = bytearray(p.read_bytes()); b[len(b) // 2] ^= 0xFF; p.write_bytes(bytes(b))
        with self.assertRaises(CheckpointError): load_checkpoint(p); 
        with self.assertRaises(CheckpointError): load_checkpoint(d / "c" / "missing.pt")
        shutil.rmtree(d)
    def test_04_resume_continues_and_restores_state(self):
        from research.supermimic_openarm.trainer import Trainer
        ck = load_checkpoint(self.rd / "checkpoints/latest.pt"); self.assertEqual(ck["iteration"], 2)
        tr2 = Trainer(self.cfg, self.rd, resume=True, log=lambda *a: None); self.assertEqual(tr2.iteration, 2)
        for a, b in zip(self.tr.actor.parameters(), tr2.actor.parameters()): torch.testing.assert_close(a, b)
        self.assertGreater(tr2.replay.size, 0, "replay buffer restored (save_replay: true)"); out = tr2.train(); self.assertEqual(out["iteration"], 4); self.assertTrue((self.rd / "COMPLETED").is_file())
        rows = [json.loads(l) for l in (self.rd / "metrics.jsonl").read_text().splitlines()]; self.assertEqual([r["iteration"] for r in rows], [1, 2, 3, 4])
    def test_05_completed_run_is_never_overwritten(self):
        if not (self.rd / "COMPLETED").is_file(): self.skipTest("depends on test_04")
        with self.assertRaises(FileExistsError): runs.prepare_run_dir(self.cfg, self.rd); 
        with self.assertRaises(FileExistsError): runs.prepare_run_dir(self.cfg, self.rd, resume=True)
    def test_06_nonempty_dir_requires_resume(self):
        d = Path(tempfile.mkdtemp()) / "x"; d.mkdir(); (d / "f").write_text("1")
        with self.assertRaises(FileExistsError): runs.prepare_run_dir(self.cfg, d)
    def test_07_evaluate_cli_and_correctness(self):
        if not (self.rd / "checkpoints/latest.pt").is_file(): self.skipTest("no checkpoint")
        from research.supermimic_openarm import evaluate as EV
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): rc = EV.main(["--checkpoint", str(self.rd / "checkpoints/latest.pt"), "--task", "reaching", "--episodes", "2"])
        self.assertEqual(rc, 0); j = json.loads(buf.getvalue().split("\nwritten:")[0]); self.assertEqual(j["n_episodes"], 2); self.assertTrue(0.0 <= j["success_rate"] <= 1.0); self.assertTrue(j["privileged_state"]); self.assertGreater(j["actor_inference_latency_ms"], 0)
        self.assertTrue(all(s >= 9_000_000 for s in j["seeds"]) and not set(j["seeds"]) & utils.PROTECTED_SEEDS)
        with contextlib.redirect_stderr(io.StringIO()): self.assertEqual(EV.main(["--checkpoint", str(self.rd / "checkpoints/latest.pt"), "--task", "pushing"]), 2)
        with contextlib.redirect_stderr(io.StringIO()): self.assertEqual(EV.main(["--checkpoint", str(self.rd / "checkpoints/nope.pt"), "--task", "reaching"]), 2)
    def test_08_studio_adapter_reads_the_run(self):
        root = self.cfg.experiment.output_dir; rid = f"{self.cfg.experiment.name}.seed{self.cfg.experiment.seed}"
        code, runs_ = SA.handle("runs", root=root); self.assertEqual(code, 200); self.assertEqual(runs_[0]["id"], rid); code, c = SA.handle(f"runs/{rid}/curves", root=root); self.assertIn("wm_loss", c["series"]); code, ev = SA.handle(f"runs/{rid}/evaluations", root=root); self.assertTrue(ev["evaluations"])
        code, tl = SA.handle(f"runs/{rid}/trajectories", root=root); self.assertTrue(tl["files"]); code, t = SA.handle(f"runs/{rid}/trajectory/{tl['files'][0]}", root=root); self.assertEqual(len(t["wm_one_step"]), t["n_steps"]); self.assertEqual(len(t["one_step_error"]), t["n_steps"]); self.assertEqual(len(t["states"]), t["n_steps"] + 1)
        self.assertEqual(SA.handle("runs/..%2Fetc/curves", root=root)[0], 404); self.assertEqual(SA.handle("nope", root=root)[0], 404); code, ts = SA.handle("tasks", root=root); self.assertEqual({x["id"] for x in ts}, {"reaching", "pushing", "pick_place"})
class TestAdapterIndependence(unittest.TestCase):
    def test_studio_adapter_needs_no_torch_or_mujoco(self):
        code = "import sys; sys.modules['torch']=None; sys.modules['mujoco']=None; sys.path.insert(0, %r); from research.supermimic_openarm import studio_adapter as S; print(S.handle('status', root='/nonexistent')[0])" % str(ROOT)
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True); self.assertEqual(r.stdout.strip(), "200", r.stderr)
    def test_missing_root_is_empty_not_error(self): self.assertEqual(SA.list_runs("/definitely/not/here"), [])
class TestCLI(unittest.TestCase):
    def run_cli(self, *a):
        return subprocess.run([sys.executable, "-m", "research.supermimic_openarm.train", *a], capture_output=True, text=True, cwd=ROOT)
    def test_dry_run_prints_resolved_config(self):
        r = self.run_cli("--config", str(PKG / "configs/reaching_4090.yaml"), "--dry-run"); self.assertEqual(r.returncode, 0, r.stderr); self.assertIn("resolved configuration", r.stdout); self.assertIn('"horizon": 16', r.stdout)
    def test_missing_config_and_bad_override_fail_clearly(self):
        r = self.run_cli("--config", "nope.yaml", "--dry-run"); self.assertEqual(r.returncode, 2); self.assertIn("not found", r.stderr)
        r = self.run_cli("--config", str(PKG / "configs/debug_cpu.yaml"), "--override", "imagination.horizon=5", "--dry-run"); self.assertEqual(r.returncode, 2); self.assertIn("horizon", r.stderr)
    def test_cuda_request_without_cuda_fails_clearly(self):
        if torch.cuda.is_available(): self.skipTest("CUDA is available here")
        r = self.run_cli("--config", str(PKG / "configs/reaching_4090.yaml"), "--run-name", "gpu_absent_check_" + str(os.getpid())); self.assertEqual(r.returncode, 2); self.assertIn("CUDA", r.stderr)
@needs_sim
class TestBenchmarkBaselinesVector(unittest.TestCase):
    def test_benchmark_cli_cpu(self):
        from research.supermimic_openarm import benchmark as BM; tmp = tempfile.mkdtemp(); buf = io.StringIO()
        with contextlib.redirect_stdout(buf): rc = BM.main(["--config", str(PKG / "configs/debug_cpu.yaml"), "--steps", "32", "--updates", "1", "--override", f"experiment.output_dir={tmp}"])
        self.assertEqual(rc, 0); self.assertIn("NOT_TESTED_NO_GPU", buf.getvalue()); shutil.rmtree(tmp, ignore_errors=True)
    def test_bc_and_ppo_mini_runs(self):
        from research.supermimic_openarm.baselines.bc import train_bc; from research.supermimic_openarm.baselines.ppo import train_ppo; tmp = tempfile.mkdtemp()
        c = cfg_for(tmp, **{"bc.demos": 2, "bc.epochs": 2, "train.eval_episodes": 1, "env.max_steps": 20}); d = Path(tmp) / "bc"; d.mkdir(); r = train_bc(c, d); self.assertTrue(np.isfinite(r["mean_return"])); self.assertGreater(r["demo_env_transitions"], 20)
        c2 = cfg_for(tmp, **{"ppo.rollout_steps": 8, "ppo.epochs": 1, "ppo.minibatches": 1, "train.iterations": 1, "train.eval_episodes": 1, "env.max_steps": 20}); d2 = Path(tmp) / "ppo"; d2.mkdir(); r2 = train_ppo(c2, d2); self.assertEqual(r2["env_transitions"], 8 * c2.env.num_envs); self.assertTrue(np.isfinite(r2["mean_return"]))
        c3 = cfg_for(tmp, **{"env.task": "pushing"})
        with self.assertRaises(ValueError): from research.supermimic_openarm.baselines.bc import collect_demos; collect_demos(c3, __import__("research.supermimic_openarm.envs.tasks", fromlist=["x"]).make_task("pushing"), 1, 1_000_000)
        shutil.rmtree(tmp, ignore_errors=True)
    def test_subproc_vector_env_matches_sync(self):
        from research.supermimic_openarm.envs.vector import make_vec_env
        a = make_vec_env("sync", 2, "reaching", {}, 10, 20, 123); b = make_vec_env("subproc", 2, "reaching", {}, 10, 20, 123)
        try:
            sa, ga, _ = a.reset(); sb, gb, _ = b.reset(); np.testing.assert_array_equal(sa, sb); np.testing.assert_array_equal(ga, gb); tg = np.tile(sa[:, 16:24].astype(np.float64), 1)
            for _ in range(3): oa = a.step(tg); ob = b.step(tg); np.testing.assert_allclose(oa["next_state"], ob["next_state"], atol=1e-6)
        finally: b.close()
class TestSafetyAndIsolation(unittest.TestCase):
    def sources(self): return [p for p in PKG.rglob("*") if p.suffix in (".py", ".sh", ".yaml", ".toml", ".md", ".txt") and "__pycache__" not in p.parts and p.name != "test_pipeline.py"]
    def test_no_protected_data_or_act_access(self):
        bad = re.compile(r"outputs/datasets|episodes\.jsonl|act_full|closed_loop_pilot|g2b_|g2a_act|policies\.act|from policies|import policies|\beval\.g2b|studio_demo_bundle|\bsealed\b|\breserve\b")
        for p in self.sources():
            if p.name in ("README.md", "reproduction_audit.md", "test_core.py", "validation_report.md") or p.suffix == ".md": continue
            self.assertIsNone(bad.search(p.read_text()), f"forbidden reference in {p}")
    def test_no_hardcoded_absolute_paths(self):
        bad = re.compile(r"/home/user|/tmp/claude|\.venvs/|/root/")
        for p in self.sources(): self.assertIsNone(bad.search(p.read_text()), f"absolute cloud path in {p}")
    def test_existing_acts_g2b_and_upstream_files_untouched(self):
        r = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "policies", "control", "sim", "app"], capture_output=True, text=True).stdout.strip()
        self.assertEqual(r, "", f"existing source directories were modified:\n{r}")
        up = ROOT / "sim" / "openarm_mujoco"
        if (up / ".git").exists(): self.assertEqual(subprocess.run(["git", "-C", str(up), "status", "--porcelain"], capture_output=True, text=True).stdout.strip(), "", "upstream OpenArm model modified")
    def test_package_writes_only_to_gitignored_outputs(self):
        self.assertIn("outputs/", (ROOT / ".gitignore").read_text()); self.assertTrue(C.Config().experiment.output_dir.startswith("outputs/"))
    def test_no_torch_import_by_package_init_or_adapter(self):
        r = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(ROOT)!r}); import research.supermimic_openarm; print('torch' in sys.modules)"], capture_output=True, text=True); self.assertEqual(r.stdout.strip(), "False", r.stderr)
if __name__ == "__main__": unittest.main()
