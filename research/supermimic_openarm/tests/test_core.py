"""Unit tests: config, differentiable FK, adapter parity, environment/task contracts, world model, imagination gradients, replay.  CPU only."""
import copy, sys, unittest
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from research.supermimic_openarm import config as C, utils
from research.supermimic_openarm.envs import layout as L
from research.supermimic_openarm.envs.tasks import make_task, TASK_CLASSES
from research.supermimic_openarm.models import WorldModel, Actor, Critic, critic_features
from research.supermimic_openarm.imagination import imagine, survival_weighted_return, lambda_returns, critic_loss_on, set_requires_grad
from research.supermimic_openarm.replay import Replay
CFG = C.load_config(str(ROOT / "research/supermimic_openarm/configs/debug_cpu.yaml"))
try:
    from research.supermimic_openarm.envs.openarm_env import OpenArmTaskEnv, _load_sim; _load_sim(); SIM_OK, SIM_WHY = True, ""
except Exception as e: SIM_OK, SIM_WHY = False, str(e)
needs_sim = unittest.skipUnless(SIM_OK, f"simulation dependencies / upstream model missing: {SIM_WHY}")
class TestConfig(unittest.TestCase):
    def test_all_shipped_configs_validate(self):
        for f in (ROOT / "research/supermimic_openarm/configs").glob("*.yaml"): C.load_config(str(f))
    def test_rejects_bad_values_with_reason(self):
        for ov, frag in (("imagination.horizon=7", "horizon"), ("env.task=walking", "task"), ("env.sim_backend=mjwarp", "mujoco_cpu"), ("system.precision=fp16", "precision"), ("experiment.seed=32", "reserved"), ("train.eval_seed_base=31", "reserved"), ("env.control_hz=50", "control_hz")):
            with self.assertRaises(ValueError, msg=ov) as cm: C.load_config(str(ROOT / "research/supermimic_openarm/configs/debug_cpu.yaml"), [ov])
            self.assertIn(frag, str(cm.exception))
        with self.assertRaises(ValueError): C.load_config(str(ROOT / "research/supermimic_openarm/configs/debug_cpu.yaml"), ["nonsense.key=1"])
        with self.assertRaises(FileNotFoundError): C.load_config("/no/such/config.yaml")
    def test_protected_seed_guard(self):
        with self.assertRaises(ValueError): utils.assert_seeds_allowed(31)
        utils.assert_seeds_allowed(1000, 9_000_000)
@needs_sim
class TestKinematicsAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from research.supermimic_openarm.trainer import build_fk_and_adapter
        cls.fk, cls.adapter, cls.info = build_fk_and_adapter(CFG, torch.device("cpu")); cls.M = _load_sim()
    def test_fk_matches_mujoco(self):
        mj, lab = self.M["mujoco"], self.M["lab_scene"]; m, d = lab.compile_lab(with_cameras=False); A = self.M["ActionAdapter"](m, 20); tcp = m.body("openarm_left_hand_tcp").id; rng = np.random.default_rng(0); worst = 0.0
        for _ in range(25):
            q = np.clip(rng.normal(0, 0.7, 8), A.limits.lo[:8] + 1e-3, A.limits.hi[:8] - 1e-3); d.qpos[A.qadr[:8]] = q; mj.mj_kinematics(m, d); ref = d.xpos[tcp].copy()
            out = self.fk(torch.tensor(q[None], dtype=torch.float64))[0].numpy(); worst = max(worst, float(np.abs(out - ref).max()))
        self.assertLess(worst, 1e-6, f"FK error {worst}")
    def test_fk_is_differentiable(self):
        q = torch.zeros(3, 8, requires_grad=True); self.fk(q).sum().backward(); self.assertTrue(torch.isfinite(q.grad).all()); self.assertGreater(q.grad[:, :7].abs().sum().item(), 0)
    def test_torch_adapter_parity_with_original_clip_slew(self):
        A = self.M["ActionAdapter"](self.M["lab_scene"].compile_lab(with_cameras=False)[0], 20); rng = np.random.default_rng(1); ta = self.adapter
        for _ in range(50):
            prev = rng.uniform(A.limits.lo, A.limits.hi); tgt = rng.uniform(A.limits.lo - 1, A.limits.hi + 1); ref = A.to_ctrl(tgt, prev)[0][:8]
            out = ta.execute(torch.tensor(tgt[:8], dtype=torch.float32)[None], torch.tensor(prev[:8], dtype=torch.float32)[None])[0].numpy(); np.testing.assert_allclose(out, ref, atol=1e-5)
@needs_sim
class TestEnvAndTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.envs = {t: OpenArmTaskEnv(t, max_steps=20) for t in TASK_CLASSES}
    def test_state_and_action_dimensions(self):
        for t, e in self.envs.items():
            s, g, tcp = e.reset(7); self.assertEqual(s.shape, (L.STATE_DIM,)); self.assertEqual(g.shape, (e.task.goal_dim,)); self.assertEqual(tcp.shape, (3,)); self.assertTrue(np.isfinite(s).all())
            o = e.step(s[L.PREV]); self.assertEqual(o["state"].shape, (L.STATE_DIM,)); self.assertEqual(o["executed"].shape, (L.ACT_DIM,))
            self.assertEqual(e.task.actor_obs(torch.as_tensor(s)[None], torch.as_tensor(g)[None], torch.as_tensor(tcp)[None]).shape[-1], e.task.actor_obs_dim)
            self.assertEqual(sum(x["dim"] for x in e.task.observation_spec()["terms"]), e.task.actor_obs_dim)
    def test_every_task_exposes_all_specs(self):
        for t in TASK_CLASSES:
            k = make_task(t)
            for spec in ("observation_spec", "action_spec", "reward_spec", "termination_spec", "success_spec", "randomization_spec"): self.assertTrue(isinstance(getattr(k, spec)(), dict) and getattr(k, spec)(), f"{t}.{spec}")
    def test_clip_and_slew_applied_to_executed_action(self):
        e = self.envs["reaching"]; s, g, _ = e.reset(3); prev = s[L.PREV].copy(); o = e.step(prev + 100.0); ex = o["executed"]
        self.assertTrue(np.all(ex <= e.adapter.limits.hi[:8] + 1e-6)); self.assertTrue(np.all(np.abs(ex - prev) <= e.adapter.limits.max_speed[:8] / 20 + 1e-6)); self.assertGreater(o["info"]["n_slewed"] + o["info"]["n_clipped"], 0)
        with self.assertRaises(ValueError): e.step(np.full(8, np.nan))
        with self.assertRaises(ValueError): e.step(np.zeros(7))
    def test_deterministic_reset(self):
        for t, e in self.envs.items():
            a = e.reset(11); b = e.reset(11); c = e.reset(12)
            for x, y in zip(a, b): np.testing.assert_array_equal(x, y)
            self.assertFalse(np.array_equal(a[1], c[1]) and np.array_equal(a[0], c[0]), t)
    def test_deterministic_rollout(self):
        e = self.envs["pushing"]; run = []
        for _ in range(2):
            s, g, _ = e.reset(21); rng = np.random.default_rng(0)
            for _ in range(8): o = e.step(o["executed"] + 0.05 * rng.standard_normal(8) if _ else s[L.PREV]) if True else None
            run.append(o["state"].copy())
        np.testing.assert_array_equal(run[0], run[1])
    def test_termination_and_truncation(self):
        e = self.envs["reaching"]; e.reset(2); o = None
        for i in range(25):
            o = e.step(e.prev_ctrl[:8]); 
            if o["truncated"]: break
        self.assertTrue(o["truncated"] and not o["terminated"] and e.t == 20)
    def test_reward_consistent_with_torch_function(self):
        for t, e in self.envs.items():
            s, g, tcp = e.reset(4); o = e.step(s[L.PREV]); r = e.task.reward_torch(torch.as_tensor(o["state"])[None], torch.as_tensor(o["executed"])[None], torch.as_tensor(o["goal"])[None], torch.as_tensor(o["tcp"])[None])[0].item(); self.assertAlmostEqual(r, o["reward"], places=5)
    def test_success_criteria_unit(self):
        R, P, K = make_task("reaching"), make_task("pushing"), make_task("pick_place"); g = np.array([0.4, 0.2, 0.5])
        self.assertTrue(R.success_now(dict(tcp=g + 0.01, goal=g, tcp_speed=0.01))); self.assertFalse(R.success_now(dict(tcp=g + 0.05, goal=g, tcp_speed=0.01))); self.assertFalse(R.success_now(dict(tcp=g + 0.01, goal=g, tcp_speed=0.5)))
        base = dict(box_pos=np.array([0.44, 0.30, 0.384]), goal=np.array([0.44, 0.30]), box_speed=0.0, box_max_rise=0.05, gripper_open_m=0.044, finger_contacts=0)
        self.assertTrue(K.success_now(base)); self.assertFalse(K.success_now({**base, "finger_contacts": 1}), "still held by a finger"); self.assertFalse(K.success_now({**base, "gripper_open_m": 0.01}), "gripper closed")
        self.assertFalse(K.success_now({**base, "box_max_rise": 0.0}), "never lifted: position alone must not count"); self.assertFalse(K.success_now({**base, "box_pos": np.array([0.44, 0.30, 0.45])}), "box in the air")
        self.assertFalse(K.success_now({**base, "box_speed": 0.1})); self.assertEqual(K.hold_steps, 10)
        pb = dict(box_pos=np.array([0.5, 0.1, 0.3805]), goal=np.array([0.5, 0.1]), box_speed=0.0, box_max_rise=0.0); self.assertTrue(P.success_now(pb)); self.assertFalse(P.success_now({**pb, "box_max_rise": 0.05}), "carried, not pushed")
    def test_deployment_observation_has_no_privileged_state(self):
        priv = {"box_pos", "box_quat", "box_vel"}
        for t, e in self.envs.items():
            s, g, tcp = e.reset(1); dep = e.deployment_observation(s, g, tcp); self.assertFalse(priv & set(dep), t)
            names = {x["name"] for x in e.task.observation_spec()["terms"] if x["privileged"]}; self.assertEqual(bool(names), e.task.uses_privileged_obs); self.assertEqual(not names, e.task.deployable_without_perception)
        r = make_task("reaching"); self.assertEqual(r.actor_obs_dim, sum(len(v) for v in self.envs["reaching"].deployment_observation(*self.envs["reaching"].reset(1)).values()), "reaching actor input == deployment observation")
class TestWorldModelAndImagination(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0); cls.task = make_task("reaching"); cls.wm = WorldModel(3, 64, 2, True, True); cls.actor = Actor(cls.task.actor_obs_dim, 64, 2); cls.critic = Critic(L.STATE_DIM + 3 + 3, 64, 2)
        class FK(torch.nn.Module):
            def forward(self, q): return torch.stack([0.4 + 0.1 * torch.sin(q[:, 0]) * torch.cos(q[:, 1]), 0.15 + 0.1 * torch.sin(q[:, 2]), 0.45 + 0.1 * torch.cos(q[:, 3])], -1)
        cls.fk = FK()
        class AD:
            lo = torch.full((8,), -3.0); hi = torch.full((8,), 3.0); max_step = torch.full((8,), 0.1)
            def act(self, a, prev): t = prev + a * self.max_step; c = torch.minimum(torch.maximum(t, self.lo), self.hi); return prev + torch.minimum(torch.maximum(c - prev, -self.max_step), self.max_step)
        cls.ad = AD(); cls.B = 16; cls.s0 = torch.randn(cls.B, L.STATE_DIM) * 0.1; cls.goal = torch.randn(cls.B, 3) * 0.1 + 0.4
        cls.wm.update_norms(torch.randn(256, L.STATE_DIM) * 0.2, torch.randn(256, 8) * 0.1, torch.randn(256, L.STATE_DIM) * 0.2)
    def test_one_step_prediction(self):
        a = torch.randn(self.B, 8) * 0.1; s1 = self.wm(self.s0, a); self.assertEqual(s1.shape, self.s0.shape); self.assertTrue(torch.isfinite(s1).all()); torch.testing.assert_close(s1[:, L.PREV], a, msg="prev_ctrl must equal the executed action")
    def test_multistep_imagination_shapes(self):
        for H in (8, 16, 32):
            tr = imagine(self.wm, lambda o, t: self.actor(o), self.task, self.fk, self.ad, self.s0, self.goal, H)
            self.assertEqual(len(tr["states"]), H + 1); self.assertEqual(len(tr["rewards"]), H); self.assertEqual(tr["states"][-1].shape, (self.B, L.STATE_DIM)); self.assertEqual(tr["rewards"][0].shape, (self.B,)); self.assertEqual(tr["actions"][0].shape, (self.B, 8))
    def test_actor_gradient_flows_through_world_model(self):
        H = 8; set_requires_grad(self.wm, False); set_requires_grad(self.critic, False); self.actor.zero_grad(set_to_none=True)
        tr = imagine(self.wm, lambda o, t: self.actor(o), self.task, self.fk, self.ad, self.s0, self.goal, H); v = self.critic(critic_features(self.task, tr["states"][-1], self.goal, tr["tcps"][-1])); J, _ = survival_weighted_return(tr["rewards"], tr["terms"], v, 0.99); (-J.mean()).backward()
        g_full = torch.cat([p.grad.flatten() for p in self.actor.parameters()]); self.assertTrue(torch.isfinite(g_full).all()); self.assertGreater(g_full.norm().item(), 0.0)
        for p in self.wm.parameters(): self.assertIsNone(p.grad, "world-model parameters must not receive gradients during the actor phase")
        # the dynamics path must contribute: cut it (detach the predicted states) and the gradient changes
        self.actor.zero_grad(set_to_none=True)
        orig = self.wm.forward
        def cut(s, a):                                    # keep the clip/slew action path (prev_ctrl := executed action) but cut the learned dynamics
            o = orig(s, a).detach(); return torch.cat([o[:, :L.PREV.start], a, o[:, L.PREV.stop:]], -1)
        self.wm.forward = cut
        try:
            tr2 = imagine(self.wm, lambda o, t: self.actor(o), self.task, self.fk, self.ad, self.s0, self.goal, H); v2 = self.critic(critic_features(self.task, tr2["states"][-1], self.goal, tr2["tcps"][-1])); J2, _ = survival_weighted_return(tr2["rewards"], tr2["terms"], v2, 0.99); (-J2.mean()).backward()
        finally: self.wm.forward = orig
        g_cut = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel()) for p in self.actor.parameters()])
        self.assertGreater((g_full - g_cut).norm().item(), 1e-8, "gradient through the dynamics is zero"); set_requires_grad(self.wm, True); set_requires_grad(self.critic, True)
    def test_bptt_reaches_first_action_from_last_reward(self):
        H = 8; acts = torch.zeros(self.B, H, 8, requires_grad=True); set_requires_grad(self.wm, False)
        tr = imagine(self.wm, lambda o, t: acts[:, t], self.task, self.fk, self.ad, self.s0, self.goal, H); tr["rewards"][-1].sum().backward(); set_requires_grad(self.wm, True)
        self.assertGreater(acts.grad[:, 0].abs().sum().item(), 0.0, "no gradient from the final reward to the FIRST action: BPTT through the dynamics is broken"); self.assertTrue(torch.isfinite(acts.grad).all())
    def test_lambda_returns_terminal_masking(self):
        R = torch.ones(3, 2); V = torch.full((4, 2), 100.0); C = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]); G = lambda_returns(R, V, C, 0.9, 0.95)
        self.assertAlmostEqual(G[0, 1].item(), 1.0, places=5, msg="terminal sample must not bootstrap"); self.assertGreater(G[0, 0].item(), 50.0, "non-terminal sample bootstraps")
        J, surv = survival_weighted_return([torch.ones(2)] * 2, [torch.tensor([0.0, 1.0])] * 2, torch.full((2,), 100.0), 0.9); self.assertAlmostEqual(J[1].item(), 1.0, places=5); self.assertAlmostEqual(surv[1].item(), 0.0)
    def test_critic_gradients_and_update(self):
        tr = imagine(self.wm, lambda o, t: self.actor(o), self.task, self.fk, self.ad, self.s0, self.goal, 8); trd = {k: [x.detach() for x in v] for k, v in tr.items() if k in ("states", "tcps", "rewards", "terms")}
        opt = torch.optim.Adam(self.critic.parameters(), 1e-3); before = [p.clone() for p in self.critic.parameters()]; loss = critic_loss_on(trd, self.critic, self.task, self.goal, 0.99, 0.95, 2, 2, opt)
        self.assertTrue(np.isfinite(loss)); self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, self.critic.parameters())))
class TestReplay(unittest.TestCase):
    def test_windows_stay_in_episode_and_ring_wraps(self):
        n, cap, k = 2, 30, 4; r = Replay(n, cap, 3, seed=0); s = np.zeros((n, L.STATE_DIM), np.float32)
        for t in range(100):
            done = np.array([t % 7 == 6, t % 5 == 4]); st = s + t; r.add(st, np.zeros((n, 3)), np.zeros((n, 8)), st + 1, np.zeros(n), done & False, done, np.zeros((n, 3)))
        self.assertEqual(r.size, cap)
        for _ in range(20):
            b = r.sample_windows(64, k); S = b["states"]; self.assertEqual(S.shape, (64, k + 1, L.STATE_DIM))
            step = S[:, 1:, 0] - S[:, :-1, 0]; self.assertTrue(np.all(step == 1.0), "window crosses an episode boundary or is not contiguous")
        self.assertEqual(r.sample_transitions(10)["state"].shape, (10, L.STATE_DIM)); st, g = r.sample_starts(5); self.assertEqual(g.shape, (5, 3))
    def test_state_dict_roundtrip(self):
        r = Replay(2, 20, 3); 
        for t in range(25): r.add(np.full((2, L.STATE_DIM), t, np.float32), np.zeros((2, 3)), np.zeros((2, 8)), np.full((2, L.STATE_DIM), t + 1, np.float32), np.zeros(2), np.zeros(2, bool), np.zeros(2, bool), np.zeros((2, 3)))
        r2 = Replay(2, 20, 3); r2.load_state_dict(r.state_dict()); self.assertEqual(r2.size, 20); self.assertEqual(r2.state[0, 0, 0], 5.0)
if __name__ == "__main__": unittest.main()
