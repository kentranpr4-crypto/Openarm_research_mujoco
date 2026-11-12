"""Unit tests of the opt-in P1 variants (B1 trust region, B2 ensemble penalty, B3 contact gating).  Single update calls on a tiny CPU replay; NO training loop is run.  Also checks that B0 is the untouched original path."""
import contextlib, io
from pathlib import Path
import numpy as np, pytest, torch
from research.supermimic_openarm import config as C, p1_train_variant as PV, p1_variants as V, trainer as T, train as TRAIN
from research.supermimic_openarm.envs import layout as L
from research.supermimic_openarm.envs.openarm_env import OpenArmTaskEnv
PKG = Path(__file__).resolve().parents[1]
def cfg(tmp, **over):
    o = {"env.task": "pushing", "env.max_steps": 30, "env.num_envs": 2, "env.backend": "sync", "imagination.horizon": 8, "imagination.batch_size": 32, "imagination.chunk_size": 16, "world_model.batch_size": 32, "world_model.window": 3, "experiment.output_dir": str(tmp), **over}
    return C.load_config(str(PKG / "configs/debug_cpu.yaml"), [f"{k}={v}" for k, v in o.items()])
def trainer(tmp, variant, **v):
    tr = V.P1VariantTrainer(cfg(tmp), Path(tmp) / "run", variant, log=lambda *a: None, **v); tr._make_vec(); tr.collect(10, random=True); return tr
def test_b0_is_the_original_path(tmp_path):
    a, b = io.StringIO(), io.StringIO(); args = ["--config", str(PKG / "configs/debug_cpu.yaml"), "--override", f"experiment.output_dir={tmp_path}", "--run-name", "x", "--dry-run"]
    with contextlib.redirect_stdout(a): rc1 = PV.main(["--variant", "b0", *args])
    with contextlib.redirect_stdout(b): rc2 = TRAIN.main(args)
    assert rc1 == rc2 == 0 and a.getvalue() == b.getvalue()                                                                      # byte-identical output: B0 == train.py
    assert PV.main(["--variant", "b0", "--variant-arg", "rho=0.1", *args]) == 2
    assert not hasattr(T, "P1VariantTrainer") and "p1_variants" not in open(T.__file__).read()                              # the original trainer does not know about the variants
def test_unknown_variant_arguments_are_rejected(tmp_path):
    with pytest.raises(ValueError): V.P1VariantTrainer(cfg(tmp_path), Path(tmp_path) / "r", "b0")
    with pytest.raises(ValueError): V.P1VariantTrainer(cfg(tmp_path), Path(tmp_path) / "r", "b1", bogus=1)
    assert PV.main(["--variant", "b1", "--variant-arg", "bogus=1", "--config", str(PKG / "configs/debug_cpu.yaml"), "--dry-run"]) == 2
def test_b3_gate_attenuates_a_still_box_in_free_space_and_opens_near_the_pusher():
    c = C.Config(); c.env.task = "pushing"; dev = torch.device("cpu"); fk, _, _ = T.build_fk_and_adapter(c, dev); env = OpenArmTaskEnv("pushing", None, 30, 20); s, g, tcp = env.reset(23_600_000); torch.manual_seed(0)
    gated = V.GatedWorldModel(2, 64, 2, True, False, fk); gated.update_norms(torch.randn(256, L.STATE_DIM), torch.rand(256, 8), torch.randn(256, L.STATE_DIM)); gated.eval(); T0 = torch.as_tensor(tcp)[None]; b0 = L.BOX_POS.start
    far = torch.as_tensor(s)[None].clone(); far[:, L.BOX_POS] = T0 + torch.tensor([[0.5, 0.0, 0.0]]); far[:, L.BOX_VEL] = 0.0; near = far.clone(); near[:, L.BOX_POS] = T0 + torch.tensor([[0.03, 0.0, 0.0]]); moving = far.clone(); moving[:, L.BOX_VEL] = torch.tensor([[0.3, 0.0, 0.0]]); a = torch.as_tensor(env.prev_ctrl[:8], dtype=torch.float32)[None]
    with torch.no_grad():
        ungated = V.WorldModel.forward(gated, far, a); out = gated(far, a); g_far = gated.gate(far).item()
        assert g_far < 0.06                                                                                                          # at initialisation the rest-velocity branch is sigmoid(-kv*v0) = 0.047 (learned afterwards); the proximity branch is ~0
        assert (out[:, b0:] - far[:, b0:]).abs().max() <= g_far * (ungated[:, b0:] - far[:, b0:]).abs().max() + 1e-7                  # the predicted box change is the ungated change scaled by the gate
        assert gated.gate(near).item() > 0.9 and gated.gate(moving).item() > 0.9                                                    # pusher within reach, or box already sliding: gate open
        un_near = V.WorldModel.forward(gated, near, a); assert torch.allclose(gated(near, a)[:, b0:] - near[:, b0:], gated.gate(near) * (un_near[:, b0:] - near[:, b0:]), atol=1e-6)
    assert "r0" in gated.state_dict() and not any(k.startswith("_fk") for k in gated.state_dict())                                  # fk is not serialised
def test_b1_policy_stays_inside_the_trust_region(tmp_path):
    tr = trainer(tmp_path, "b1", rho=0.05)
    with torch.no_grad():
        for p in tr.actor.net.parameters(): p.add_(torch.randn_like(p) * 0.5)                                                      # make the actor differ strongly from its reference
    obs = torch.randn(16, tr.task.actor_obs_dim); pol = tr.actor_policy(); a = pol(obs, 0)
    with torch.no_grad(): ref = torch.tanh(tr.ref_net(tr.actor.obs_norm(obs)))
    assert (a - ref).abs().max() <= 0.05 + 1e-6 and (tr.actor(obs) - ref).abs().max() > 0.05                                        # clamped inside the radius although the raw actor is far away
    out = tr.update_actor_critic(); assert out["bad_grads"] == 0 and np.isfinite(out["actor_grad_norm"]) and "b1_frac_outside_radius" in out and 0.0 <= out["b1_frac_outside_radius"] <= 1.0
def test_b2_ensemble_penalty_runs_and_restores_imagine(tmp_path):
    tr = trainer(tmp_path, "b2", lam=3.0, members=3); orig = T.imagine; m = tr.update_world_model(); assert "wm_loss_member1" in m and "wm_loss_member2" in m and len(tr.members) == 2 and np.isfinite(m["wm_loss"])
    out = tr.update_actor_critic(); assert T.imagine is orig                                                                         # the module-level hook is restored (also after exceptions: try/finally)
    assert out["bad_grads"] == 0 and np.isfinite(out["actor_grad_norm"]) and out["b2_mean_box_disagreement_cm"] >= 0.0 and all(p.requires_grad for mm in tr.members for p in mm.parameters())
    tr0 = trainer(tmp_path / "l0", "b2", lam=0.0, members=3); assert np.isfinite(tr0.update_actor_critic()["imag_objective"])
def test_b3_trainer_updates_and_checkpoint_roundtrip(tmp_path):
    tr = trainer(tmp_path, "b3", gate_r0=0.12); assert isinstance(tr.wm, V.GatedWorldModel) and abs(float(tr.wm.r0.detach()) - 0.12) < 1e-6; r0_before = float(tr.wm.r0.detach())
    m = tr.update_world_model(); assert np.isfinite(m["wm_loss"]) and float(tr.wm.r0.detach()) != r0_before                                  # the gate parameters are trained with the world model
    out = tr.update_actor_critic(); assert out["bad_grads"] == 0; tr.save("latest.pt"); ck = torch.load(Path(tr.ckpt_dir) / "latest.pt", weights_only=False); assert ck["variant"] == "b3" and ck["variant_args"]["gate_r0"] == 0.12
    wm2 = V.build_world_model("b3", 2, tr.cfg.world_model, tr.fk, torch.device("cpu"), gate_r0=0.12); wm2.load_state_dict(ck["wm"]); assert float(wm2.r0.detach()) == float(tr.wm.r0.detach())                         # what p1_wm_eval does for a b3 checkpoint
    tr2 = V.P1VariantTrainer(tr.cfg, tr.run_dir, "b3", resume=True, log=lambda *a: None); assert float(tr2.wm.r0.detach()) == float(tr.wm.r0.detach())
    with pytest.raises(ValueError): V.P1VariantTrainer(tr.cfg, tr.run_dir, "b1", resume=True, log=lambda *a: None)                                                     # a checkpoint of another variant is refused
def test_load_actor_works_for_every_variant_checkpoint(tmp_path):
    for v in ("b1", "b2", "b3"):
        tr = trainer(Path(tmp_path) / v, v); tr.save("latest.pt"); ck = torch.load(Path(tr.ckpt_dir) / "latest.pt", weights_only=False); actor, adapter = V.load_actor(tr.cfg, ck, torch.device("cpu"), tr.task)
        obs = torch.randn(3, tr.task.actor_obs_dim); assert torch.allclose(actor(obs), tr.actor(obs), atol=1e-6) and adapter is not None
def test_ab_protocol_references_the_registered_gates_and_four_variants():
    import hashlib, yaml
    from research.supermimic_openarm import p1_gate_check as GC
    d = yaml.safe_load((PKG / "experiments/p1_actor_ab_protocol.yaml").read_text()); assert d["protocol"]["gates_sha256_registered"] == GC.REGISTERED_SHA256 == hashlib.sha256(Path(GC.GATES).read_bytes()).hexdigest()
    assert set(d["variants"]) == {"b0", "b1", "b2", "b3"} and d["shared"]["config"].endswith("pushing_p1_pilot.yaml") and d["phases"]["pilot_0"]["variants"] == ["b0"]
    assert "28,000,000" in d["data_separation"]["final_evaluation"]["seeds"] and "9,000,000" in d["data_separation"]["checkpoint_selection"]["seeds"]          # selection and final evaluation use different goals
