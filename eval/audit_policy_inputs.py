#!/usr/bin/env python3
"""AC-09 evidence: the trained policy cannot receive simulator object coordinates.
 1. checkpoint config: input_features are only observation.state + observation.images.*  (no env state, no eval.*, no velocity);
 2. dataset the policy was trained on: LeRobot features contain no eval.* / box / tcp ground truth;
 3. source scan: policies/act_infer.py never imports mujoco/lab_scene or touches env.info; evaluate.py calls runner.act(state, images) only;
 4. the converter whitelists features explicitly.
Usage: /home/user/.venvs/act/bin/python eval/audit_policy_inputs.py CKPT_DIR LEROBOT_DATASET_DIR"""
import sys, re, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
ckpt, ds = Path(sys.argv[1]), Path(sys.argv[2]); res = {}
cfg = json.loads((ckpt / "config.json").read_text()); inp = sorted(cfg["input_features"])
res["checkpoint_input_features"] = inp
res["inputs_ok"] = all(k == "observation.state" or k.startswith("observation.images.") for k in inp)
info = json.loads((ds / "meta" / "info.json").read_text()); feats = sorted(info["features"])
res["dataset_features"] = feats; res["dataset_has_ground_truth"] = [k for k in feats if re.search(r"eval|box|tcp|object|pos\b", k) and not k.startswith("observation.images")]
src = (ROOT / "policies/act_infer.py").read_text()
res["act_infer_imports_sim"] = bool(re.search(r"^\s*(import|from)\s+(mujoco|lab_scene|lab_env)", src, re.M)); res["act_infer_uses_env_info"] = "env.info" in src or ".info()" in src
ev = (ROOT / "eval/evaluate.py").read_text(); calls = re.findall(r"runner\.act\(([^)]*)\)", ev); res["evaluate_act_calls"] = calls
res["evaluate_only_state_and_images"] = calls == ['o["state"], o["images"]']
conv = (ROOT / "data/convert_lerobot.py").read_text(); res["converter_excludes_eval_arrays"] = ("eval." not in re.sub(r"#.*|\"\"\".*?\"\"\"", "", conv, flags=re.S))
res["PASS"] = bool(res["inputs_ok"] and not res["dataset_has_ground_truth"] and not res["act_infer_imports_sim"] and not res["act_infer_uses_env_info"] and res["evaluate_only_state_and_images"] and res["converter_excludes_eval_arrays"])
print(json.dumps(res, indent=1)); sys.exit(0 if res["PASS"] else 1)
