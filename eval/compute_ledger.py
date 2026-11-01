#!/usr/bin/env python3
"""CPU-hour ledger (cumulative from WM-2 onwards; training, evaluation, dataset preparation and simulation; excludes historical ACT research).  Two kinds of numbers:
  (a) MEASURED since the current VM boot: cgroup cpuacct.usage (all processes of the sandbox) + per-process getrusage totals stored in manifests;
  (b) ESTIMATED before the boot (Oct 31 sessions) from log wall times: lower bound = wall x 1 thread, upper bound = wall x configured threads (3 for NN training) / x 4 workers for simulation.   Run: python eval/compute_ledger.py"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
def cg(): return int(open("/sys/fs/cgroup/cpuacct/cpuacct.usage").read()) / 1e9
training_wall = {"WM-2 dry run": 280, "WM-2 final learned run": 2446, "WM-2 GRU smoke/diagnostics (est.)": 120, "WM-2.1 reproduction + audit (2418 + ~100)": 2518, "WM3-B training incl. ridge + crash": 2160, "WM3-B reload run": 87, "WM3-B unit tests (earlier runs, est.)": 300}
sim_pre = {"WM-1 intervention full collection (4 workers)": 639 * 4, "WM-1 pilot collection (est., 4 workers)": 100 * 4}; build_pre = {"WM-2 dataset build + determinism check + windows (est.)": 300}
lo = sum(training_wall.values()) / 3600 + sum(build_pre.values()) / 3600 + sum(sim_pre.values()) / 3600; hi = sum(training_wall.values()) * 3 / 3600 + sum(build_pre.values()) / 3600 + sum(sim_pre.values()) / 3600
meas = cg() / 3600; man = {}
for name, p in (("v1.1 build", "outputs/wm2_v1_1/dataset/dataset_manifest.json"), ("v2 build", "outputs/wm2_v2/dataset/dataset_manifest.json")):
    try: man[name] = json.loads((ROOT / p).read_text())["cpu_seconds"] / 3600
    except Exception: man[name] = None
for name, p in (("v1.1 integrity", "outputs/wm2_v1_1/integrity.json"), ("v2 integrity", "outputs/wm2_v2/integrity.json")):
    try: man[name] = json.loads((ROOT / p).read_text())["cpu_seconds"] / 3600
    except Exception: man[name] = None
for name, p_ in (("E1-lite", "outputs/e1_lite/e1_lite_results.json"), ("E1-lite-B", "outputs/e1_lite_b/results.json"), ("E1-lite-C (final run)", "outputs/e1_lite_c/results.json")):
    try: man[name] = json.loads((ROOT / p_).read_text())["budget"]["cpu_hours"]
    except Exception: man[name] = None
man["E1-lite-C crashed first attempt (upper bound, wall 51 s x 3 threads)"] = 160 / 3600
tot_lo, tot_hi = lo + meas, hi + meas
res = dict(convention="22 CPU-h cumulative from WM-2 onwards incl. training, evaluation, dataset preparation, simulation; excludes historical ACT research", pre_boot_estimate_cpu_h=dict(lower=lo, upper=hi, items=dict(training_and_eval_wall_s=training_wall, simulation_cpu_s=sim_pre, dataset_prep_s=build_pre)),
           measured_since_vm_boot_cpu_h=meas, measured_note="cgroup cpuacct.usage of the sandbox since boot (03:36 UTC Nov 1): includes I-1 collection, I-1 predictions/evaluation, yaw audits, Gate v3 tests, replay validation, v1.1 build, v2 build, integrity, E1-lite as far as they have run; also tool overheads",
           per_process_measured_cpu_h=man, total_used_cpu_h=dict(lower=tot_lo, upper=tot_hi), remaining_cpu_h=dict(lower=22 - tot_hi, upper=22 - tot_lo))
(ROOT / "outputs/contract_v2/compute_ledger.json").write_text(json.dumps(res, indent=1, default=str)); print(json.dumps(res, indent=1, default=str))
