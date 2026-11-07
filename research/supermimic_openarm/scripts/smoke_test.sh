#!/usr/bin/env bash
# CPU/environment smoke tests (minutes).  Add --gpu to also run the 1000-step GPU benchmark.  Does not start real training.
source "$(dirname "$0")/_common.sh"
GPU=0; [ "${1:-}" = "--gpu" ] && GPU=1
echo "== 1/5 unit tests"; py -m unittest discover -s research/supermimic_openarm/tests -t . 2>&1 | tail -5
echo "== 2/5 configuration validation (dry-run)"; for c in research/supermimic_openarm/configs/*.yaml; do py -m research.supermimic_openarm.train --config "$c" --dry-run >/dev/null && echo "ok $(basename "$c")"; done
RUN="smoke_$(date +%Y%m%dT%H%M%S)"
echo "== 3/5 CPU mini-training ($RUN)"; py -m research.supermimic_openarm.train --config research/supermimic_openarm/configs/debug_cpu.yaml --run-name "$RUN" | tail -3
echo "== 4/5 evaluation of the smoke checkpoint"; py -m research.supermimic_openarm.evaluate --checkpoint "outputs/supermimic_openarm/$RUN/seed1000/checkpoints/latest.pt" --task reaching --episodes 2 | grep -E "success_rate|wm_rollout|latency"
echo "== 5/5 CPU benchmark"; py -m research.supermimic_openarm.benchmark --config research/supermimic_openarm/configs/debug_cpu.yaml --steps 64 --updates 1 | grep -E "transitions_per_s|verdict"
if [ "$GPU" = 1 ]; then echo "== GPU gate: check + 1000-step benchmark (hard VRAM cap from the config)"; "$SM_DIR/scripts/check_gpu.sh"; py -m research.supermimic_openarm.benchmark --config research/supermimic_openarm/configs/reaching_4090.yaml --steps 1000; fi
echo "SMOKE TESTS FINISHED. This is NOT a go-ahead for long training: read the GO/NO-GO gates in README.md."
