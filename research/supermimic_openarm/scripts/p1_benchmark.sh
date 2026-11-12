#!/usr/bin/env bash
# Thin wrapper: bounded P1 benchmark on ONE GPU chosen by UUID (measures peak VRAM, env/simulator speed, training-step latency, GPU util, CPU use; trains nothing; refuses a GPU that already runs another compute process).
#   bash research/supermimic_openarm/scripts/p1_benchmark.sh --list
#   bash research/supermimic_openarm/scripts/p1_benchmark.sh --gpu-uuid GPU-xxxx [--budget-s 300 --num-envs 8 --steps 800 --updates 5 --max-vram-gb 8 --allow-shared]
source "$(dirname "$0")/_common.sh"
exec "$VENV/bin/python" -m research.supermimic_openarm.p1_benchmark "$@"
