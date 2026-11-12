#!/usr/bin/env bash
# P1 (pushing) ONE-SEED PILOT launcher.  DEFAULT = SAFE DRY-RUN (plan + read-only resource preflight; trains NOTHING).
#   --gpu-uuid GPU-xxxx (required; list with:  python -m research.supermimic_openarm.p1_benchmark --list)  --seed 1000  --run-name NAME  --max-hours H (<= 2, hard cap)  --num-envs N  --resume  --execute
# The GPU is selected by UUID (CUDA_VISIBLE_DEVICES=<uuid>), never by index: CUDA's default device order differs from nvidia-smi's, so an index can silently land on another job's GPU.
# Read-only preflight: refuses (exit 3, RESOURCE_BLOCKED) if that GPU has ANY other compute process, is busy, short of free memory, or there are too few spare CPU cores / disk.  It never kills or modifies other jobs (e.g. OpenArm-WAM),
# runs at low CPU priority, writes to its OWN new run directory (train.py refuses an existing one), stops at the hard time cap (SIGTERM; at most `checkpoint_interval` iterations are lost; continue with --resume).
# --execute is NOT to be used before: (1) a measured p1_benchmark on this UUID, (2) the gates file hash recorded, (3) separate explicit approval.
source "$(dirname "$0")/_common.sh"
UUID=""; SEED=1000; RUN=""; HOURS=2; NENVS=24; RESUME=0; EXEC=0
while [ $# -gt 0 ]; do case "$1" in
  --gpu-uuid) UUID="$2"; shift 2;; --seed) SEED="$2"; shift 2;; --run-name) RUN="$2"; shift 2;; --max-hours) HOURS="$2"; shift 2;; --num-envs) NENVS="$2"; shift 2;; --resume) RESUME=1; shift;; --execute) EXEC=1; shift;; --dry-run) EXEC=0; shift;; -h|--help) sed -n 2,9p "$0"; exit 0;; *) echo "unknown argument: $1" >&2; exit 2;; esac; done
[ -n "$UUID" ] || { echo "ERROR: --gpu-uuid is required (no device is chosen implicitly; see p1_benchmark --list)" >&2; exit 2; }
awk "BEGIN{exit !($HOURS > 0 && $HOURS <= 2)}" || { echo "ERROR: --max-hours must be in (0, 2] for the pilot" >&2; exit 2; }
RUN="${RUN:-pushing_p1_pilot_s$SEED}"; CFG="research/supermimic_openarm/configs/pushing_p1_pilot.yaml"; OUTDIR="outputs/supermimic_openarm/$RUN/seed$SEED"; LOGDIR="outputs/supermimic_openarm/_logs"; SECS=$(awk "BEGIN{printf \"%d\", $HOURS*3600}")
echo "== plan: config $CFG | seed $SEED | run $RUN | gpu $UUID | num_envs $NENVS | hard cap ${HOURS} h | output $OUTDIR"
echo "== read-only preflight"; PF=0
py - "$UUID" "$NENVS" <<'PY' || PF=$?
import os, shutil, subprocess, sys
uuid, nenv = sys.argv[1], int(sys.argv[2]); bl = []
def q(args):
    try: return subprocess.run(args, capture_output=True, text=True, timeout=20)
    except Exception: return None
cores = os.cpu_count() or 1; load = os.getloadavg()[0]; spare = max(0, int(cores - load)); print(f"  CPU: {cores} cores, load1 {load:.1f}, spare ~{spare}")
if spare < nenv + 2: bl.append(f"spare CPU cores ~{spare} < num_envs {nenv} + 2")
free_gb = shutil.disk_usage(".").free / 2 ** 30; print(f"  disk free: {free_gb:.1f} GB")
if free_gb < 10: bl.append(f"free disk {free_gb:.1f} GB < 10 GB")
g = q(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
if not g or g.returncode != 0: bl.append("nvidia-smi unavailable: no GPU visible")
else:
    row = next(([x.strip() for x in l.split(",")] for l in g.stdout.strip().splitlines() if uuid in l), None)
    if row is None: bl.append(f"GPU UUID {uuid} not found")
    else:
        procs = q(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory,process_name", "--format=csv,noheader,nounits"]); mine = [l for l in (procs.stdout.strip().splitlines() if procs else []) if uuid in l]; free = (int(row[3]) - int(row[4])) / 1024
        print(f"  GPU {row[0]} {row[2]} ({uuid}): free {free:.1f} GiB | util {row[5]}% | compute processes on it: {len(mine)}")
        if mine: bl.append(f"GPU {row[0]} already has {len(mine)} compute process(es)")
        if int(row[5]) > 30: bl.append(f"GPU utilisation {row[5]}% > 30%")
        if free < 8: bl.append(f"GPU free memory {free:.1f} GiB < 8 GiB")
        o = q(["ps", "-eo", "pid,args"]); wam = [l for l in (o.stdout.splitlines() if o else []) if "research.openarm_wam" in l and "grep" not in l]
        if wam: print(f"  note: {len(wam)} OpenArm-WAM process(es) running (untouched; the per-GPU check above decides)")
for b in bl: print("  BLOCKER:", b)
sys.exit(3 if bl else 0)
PY
if [ "$PF" -ne 0 ]; then echo "RESOURCE_BLOCKED (exit 3)"; exit 3; fi
if [ "$RESUME" = 1 ]; then ARGS=(--resume "$OUTDIR"); else ARGS=(--config "$CFG" --run-name "$RUN" --override experiment.seed="$SEED" --override env.num_envs="$NENVS" --override system.device=cuda:0); fi
CMD="CUDA_VISIBLE_DEVICES=$UUID nice -n 5 timeout --signal=TERM --kill-after=60 $SECS $VENV/bin/python -m research.supermimic_openarm.train ${ARGS[*]}"
if [ "$EXEC" != 1 ]; then echo "== DRY-RUN (add --execute to train; nothing was started). Exact command that would run:"; echo "$CMD"; py -m research.supermimic_openarm.train --config "$CFG" --run-name "$RUN" --override experiment.seed="$SEED" --override env.num_envs="$NENVS" --override system.device=cuda:0 --dry-run | tail -2; echo "READY"; exit 0; fi
mkdir -p "$LOGDIR"; echo "== EXECUTING (hard cap ${HOURS} h); log: $LOGDIR/$RUN.log"; export CUDA_VISIBLE_DEVICES="$UUID"
nice -n 5 timeout --signal=TERM --kill-after=60 "$SECS" "$VENV/bin/python" -m research.supermimic_openarm.train "${ARGS[@]}" > "$LOGDIR/$RUN.log" 2>&1; RC=$?
echo "trainer exit code $RC (124 = hard time cap reached; use --resume to continue)"; tail -n 3 "$OUTDIR/metrics.jsonl" 2>/dev/null | cut -c1-300; exit $RC
