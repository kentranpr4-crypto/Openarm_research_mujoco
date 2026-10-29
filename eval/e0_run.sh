#!/usr/bin/env bash
# E0 queue (frozen by outputs/e0/e0_config.json). Resumable at run granularity (.done markers). New runs only: C0/S1 x seeds 1,2.
# Seed-0 runs are REUSED from Stage 1 and never touched. Dev eval seeds 21/22 only; anything else aborts.
cd /home/user/Openarm_research_mujoco; mkdir -p outputs/e0; echo $$ > outputs/e0/queue.pid; export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
EXPECT=25f20ea4c887534ddc80c8aedee01e93da05790dac182d1b0a86f4217d545e9d
[ "$(sha256sum outputs/e0/e0_config.json | cut -d' ' -f1)" = "$EXPECT" ] || { echo "CONFIG CHANGED AFTER FREEZE - abort"; touch outputs/e0/STOP_config_changed; exit 2; }
[ -f outputs/e0/STOP ] && { echo "stop flag present"; exit 3; }
PY=/home/user/.venvs/act/bin/python; D=outputs/lerobot/pickplace_v1; C=outputs/checkpoints; E=outputs/eval/e0; T=outputs/e0/timing.jsonl; mkdir -p $E
COMMON="--dataset $D --steps 5400 --batch 16 --lr 1e-4 --workers 1 --threads 3 --val-every 250 --save-every 1000 --stats-source train_only"
armflags() { case $1 in c0) echo "";; s1) echo "--state-std-floor 0.02 --gripper-std-floor 0.002";; esac; }
stamp() { echo "{\"event\":\"$1\",\"run\":\"$2\",\"t\":$(date +%s)}" >> $T; }

train() {   # arm seed
  arm=$1; seed=$2; name=e0_${arm}_seed${seed}; [ -f $C/$name.done ] && return 0
  if [ -d $C/$name ]; then mv $C/$name $C/${name}_interrupted_$(date +%H%M%S); [ -f $C/$name.log ] && mv $C/$name.log $C/${name}_interrupted_$(date +%H%M%S).log; fi
  stamp train_start $name
  $PY policies/act_train.py $COMMON $(armflags $arm) --seed $seed --out $C/$name > $C/$name.log 2>&1 &
  pid=$!
  # runtime guard (pre-registered): average speed over the first 200 steps must be <= 1.10 s/step
  while kill -0 $pid 2>/dev/null; do
    n=$(wc -l < $C/$name/train_log.jsonl 2>/dev/null || echo 0)
    if [ "$n" -ge 200 ]; then
      avg=$(python3 -c "import json;r=[json.loads(l) for l in open('$C/$name/train_log.jsonl')][199];print(r['elapsed_s']/200)")
      echo "$name first-200-steps average: $avg s/step" >> outputs/e0/guard.log
      python3 -c "import sys;sys.exit(0 if $avg<=1.10 else 1)" || { kill $pid; touch outputs/e0/STOP; echo "RUNTIME GUARD TRIPPED ($avg s/step)" >> outputs/e0/guard.log; exit 4; }
      break
    fi
    sleep 10
  done
  wait $pid && echo DONE > $C/$name.done; stamp train_end $name
}

evalrun() { # arm seed
  arm=$1; seed=$2; name=e0_${arm}_seed${seed}; CK=$C/$name/last; O=$E/${arm}_seed${seed}; [ -f $O/arm.done ] && return 0; mkdir -p $O; stamp eval_start $name
  r() { tag=$1; sd=$2; shift 2
        case $sd in 21|22) ;; *) echo "FORBIDDEN EVAL SEED $sd" >&2; touch outputs/e0/STOP; exit 5;; esac
        [ -f $O/$tag/act/*/summary.json ] && return 0
        $PY eval/evaluate.py --agent act --ckpt $CK --workers 4 --save-rollouts --no-video --seed $sd --out $O/$tag "$@" > $O/$tag.log 2>&1; }
  r nom21 21 --conditions nominal --n 30
  r bound21 21 --conditions unseen_pose --n 30
  r bound22 22 --conditions unseen_pose --n 30
  r pert_right_mean 21 --conditions nominal --ids $(seq 0 19) --state-perturb right_mean
  r pert_jit_small 21 --conditions nominal --ids $(seq 0 19) --state-perturb right_jitter_small
  $PY policies/open_loop_check.py $CK $D --episodes $($PY -c "import json;print(' '.join(map(str,json.load(open('$D/openarm_splits.json'))['val'])))") --stride 4 --out $O/open_loop_val.json > $O/open_loop_val.log 2>&1
  echo done > $O/arm.done; stamp eval_end $name
}
for pair in "c0 1" "s1 1" "c0 2" "s1 2"; do set -- $pair; train $1 $2; [ -f outputs/e0/STOP ] && exit 4; evalrun $1 $2; done
echo E0_DONE > outputs/e0/e0.done
