#!/usr/bin/env bash
# Coverage-experiment queue (frozen by outputs/cov/cov_config.json). Resumable at run granularity (.done markers). New runs only; baselines are never touched.
# Order: ext_c0_seed3 ext_s1_seed3 vol_c0_seed3 vol_s1_seed3 ext_c0_seed4 ext_s1_seed4 ; each run is trained then evaluated (x-sweep, shortcut, open-loop).
cd /home/user/Openarm_research_mujoco; mkdir -p outputs/cov; echo $$ > outputs/cov/queue.pid; export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
EXPECT=f01f4733023b09da42409b1acf66c114f1890d3bf27ead8dc7860ebc8763aef3
[ "$(sha256sum outputs/cov/cov_config.json | cut -d' ' -f1)" = "$EXPECT" ] || { echo "CONFIG CHANGED AFTER FREEZE"; touch outputs/cov/STOP_config; exit 2; }
[ -f outputs/cov/STOP ] && { echo "stop flag"; exit 3; }
PY=/home/user/.venvs/act/bin/python; C=outputs/checkpoints; E=outputs/cov/eval; T=outputs/cov/timing.jsonl; mkdir -p $E
COMMON="--steps 5400 --batch 16 --lr 1e-4 --workers 1 --threads 3 --val-every 250 --save-every 1000 --stats-source train_only"
armflags() { case $1 in c0) echo "";; s1) echo "--state-std-floor 0.02 --gripper-std-floor 0.002";; esac; }
stamp() { echo "{\"event\":\"$1\",\"run\":\"$2\",\"t\":$(date +%s)}" >> $T; }
VAL=$(python3 -c "import json;print(' '.join(map(str,json.load(open('outputs/lerobot/pickplace_v1/openarm_splits.json'))['val'])))")

train() { # branch seed
  br=$1; seed=$2; data=outputs/lerobot/cov_${br%_*}; arm=${br#*_}; name=cov_${br}_seed${seed}; [ -f $C/$name.done ] && return 0
  if [ -d $C/$name ]; then mv $C/$name $C/${name}_interrupted_$(date +%H%M%S); [ -f $C/$name.log ] && mv $C/$name.log $C/${name}_interrupted_$(date +%H%M%S).log; fi
  stamp train_start $name
  $PY policies/act_train.py --dataset $data $COMMON $(armflags $arm) --seed $seed --out $C/$name > $C/$name.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    n=$(wc -l < $C/$name/train_log.jsonl 2>/dev/null || echo 0)
    if [ "$n" -ge 200 ]; then
      avg=$(python3 -c "import json;r=[json.loads(l) for l in open('$C/$name/train_log.jsonl')][199];print(r['elapsed_s']/200)")
      echo "$name first-200-steps average: $avg s/step" >> outputs/cov/guard.log
      python3 -c "import sys;sys.exit(0 if $avg<=1.10 else 1)" || { kill $pid; touch outputs/cov/STOP; echo "RUNTIME GUARD TRIPPED ($avg s/step)" >> outputs/cov/guard.log; exit 4; }
      break
    fi
    sleep 10
  done
  wait $pid && echo DONE > $C/$name.done; stamp train_end $name
}
evalrun() { # branch seed
  br=$1; seed=$2; name=cov_${br}_seed${seed}; CK=$C/$name/last; O=$E/${br}_seed${seed}; [ -f $O.done ] && return 0; [ -f $C/$name.done ] || return 1
  mkdir -p $O; stamp eval_start $name
  [ -f $O/sweep/act/sweep/episodes.jsonl ] || $PY eval/xsweep_run.py --ckpt $CK --out $O/sweep --workers 4 > $O/sweep.log 2>&1
  r() { tag=$1; shift 1; [ -f $O/$tag/act/nominal/summary.json ] && return 0
        $PY eval/evaluate.py --agent act --ckpt $CK --workers 4 --save-rollouts --no-video --seed 21 --conditions nominal --ids $(seq 0 19) --out $O/$tag "$@" > $O/$tag.log 2>&1; }
  r unpert20; r pert_right_mean --state-perturb right_mean; r pert_jit_small --state-perturb right_jitter_small
  $PY policies/open_loop_check.py $CK outputs/lerobot/pickplace_v1 --episodes $VAL --stride 4 --out $O/open_loop_val.json > $O/open_loop_val.log 2>&1
  echo done > $O.done; stamp eval_end $name
}
for pair in "ext_c0 3" "ext_s1 3" "vol_c0 3" "vol_s1 3" "ext_c0 4" "ext_s1 4"; do set -- $pair; train $1 $2; [ -f outputs/cov/STOP ] && exit 4; evalrun $1 $2; done
touch outputs/cov/cov.done
