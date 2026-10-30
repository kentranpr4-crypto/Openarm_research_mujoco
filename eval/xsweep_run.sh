#!/usr/bin/env bash
# x-sweep queue (frozen by outputs/xsweep/xsweep_config.json). Resumable (.done markers). Evaluation only; no training; no forbidden seeds (poses are explicit).
cd /home/user/Openarm_research_mujoco; mkdir -p outputs/xsweep/eval; echo $$ > outputs/xsweep/queue.pid; export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
EXPECT=6c59292d6d951673ab813fc50eba38612c01777534ccdad2a6f3c5797b1f596c
[ "$(sha256sum outputs/xsweep/xsweep_config.json | cut -d' ' -f1)" = "$EXPECT" ] || { echo "CONFIG CHANGED AFTER FREEZE"; touch outputs/xsweep/STOP_config; exit 2; }
PY=/home/user/.venvs/act/bin/python; T=outputs/xsweep/timing.jsonl; stamp() { echo "{\"event\":\"$1\",\"run\":\"$2\",\"t\":$(date +%s)}" >> $T; }
for pair in "c0 0" "s1 0" "c0 1" "s1 1" "c0 2" "s1 2"; do set -- $pair; arm=$1; s=$2
  O=outputs/xsweep/eval/${arm}_seed$s; [ -f $O.done ] && continue; [ -f outputs/xsweep/STOP ] && exit 4
  if [ $s = 0 ]; then CK=outputs/checkpoints/e6b_${arm}_seed0/last; else CK=outputs/checkpoints/e0_${arm}_seed$s/last; fi
  [ -d $O ] && mv $O ${O}_partial_$(date +%H%M%S); stamp start ${arm}_seed$s
  $PY eval/xsweep_run.py --ckpt $CK --out $O --workers 4 > $O.log 2>&1 && echo DONE > $O.done; stamp end ${arm}_seed$s
done
touch outputs/xsweep/xsweep.done
