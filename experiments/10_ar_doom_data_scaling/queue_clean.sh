#!/bin/bash
# AT-1499 clean-data (general data, no planted errors, no re-check lines) control arm.
# usage: queue_clean.sh <gpu> "<depths:N:seed> ..."
W=/lfs/skampere1/0/brando9/at1445; cd $W/src
export CUDA_VISIBLE_DEVICES=$1 HF_HOME=/lfs/skampere1/0/brando9/.cache/huggingface
for cfg in $2; do
  IFS=: read depths n seed <<< "$cfg"
  name=clean_d${depths//,/}_n${n}_s${seed}
  [ -f ../res/$name.json ] && continue
  ../venv/bin/python build_data.py --arm clean --depths $depths --n $n --out ../data/$name.jsonl > ../logs/${name}_build.log 2>&1
  ../venv/bin/python train.py --data ../data/$name.jsonl --out ../runs/$name --seed $seed > ../logs/${name}_train.log 2>&1 || { echo FAIL $name >> ../logs/queue_clean_$1.log; continue; }
  ../venv/bin/python eval.py --adapter ../runs/$name --out ../res/$name.json > ../logs/${name}_eval.log 2>&1
  echo DONE $name $(date +%T) >> ../logs/queue_clean_$1.log
done
echo QUEUE_END $(date +%T) >> ../logs/queue_clean_$1.log
