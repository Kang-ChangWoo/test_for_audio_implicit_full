#!/bin/bash
# Aggregation / anti-discreteness suite (cause 2 = coarse token grid; + combo):
#   Bnode2_cross_hitokflip     : cross + /4 hi-tokens + flip   (finer audio tokens -> less block)
#   Bnode2_crossself_hitokflip : crossself + /4 hi-tokens + flip (ray self-attn + finer tokens)
# Together with extra7 (crossself+flip) this tests both root causes of cross's discrete look.
cd "$(dirname "$0")"
ML=logs/_extra8.log
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ML"; }

say "extra8 queued — waiting for run_extra7 ..."
while pgrep -f run_extra7.sh >/dev/null 2>&1; do sleep 60; done
say "clear — launching cross_hitokflip + crossself_hitokflip"

j () { g=$1; n=$2; s=$3; mdl=$4; bs=$5
  CUDA_VISIBLE_DEVICES=$g bash -c \
   "python train.py --model $mdl --run-name $n --seed $s --epochs 25 --batch-size $bs \
      --n-rays 2048 --num-workers 6 --lr 3e-4 --in-ch 2 --hi-tokens True --flip-aug True \
    && python eval.py --run-name $n --controls True" > logs/$n.log 2>&1 &
  echo "launched $n gpu$g pid $!"; }

j 0 Bnode2_cross_hitokflip_s0 0 cross 12
j 1 Bnode2_cross_hitokflip_s1 1 cross 12
j 2 Bnode2_cross_hitokflip_s2 2 cross 12
j 3 Bnode2_crossself_hitokflip_s0 0 crossself 8
j 4 Bnode2_crossself_hitokflip_s1 1 crossself 8
j 5 Bnode2_crossself_hitokflip_s2 2 crossself 8
wait
say "extra8 jobs done — aggregate + push"
python agg_full.py > logs/_agg_extra8.log 2>&1
python update_readme.py >> "$ML" 2>&1
TOKEN=$(grep github.com ~/.git-credentials | head -1 | sed -E 's#^https?://##; s#@github.com.*##; s#^[^:]*:##')
git add -A
git -c user.name="Kang-ChangWoo" -c user.email="branden.c.w.kang@gmail.com" \
    commit -q -m "Add hi-token + crossself aggregation suite (anti-discreteness)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" \
  && git -c credential.helper="!f(){ echo username=Kang-ChangWoo; echo password=$TOKEN; };f" \
     push origin main >> "$ML" 2>&1 && say "extra8 pushed" || say "extra8 nothing to push"
say "extra8 complete"
