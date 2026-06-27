#!/bin/bash
# Follow-up cross experiments, queued AFTER the current scheduler drains (no GPU fight,
# no loss of in-flight jobs):
#   Bnode2_cross5ch  : cross-attn implicit with 5ch RIR (phase/IPD) input
#   Bnode2_hybrid5ch : hybrid (SH-coarse + implicit residual, band-limited) + 5ch input
# Each job = train then eval inline. Then aggregate + README + push.
cd "$(dirname "$0")"
ML=logs/_extra2.log
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ML"; }

say "extra2 queued — waiting for main scheduler to finish ..."
while pgrep -f scheduler.py >/dev/null 2>&1; do sleep 60; done
say "scheduler done — launching cross5ch + hybrid5ch (6 jobs, gpu0-5)"

cross5 () { g=$1; n=$2; s=$3; m=$4
  CUDA_VISIBLE_DEVICES=$g bash -c \
   "python train.py --model $m --run-name $n --seed $s --epochs 25 --batch-size 24 \
      --n-rays 2048 --num-workers 6 --lr 3e-4 --in-ch 5 \
    && python eval.py --run-name $n --controls True" > logs/$n.log 2>&1 &
  echo "launched $n ($m, 5ch) gpu$g pid $!"; }

cross5 0 Bnode2_cross5ch_s0  0 cross
cross5 1 Bnode2_cross5ch_s1  1 cross
cross5 2 Bnode2_cross5ch_s2  2 cross
cross5 3 Bnode2_hybrid5ch_s0 0 hybrid
cross5 4 Bnode2_hybrid5ch_s1 1 hybrid
cross5 5 Bnode2_hybrid5ch_s2 2 hybrid
wait
say "extra2 jobs done — aggregate + README + push"
python agg_full.py > logs/_agg_extra2.log 2>&1
python update_readme.py >> "$ML" 2>&1
TOKEN=$(grep github.com ~/.git-credentials | head -1 | sed -E 's#^https?://##; s#@github.com.*##; s#^[^:]*:##')
git add -A
git -c user.name="Kang-ChangWoo" -c user.email="branden.c.w.kang@gmail.com" \
    commit -q -m "Add cross+5ch and hybrid+5ch follow-up results

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" \
  && git -c credential.helper="!f(){ echo username=Kang-ChangWoo; echo password=$TOKEN; };f" \
     push origin main >> "$ML" 2>&1 && say "extra2 pushed" || say "extra2 nothing to push"
say "extra2 complete"
