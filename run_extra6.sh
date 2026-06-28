#!/bin/bash
# #3(a) Uncertainty-as-product: probabilistic head (K hypotheses + per-pixel Laplace
# scale) on the richest input (5ch phase). Output = depth + calibrated confidence map.
# eval_prob reports best-of-K + uncertainty-error correlation (the one real signal).
# Queued after run_extra5. (Oracle decomp #3(b) ran separately, read-only.)
cd "$(dirname "$0")"
ML=logs/_extra6.log
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ML"; }

say "extra6 queued — waiting for run_extra5 ..."
while pgrep -f run_extra5.sh >/dev/null 2>&1; do sleep 60; done
say "clear — launching prob-head on 5ch (K=5, 3 seeds)"

pj () { g=$1; n=$2; s=$3
  CUDA_VISIBLE_DEVICES=$g bash -c \
   "python train_prob.py --run-name $n --seed $s --epochs 25 --batch-size 32 \
      --num-workers 6 --lr 2e-3 --prob-k 5 --in-ch 5 \
    && python eval_prob.py --run-name $n" > logs/$n.log 2>&1 &
  echo "launched $n gpu$g pid $!"; }

pj 0 P_5ch_k5_s0 0
pj 1 P_5ch_k5_s1 1
pj 2 P_5ch_k5_s2 2
wait
say "extra6 jobs done — aggregate + push"
python agg_full.py > logs/_agg_extra6.log 2>&1
python update_readme.py >> "$ML" 2>&1
TOKEN=$(grep github.com ~/.git-credentials | head -1 | sed -E 's#^https?://##; s#@github.com.*##; s#^[^:]*:##')
git add -A
git -c user.name="Kang-ChangWoo" -c user.email="branden.c.w.kang@gmail.com" \
    commit -q -m "Add #3(a) uncertainty: prob-head on 5ch (calibrated confidence) + oracle decomp

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" \
  && git -c credential.helper="!f(){ echo username=Kang-ChangWoo; echo password=$TOKEN; };f" \
     push origin main >> "$ML" 2>&1 && say "extra6 pushed" || say "extra6 nothing to push"
say "extra6 complete"
