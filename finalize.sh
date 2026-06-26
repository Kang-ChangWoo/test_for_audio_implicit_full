#!/bin/bash
# Wait until the training scheduler AND eval daemon (and all their jobs) finish,
# then aggregate everything into RESULTS_full.md and drop a sentinel.
cd "$(dirname "$0")"
busy () {
  pgrep -f "scheduler.py"  >/dev/null && return 0
  pgrep -f "eval_daemon.py" >/dev/null && return 0
  pgrep -f "train_fullmap.py|train.py|train_prob.py" >/dev/null && return 0
  pgrep -f "eval_fullmap.py|eval_prob.py" >/dev/null && return 0
  pgrep -f " eval.py "     >/dev/null && return 0
  return 1
}
while busy; do sleep 60; done
sleep 15
python agg_full.py > logs/_agg_final.log 2>&1
{ echo "ALL DONE $(date)"; echo "trained: $(ls out/*/train_done.json 2>/dev/null|wc -l)/93";
  echo "evaluated: $(ls out/*/metrics_test.json out/*/prob_eval.json 2>/dev/null|wc -l)/93"; } > logs/_ALL_DONE
echo "[finalize] done -> RESULTS_full.md + logs/_ALL_DONE"
