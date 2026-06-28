#!/bin/bash
cd "$(dirname "$0")"
while [ "$(ls out/*/metrics_test.json out/*/prob_eval.json 2>/dev/null | wc -l)" -lt 129 ]; do sleep 120; done
echo "MEGA WAVE: reached 129 evaluated $(date)"
