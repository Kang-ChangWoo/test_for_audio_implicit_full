#!/bin/bash
cd "$(dirname "$0")"
while [ "$(ls out/Bnode2_cross5ch_s*/metrics_test.json out/Bnode2_hybrid5ch_s*/metrics_test.json 2>/dev/null|wc -l)" -lt 6 ]; do sleep 60; done
echo "5CH (cross5ch+hybrid5ch) 6 EVALUATED $(date)"
