#!/bin/bash
cd "$(dirname "$0")"
while [ "$(ls out/Bnode2_cross_frontwt_s*/metrics_test.json out/Bnode2_cross_hitok_s*/metrics_test.json 2>/dev/null|wc -l)" -lt 6 ]; do sleep 90; done
echo "FRONTWT+HITOK (6) EVALUATED $(date)"
