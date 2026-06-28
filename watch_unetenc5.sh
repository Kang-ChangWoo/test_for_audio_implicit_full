#!/bin/bash
cd "$(dirname "$0")"
while [ "$(ls out/Bnode2_cross_unetenc5_s*/metrics_test.json 2>/dev/null | wc -l)" -lt 3 ]; do sleep 120; done
echo "CROSS_UNETENC5 (3 seeds) EVALUATED $(date)"
