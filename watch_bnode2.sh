#!/bin/bash
cd "$(dirname "$0")"
while [ "$(ls out/Bnode2_*/metrics_test.json 2>/dev/null | wc -l)" -lt 9 ]; do sleep 120; done
echo "BNODE2 ALL 9 EVALUATED $(date)"
