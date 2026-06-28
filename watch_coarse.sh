#!/bin/bash
cd "$(dirname "$0")"
while [ "$(ls out/C_unet8_*_s0/metrics_test.json 2>/dev/null | wc -l)" -lt 6 ]; do sleep 120; done
echo "COARSE LAYOUT s0 (6) EVALUATED $(date)"
