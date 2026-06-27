#!/bin/bash
cd "$(dirname "$0")"
while [ "$(ls out/Bnode2_unet8nolog_s*/metrics_test.json 2>/dev/null | wc -l)" -lt 3 ]; do sleep 60; done
echo "BNODE2_UNET8NOLOG (3 seeds) EVALUATED $(date)"
