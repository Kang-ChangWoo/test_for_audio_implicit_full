#!/bin/bash
# Hands-off chain: wait until the A15/A16 raymod job has fully finished (frees
# GPUs 0-2), then reproduce the original pix2pix U-Net baseline (Aunet, ngf=64,
# the same recipe as the sibling test_for_audio_implicit/out/Aunet_s*).
cd "$(dirname "$0")"
echo "[chain] waiting for ALL_RAYMOD_DONE ..."
while ! grep -q ALL_RAYMOD_DONE logs/_raymod_launch.log 2>/dev/null; do
  sleep 15
done
echo "[chain] raymod done -> launching Aunet baseline"
bash run_unet.sh
echo "ALL_AUNET_DONE"
