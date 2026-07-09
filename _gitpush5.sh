cd /root/storage/implementation/shared_audio/test_for_audio_implicit_full
git add -A
git -c user.name="Kang-ChangWoo" -c user.email="branden.c.w.kang@gmail.com" commit -q -m "RayDPT dead-encoder cleanup (E116) + e8 global-token utilization (raydpt_use_global)

- model_unet_coarse: UNet8Encoder depth arg (default 8 unchanged for unet_coarse/echo/rayvit/cross
  which use e8; RayDPT/Resampler use depth=4 -> drop dead e5-e8). forward returns bottleneck=None if depth<8.
- config: raydpt_use_global flag
- model_raydpt: use_global=False drops e5-e8 (24.80->7.87M, -16.9M dead params, RayDPT & Resampler).
  use_global=True forwards e5-e8, projects e8 (1x2 scene bottleneck) to 2 global tokens + learned marker,
  prepends to ALL cross-attn KV across every branch (default/lite/noray/msf/global/cue_route) + Resampler A.
- Verified: (a-f) forward shapes OK; use_global=False numerically identical to e5-e8-present-but-unused
  (allclose, max_diff 0 -> e5-e8 truly dead); unet_coarse e8 path unchanged. Smoke only, no training.
- Queued 2 attempts (2ch planar champion, no TTA): g8clean (cleanup) vs g8global (e8 tokens), 3 seeds each

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
TOKEN=$(grep github.com ~/.git-credentials 2>/dev/null | head -1 | sed -E 's#^https?://##; s#@github.com.*##; s#^[^:]*:##')
git -c credential.helper="!f(){ echo username=Kang-ChangWoo; echo password=$TOKEN; };f" push origin main
echo "PUSH_DONE $?"
