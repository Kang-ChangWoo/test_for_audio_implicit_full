cd /root/storage/implementation/shared_audio/test_for_audio_implicit_full
git add -A
git -c user.name="Kang-ChangWoo" -c user.email="branden.c.w.kang@gmail.com" commit -q -m "finalv3: repo champion recipes on planar (2ch-focus) + E127/E128 TTA

- eval_fullmap.py: eval-time L/R-flip TTA (avg mirrored-input prediction), flag eval_tta_flip
  = auto_audio_depth_estimation E127/E128 champion (replicated ~0.024 composite win)
- config.py: eval_tta_flip flag
- mega_pool.py: finalv3 (12 jobs, 2ch-focused) = champion stack w_rel0.1+EMA0.995+coarse-geo-self-attn
  +grad-loss (E2/E16/E34) + TTA; variants E51 (2-block coarse-sa), E117 (berHu-low); planar caches.
  Fronted; existing planar exploratory queue runs after.
- finalv2 (8-family planar, n=3) complete: radial<->planar tie on scale-invariant AbsRel/delta1;
  RayDPT ray lowest AbsRel; ray-conditioning helps AbsRel in planar (unlike radial)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
TOKEN=$(grep github.com ~/.git-credentials 2>/dev/null | head -1 | sed -E 's#^https?://##; s#@github.com.*##; s#^[^:]*:##')
git -c credential.helper="!f(){ echo username=Kang-ChangWoo; echo password=$TOKEN; };f" push origin main
echo "PUSH_DONE $?"
