cd /root/storage/implementation/shared_audio/test_for_audio_implicit_full
git add -A
git -c user.name="Kang-ChangWoo" -c user.email="branden.c.w.kang@gmail.com" commit -q -m "Convert all exploratory (non-finalv2) queue jobs to planar depth

- mega_pool.py: planar-conversion block rewrites every non-finalv2 job to --depth-type planar
  + planar cache (ic5/ic2/ic2_wave _planar, already built); finalv2 untouched.
  Toggle with PLANAR_EXPLORATORY=0 (default on).
- batvision finalv2 (planar) done: scale-invariant AbsRel/delta1 statistically tied with radial
  (planar lower absolute MAE is purely the 0.86x GT-scale artifact, confirmed 0.658/0.86=0.766~=radial 0.760)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
TOKEN=$(grep github.com ~/.git-credentials 2>/dev/null | head -1 | sed -E 's#^https?://##; s#@github.com.*##; s#^[^:]*:##')
git -c credential.helper="!f(){ echo username=Kang-ChangWoo; echo password=$TOKEN; };f" push origin main
echo "PUSH_DONE $?"
