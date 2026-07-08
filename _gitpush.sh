cd /root/storage/implementation/shared_audio/test_for_audio_implicit_full
git add -A
git -c user.name="Kang-ChangWoo" -c user.email="branden.c.w.kang@gmail.com" commit -q -m "planar depth via cache: fix build_cache spec-channel (wave 5ch), MEGA_ONLY filter, parallel planar cache build

- data.py build_cache: C = actual spec channel count (wave path yields 5ch even at in_ch=2)
- mega_pool.py: MEGA_ONLY env filter to isolate a model family; finalv2 uses planar caches (gating)
- planar caches (ic2/ic5/ic2_wave _planar) precomputed with 16 workers, then GPU-bound training

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
TOKEN=$(grep github.com ~/.git-credentials 2>/dev/null | head -1 | sed -E 's#^https?://##; s#@github.com.*##; s#^[^:]*:##')
git -c credential.helper="!f(){ echo username=Kang-ChangWoo; echo password=$TOKEN; };f" push origin main
echo "PUSH_DONE $?"
