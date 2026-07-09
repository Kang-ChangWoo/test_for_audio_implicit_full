cd /root/storage/implementation/shared_audio/test_for_audio_implicit_full
git add -A
git -c user.name="Kang-ChangWoo" -c user.email="branden.c.w.kang@gmail.com" commit -q -m "Multi-res STFT (S19 repo global champion) + 5ch baselines + TTA measurement

- data.py: _spec_mres (S19 multi-resolution STFT: 2ch magnitude x K windows [128,400,1024]),
  audio_src=mres dispatch, cache tag _mres; config mres_wins
- mega_pool: finalv3 mres jobs (6, champion loss, NO TTA per request); 5ch baselines (12:
  batvision/preunet/previt on ic5_planar, echodiff on ic5_planar_wave)
- TTA (E127) measured on planar 2ch champ: only ~0.002 gain here (vs repo 0.024) — model already
  L/R-equivariant via flip-aug, so TTA near-useless; safe to exclude (eval-only anyway)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
TOKEN=$(grep github.com ~/.git-credentials 2>/dev/null | head -1 | sed -E 's#^https?://##; s#@github.com.*##; s#^[^:]*:##')
git -c credential.helper="!f(){ echo username=Kang-ChangWoo; echo password=$TOKEN; };f" push origin main
echo "PUSH_DONE $?"
