cd /root/storage/implementation/shared_audio/test_for_audio_implicit_full
git add -A
git -c user.name="Kang-ChangWoo" -c user.email="branden.c.w.kang@gmail.com" commit -q -m "finalv4 input-information axis (Tier 2-4, multi-view excluded): the only levers that move honest metrics

- data.py: _specN nfft/win param; _spec_mres5 (S19-EXACT: 5ch@nfft512 fine-freq + 5ch@nfft128 fine-time
  = 10ch, short-window echo-delay=distance signal, repo global champion RMSE -7.3%); mres5 dispatch+cache tag
- mega_pool finalv4 (15): mres5 (S19-exact), win15 (longer window -> late reverb/scene-scale),
  gcc (6ch ToF/ITD), raz (13ch direction+range), prob (calibrated uncertainty) x3 seeds; all planar champion
- rationale: model/loss exhausted (only slides AbsRel-the-gamable-metric); RMSE/d1 need MORE SIGNAL

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
TOKEN=$(grep github.com ~/.git-credentials 2>/dev/null | head -1 | sed -E 's#^https?://##; s#@github.com.*##; s#^[^:]*:##')
git -c credential.helper="!f(){ echo username=Kang-ChangWoo; echo password=$TOKEN; };f" push origin main
echo "PUSH_DONE $?"
