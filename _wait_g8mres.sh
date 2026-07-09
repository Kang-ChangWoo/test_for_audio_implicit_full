cd /root/storage/implementation/shared_audio/test_for_audio_implicit_full
for i in $(seq 1 600); do
  g=$(ls out/finalv3_raydpt_2ch_g8clean_s*/metrics_test.json out/finalv3_raydpt_2ch_g8global_s*/metrics_test.json 2>/dev/null|wc -l)
  mr=$(ls out/finalv3_raydpt_mres_champ_s*/metrics_test.json out/finalv3_raydpt_mres_champ_e51_s*/metrics_test.json 2>/dev/null|wc -l)
  [ "$g" -ge 6 ] && [ "$mr" -ge 6 ] && break
  sleep 60
done
echo "=== [g8+mres 완주] $(date +%m-%d\ %H:%M) g8=$g/6 mres=$mr/6 ===" >> logs/_report_full8.log
python _report_full8.py >> logs/_report_full8.log 2>&1
