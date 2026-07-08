cd /root/storage/implementation/shared_audio/test_for_audio_implicit_full
for i in $(seq 1 360); do
  n=$(ls out/finalv2_preunet_2ch_s*/metrics_test.json out/finalv2_previt_2ch_s*/metrics_test.json 2>/dev/null | wc -l)
  [ "$n" -ge 6 ] && break
  sleep 60
done
echo "[2ch] preunet+previt $n/6 완주 $(date +%H:%M)" >> logs/_report_2ch.log
python _report_2ch.py >> logs/_report_2ch.log 2>&1
