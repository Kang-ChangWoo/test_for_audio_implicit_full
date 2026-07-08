cd /root/storage/implementation/shared_audio/test_for_audio_implicit_full
for i in $(seq 1 240); do
  n=$(ls out/finalv2_batvision_2ch_s*/metrics_test.json out/finalv2_preunet_2ch_s*/metrics_test.json out/finalv2_previt_2ch_s*/metrics_test.json 2>/dev/null | wc -l)
  [ "$n" -ge 9 ] && break
  sleep 60
done
echo "[batch1] $n/9 완주" >> logs/_report_batch1.log
python _report_batch1.py >> logs/_report_batch1.log 2>&1
