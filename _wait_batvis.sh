cd /root/storage/implementation/shared_audio/test_for_audio_implicit_full
for i in $(seq 1 180); do
  n=$(ls out/finalv2_batvision_2ch_s*/metrics_test.json 2>/dev/null | wc -l)
  [ "$n" -ge 3 ] && break
  sleep 30
done
echo "[batvis] $n/3 완주 $(date +%H:%M)" >> logs/_report_batvis.log
python _report_batvis.py >> logs/_report_batvis.log 2>&1
