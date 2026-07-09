cd /root/storage/implementation/shared_audio/test_for_audio_implicit_full
declare -A done
for i in $(seq 1 720); do
  for cfg in raydpt_2ch_ray raydpt_5ch_ray raydpt_2ch_noray raydpt_5ch_noray; do
    [ -n "${done[$cfg]}" ] && continue
    n=$(ls out/finalv2_${cfg}_s*/metrics_test.json 2>/dev/null | wc -l)
    if [ "$n" -ge 3 ]; then
      { echo "=== [$cfg] 3/3 완주 $(date +%m-%d\ %H:%M) ==="; python _report_raydpt_cfg.py $cfg; echo; } >> logs/_report_raydpt.log 2>&1
      done[$cfg]=1
    fi
  done
  [ ${#done[@]} -ge 4 ] && break
  sleep 60
done
echo "[watcher] RayDPT 4 config 전부 리포트 완료" >> logs/_report_raydpt.log
