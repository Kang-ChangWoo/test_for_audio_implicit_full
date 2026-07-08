cd /root/storage/implementation/shared_audio/test_for_audio_implicit_full
for i in $(seq 1 60); do
  d=0
  for k in ic2 ic5 wave; do grep -q "\[$k\] ALL DONE" logs/_build_$k.log 2>/dev/null && d=$((d+1)); done
  [ "$d" -ge 3 ] && break
  sleep 20
done
echo "[watcher] builds done ($d/3), starting mega" >> logs/_mega.log
# 혹시 남은 mega 정리
for p in $(pgrep -f 'mega_pool.py'); do [ "$(cat /proc/$p/comm 2>/dev/null)" = python ] && kill -9 $p 2>/dev/null; done
sleep 3
PYTHONUNBUFFERED=1 nohup python mega_pool.py >> logs/_mega.log 2>&1 &
echo "[watcher] mega launched pid $!" >> logs/_mega.log
