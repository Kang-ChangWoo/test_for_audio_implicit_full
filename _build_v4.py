import sys
from types import SimpleNamespace
from config import DEFAULTS
from data import build_cache, cache_exists
kind=sys.argv[1]
c=SimpleNamespace(**DEFAULTS); c.depth_type="planar"; c.num_workers=12
if kind=="mres5": c.in_ch=10; c.audio_src="mres5"
elif kind=="w15": c.in_ch=5; c.audio_src="binaural"; c.audio_window_m=15.0
elif kind=="gcc": c.in_ch=6; c.audio_src="gcc"
elif kind=="raz": c.in_ch=13; c.audio_src="raz"
for split in ["train","val","test"]:
    if cache_exists(c,split): print(f"{kind}/{split} exists",flush=True); continue
    print(f"{kind}/{split} building...",flush=True); build_cache(c,split)
print(f"{kind} DONE",flush=True)
