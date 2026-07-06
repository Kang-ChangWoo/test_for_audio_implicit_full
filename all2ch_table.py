import json, os, statistics as st
def load(r):
    p=f'out/{r}/metrics_ext.json'; return json.load(open(p)) if os.path.exists(p) else None
def agg(base):
    ds=[load(f'{base}_s{s}') for s in range(3)]; ds=[d for d in ds if d]
    if not ds: return None
    ks=['MAE_plain','MAE','AbsRel','RMSE','delta1','delta2','delta3']
    m={k:sum(d[k] for d in ds)/len(ds) for k in ks}; m['n']=len(ds)
    m['sd']=st.pstdev([d['MAE_plain'] for d in ds]) if len(ds)>1 else 0; return m
rows=[('pretrained UNet (ResNet50)','B2_presnet'),('pretrained ViT (ViT-B/16)','B2_pvit'),
      ('BatVision','B2_batvis'),('EchoDiffusion (retrain)','B_echodiff'),
      ('EchoDiffusion (pretrained param)',None),('RayDPT champion (ours)','F2_raydpt')]
out=["# all-2ch fair comparison (test 256x512, masked). baselines=2ch spec; EchoDiffusion=2ch spec+wave; RayDPT=2ch.",
     "","| method | input | MAE_plain | MAE | AbsRel | RMSE | d1 | d2 | d3 | n |","|---|---|---|---|---|---|---|---|---|---|"]
inp={'B2_presnet':'2ch','B2_pvit':'2ch','B2_batvis':'2ch','B_echodiff':'2ch+wave','F2_raydpt':'2ch'}
for name,b in rows:
    if b is None: out.append(f"| {name} | — | N/A | N/A | N/A | N/A | N/A | N/A | N/A | — |"); continue
    m=agg(b)
    if not m: out.append(f"| {name} | {inp.get(b,'?')} | (미완) |"); continue
    out.append(f"| {name} | {inp[b]} | {m['MAE_plain']:.4f}±{m['sd']:.3f} | {m['MAE']:.4f} | {m['AbsRel']:.4f} | {m['RMSE']:.4f} | {m['delta1']:.3f} | {m['delta2']:.3f} | {m['delta3']:.3f} | {m['n']} |")
t="\n".join(out); print(t); open("RESULTS_all2ch.md","w").write(t+"\n")
