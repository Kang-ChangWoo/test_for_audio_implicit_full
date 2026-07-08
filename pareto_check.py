import json, glob, os
RB, AB = 1.424, 0.537   # baseline 최저 문턱 (ViT RMSE / BatVision AbsRel)
rows=[]
for f in glob.glob('out/P_[rbax]*/metrics_test.json'):
    r=os.path.basename(os.path.dirname(f))
    if not r.split('_')[0]=='P': continue
    try: d=json.load(open(f))['test']
    except: continue
    both = d['RMSE']<RB and d['AbsRel']<AB
    rows.append((r,d,both))
rows.sort(key=lambda x:x[1]['RMSE'])
print(f"{'config':10}{'RMSE':>8}{'AbsRel':>8}{'MAE':>8}  both-win?(RMSE<1.424 & AbsRel<0.537)")
print('-'*56)
for r,d,b in rows:
    print(f"{r:10}{d['RMSE']:8.4f}{d['AbsRel']:8.4f}{d['MAE']:8.4f}  {'WIN' if b else ''}")
wins=[r for r,d,b in rows if b]
print(f"\n완주 {len(rows)}/20 | both-win {len(wins)}: {wins}")
