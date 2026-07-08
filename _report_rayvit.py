import json, glob, os
print("=== RayViT (ViT 인코더 + ray cross-attn) 결과 ===")
rows=[]
for f in sorted(glob.glob("out/RV_*/metrics_test.json")):
    r=os.path.basename(os.path.dirname(f)); d=json.load(open(f))["test"]
    rows.append((r,d))
for r,d in sorted(rows,key=lambda x:x[1]["MAE"]):
    print(f"  {r:22} MAE={d['MAE']:.4f} RMSE={d['RMSE']:.4f} AbsRel={d['AbsRel']:.4f} d1={d['delta1']:.3f}")
print(f"  완주 {len(rows)}/8")
print("[기준] RayDPT(U-Net8 enc) champion: MAE 0.884 RMSE 1.416 AbsRel 0.509 | U-Net8 0.893 | pretrained ViT(full) 0.903")
