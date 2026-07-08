### Table 2 — native-input (RayDPT = P_b3 both-win 대표)
| method | input | MAE_plain↓ | MAE↓ | AbsRel↓ | RMSE↓ | δ1↑ | δ2↑ | δ3↑ | n |
|---|---|---|---|---|---|---|---|---|---|
| pretrained UNet (ResNet50) | 2ch | 0.7962 | 0.9482 | 0.6002 | 1.4721 | 0.425 | 0.647 | 0.781 | 3 |
| pretrained ViT (ViT-B/16) | 2ch | 0.7619 | 0.9089 | 0.5638 | 1.4244 | 0.442 | 0.665 | 0.796 | 3 |
| BatVision | 2ch | 0.7597 | 0.9049 | 0.5374 | 1.4433 | 0.446 | 0.667 | 0.797 | 3 |
| EchoDiffusion (retrain) | 2ch+wave | 0.8020 | 0.9616 | 0.6247 | 1.4572 | 0.407 | 0.630 | 0.772 | 3 |
| EchoDiffusion (pretrained param) | — | N/A | — | — | — | — | — | — | — |
| **RayDPT (ours, P_b3)** | 5ch | 0.7431 | 0.8836 | 0.5087 | 1.4161 | 0.456 | 0.676 | 0.804 | 1 |
