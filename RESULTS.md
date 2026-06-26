# test_for_audio_implicit_n1 — Ray-conditioned U-Net 비교 결과

**과제 framing:** "순수 RayMLP가 U-Net을 이긴다"가 아니라, **ray-conditioned sparse
modulation을 붙인 강한 U-Net이 plain U-Net baseline을 이기는가** 를 본다.
모든 데이터는 스토리지 서버(NFS `/root/storage`, cache=`test_for_audio_better/cache`)에서
직접 읽음 — 이 노드(n1)에는 로컬 데이터 없음.

평가: test split 전수, best(val) 체크포인트. 지표는 cos-lat 가중. seed std는 표준편차.

## ⭐ FULL-VAL 통합표 (가장 엄밀, 3-seed 확정)

`quick_val`을 1500-subset → **full val(3543)**로 바꿔 early-stop을 더 대표성 있게 한 재학습.
plain vs ray 를 ngf64/ngf96 양쪽에서 같은 조건으로 비교.

| 용량 | arm | 설정 | test MAE ± std ↓ | MAE_low | plain 대비 Δ |
|---|---|---|---|---|---|
| ngf64 | plain (A18) | — | 0.9226 ± 0.0062 | 0.7927 | — |
| ngf64 | ray 강 (A19) | s0.4, e2+e3 | **0.9215** ± 0.0056 | 0.7920 | −0.0010 |
| ngf96 | plain (A15) | — | 0.9240 ± 0.0040 | 0.7946 | — |
| ngf96 | ray 약 (A16) | s0.1, e3 | **0.9211** ± 0.0041 | 0.7916 | −0.0030 |

**판정:** Δ(ray−plain)이 양쪽 −0.001~−0.003 으로 seed std의 절반 이내 = **통계적 동률**.
subset early-stop에서 보였던 ray 강의 +0.5% 우위는 full-val로 **소멸**(착시). ray modulation
(약·강 무관)은 plain 대비 실질 이득 없음. (용어: ray 강 = scale 0.4 + e2+e3 두 stage,
ray 약 = scale 0.1 + e3 한 stage.)

## 종합 비교표 (구 subset early-stop, test split)

| 모델 | ngf | params | test MAE [m] ↓ | MAE_low | SHcoefL1 | δ<1.25 ↑ | shuffle MAE | seeds |
|---|---|---|---|---|---|---|---|---|
| Aunet (원본 baseline) | 64 | 29M | 0.9250 ± 0.0058 | 0.7959 | 0.3568 | 0.430 | 1.128 | 3 |
| Aunet (n1 재현) | 64 | 29M | 0.9296 ± 0.0051 | 0.7998 | 0.3580 | 0.425 | 1.124 | 3 |
| A15_bigunet (plain) | 96 | 66M | **0.9203** ± 0.0063 | 0.7919 | 0.3569 | 0.437 | 1.132 | 3 |
| A16_raymod (약한 ray: s0.1, e3) | 96 | 68M | 0.9244 ± 0.0055 | 0.7951 | 0.3567 | 0.431 | 1.128 | 3 |
| A18_unet64reg (plain, wd5e-4) | 64 | 29M | 0.9268 ± 0.0027 | 0.7961 | 0.3569 | 0.428 | 1.124 | 3 |
| A18_raymod64reg (ray s0.1, e3) | 64 | 31M | 0.9234 ± 0.0051 | 0.7940 | 0.3560 | 0.434 | 1.129 | 2 |
| **A19_raymodStrong (s0.4, e2+e3)** | 64 | 31M | **0.9219** ± 0.0025 | 0.7926 | 0.3549 | 0.431 | 1.110 | 3 |

*(shuffle MAE = 오디오를 배치 내에서 섞은 음성 컨트롤; 높을수록 모델이 오디오를 실제로
사용한다는 뜻. 모든 모델이 0.92 → 1.11~1.13으로 폭증 = 오디오를 강하게 사용 중.)*

## 핵심 비교 (matched control)

| 비교 (같은 레시피) | ray | plain | Δ (향상) | 유의성 |
|---|---|---|---|---|
| **A19 vs A18** (ngf64, s0.4 e2+e3) | 0.9219 ± .0025 | 0.9268 ± .0027 | **+0.0049 (~0.5%)** | ~1.8σ, 3/3 seed 일관 ✅ |
| A18_raymod vs A18 (ngf64, s0.1 e3) | 0.9234 | 0.9268 | +0.0034 | noise 이내 |
| A16 vs A15 (ngf96, s0.1 e3) | 0.9244 | 0.9203 | **−0.0041 (오히려 나쁨)** | noise 이내 |

## 판정

1. **n1 재현 검증 ✅** — Aunet n1 재현(0.9296)이 원본(0.9250)과 noise 이내 일치 → 스토리지
   서버 셋업이 baseline까지 정확히 재현.
2. **용량은 무효** — ngf 64→96 (params 29M→66M, 2.3×)에도 plain MAE 0.927→0.920으로
   seed std 이내. 4×에 가까운 용량이 사실상 0 개선.
3. **ray modulation 효과는 작고 조건부** — 강한 ray(A19)만이 자기 매칭 baseline(A18)을
   3 seed 모두에서 일관되게 이김(+0.5%, ~1.8σ). 약한 ray(A16)는 큰 U-Net에서 오히려 나빴음.
4. **그러나 ray가 "용량"을 못 이김** — A19(0.9219, ngf64+ray) 가 A15(0.9203, ngf96 plain)
   보다 못함. 즉 "ray 붙이기" < "그냥 용량 키우기".
5. **과적합이 천장을 만듦** — 모든 모델 val MAE가 ep~10에서 바닥 치고 상승(아래). ngf64+wd5e-4
   로도 못 막음 → best(early-stop) 체크포인트가 사실상 천장.
6. **diag (A19_s1):** ray_map/γ/β sample-std=0.26/0.23/0.30, audio-sensitivity=0.14.
   A16(1.33 / 0.30)보다 **약하게** 걸림 → scale 0.4 + wd 5e-4가 modulation을 damping.
   A19의 작은 이득은 directional 신호라기보다 **약한 정규화**에 가까움.

### 과적합 곡선 예시 (val MAE/epoch, A15_bigunet_s0)
```
ep00..11: 1.122 1.070 1.117 1.050 1.024 1.026 1.016 1.013 1.033 1.021 1.013 1.007(best)
ep12..24: 1.014 1.027 1.049 1.042 1.047 1.049 1.064 1.061 1.078 1.081 1.082 1.085 1.088
```

## Probabilistic head (P_k1/k5/k10 — layout 모호성 가설 검증)

`train_prob.py`: K개 coarse-layout 가설(relaxed-WTA) + per-pixel Laplace scale(aleatoric
불확실성). "audio→depth는 다중모드라 K개로 커버하면 천장을 넘는다"는 가설을 검증. full-val,
seed0. eval_prob의 **mismatched-scene 컨트롤**이 핵심: best-of-K를 *엉뚱한* scene의 GT로 골랐을
때와 비교 → 진짜 scene별 모호성인지 단순 free-pick 운인지 분리.

| K | det baseline | mean-of-K | best-of-K (oracle) | best-of-K **컨트롤** | **REAL 다중모드 이득** | diversity | 불확실성 corr |
|---|---|---|---|---|---|---|---|
| 1 | 0.780 | 0.825 | 0.825 | 0.855 | +0.030 | — | +0.67 |
| 5 | 0.780 | 0.848 | **0.674** | 0.677 | +0.003 | 0.61 | +0.66 |
| 10 | 0.780 | 0.854 | **0.636** | 0.636 | −0.001 | 0.63 | +0.65 |

*(MAE_plain [m]. REAL 다중모드 이득 = 컨트롤 − 실제 best-of-K; >0이어야 진짜.)*

**판정:**
- best-of-K가 K↑에 따라 0.78→0.64로 크게 좋아 *보이지만*, **컨트롤도 동일(0.636)** →
  REAL 다중모드 이득 ≈ **0**(K5 +0.003, K10 −0.001). best-of-K 향상은 전부 **free-pick 착시**,
  per-scene 모호성을 해소한 게 아님.
- mean-of-K(0.85)는 결정론(0.78)보다 **나쁨**(가설 평균 흐려짐); 개별 head도 ~0.94~0.99로 어중간.
- **건진 것:** 불확실성 calibration corr=+0.65~0.67 → Laplace scale이 오차 위치를 실제로 맞춤(쓸모 有).
- ⇒ probabilistic head도 천장(0.78 MAE_plain)을 **못 뚫음**. 다중모드 framing은 usable(non-oracle)
  하게는 성립 안 함. 단 calibrated uncertainty는 진짜.

## 결론

> **성공 기준(A16이 A15·Aunet을 seed std 넘어 제침) 기준으로는 미달.**
> 강한 ray(A19)에서 매칭 baseline 대비 ~0.5%의 작고 일관된 향상은 나오나, plain 큰 U-Net을
> 못 이기고 효과 크기가 전체 0.92대 분포 안에 있음. 용량·ray 둘 다 MAE 0.92 천장을 못 뚫음
> → 병목은 **모델 표현력이 아니라 오디오→깊이의 정보량**(repo의 oracle-decomposition 결론과 일치).

probabilistic head까지 검증 완료(위 섹션): 다중모드 가설도 per-scene으로는 천장을 못 뚫음
(best-of-K는 free-pick 착시). 네 방향(implicit ray · ray-mod U-Net · capacity · probabilistic)
모두 **MAE ≈ 0.92 정보 천장**으로 수렴. 유일하게 살아있는 실용 신호는 prob head의
**calibrated uncertainty(corr +0.65)**.

**다음 후보:** point-accuracy를 더 짜내긴 어려움 → (a) calibrated uncertainty를 산출물로 활용,
또는 (b) 입력 정보 자체를 늘리는 방향(멀티-소스 RIR / 더 긴 윈도우 / 추가 모달리티).

## 재현 방법 (n1)
```bash
bash run_unet.sh            # Aunet baseline (ngf64)
bash run_unet_raymod.sh     # A15 big-unet vs A16 weak-ray (ngf96)
bash run_recipe.sh          # A18 regularized ngf64 (plain vs weak-ray)
bash run_raymod_strong.sh   # A19 strong-ray (s0.4, e2+e3) vs A18 baseline
# 평가
python eval_fullmap.py --run-name <RUN> --controls True
python diag_raymod.py  --run-name <A16/A18/A19 run>   # γ/β collapse·audio-sensitivity
```
