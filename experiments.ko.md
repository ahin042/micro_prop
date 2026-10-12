# 실험 로그

언어: [English](experiments.md) | 한국어 | [日本語](experiments.ja.md)

| 날짜 | 단계 | 방법 | CV RMSE ± std | LB | 결론 |
|---|---|---|---|---|---|
| 2026-10-12 | 0 | 라벨/이미지 재구성(`src/stage0_recon.py`) | - | - | 연속형 라벨(500/500 unique, 소수 14자리, grid 없음). log(y) ~ normal(Shapiro p 0.14 vs raw 4e-5). 정확/근접 중복 없음(min 32px NN-MSE 0.58), PNG 메타데이터 없음, 파일 크기 r=-0.03. 잠재변수 추정 + 공식 복원 문제로 간주. |
| 2026-10-12 | 0v2 | 감사된 재구성(`src/stage0_recon.py v2`) | - | - | 라벨 = 반올림되지 않은 float64(500/500 shortest-repr). grid 없음(모든 g에서 Rayleigh R<0.1). Box-Cox λ=-0.69 [-1.46,0.08] = 타깃 변환 후보일 뿐; "log-normal ⇒ multiplicative"는 철회. 1500개 파일 모두 동일한 PNG encoder signature. KS: 이동한 통계 0/28. D4 불변 구조 중복 검색(positive-control power 100%): 0 pairs. 어떤 이미지 signature로도 kNN은 constant보다 나쁨 ⇒ 전역 유사도에는 라벨 정보가 없음. |
| 2026-10-12 | 0v2 | **BASELINE FLOOR** const mean | 17.665 ± 1.329 | - | RepeatedKFold(5x3, seed 42). 단순 기준선. |
| 2026-10-12 | 0v2 | **BASELINE FLOOR** 28개 cheap global stats 기반 Ridge | 15.156 ± 1.227 | - | Stage 1은 이 성능을 넘어야 함(R² ≈ 0.26). |
| 2026-10-12 | 0v2 | **BASELINE FLOOR** z-norm 32px box thumbnail 기반 kNN-5 | 19.606 ± 1.464 | - | constant보다 나쁨(struct64 18.65, radial 18.92도 더 나쁨). |
| 2026-10-12 | 0-close | F1 control family 분리; F2 left-tail robust null(median−5·MAD); F3 Hann window; F4 regex=False | - | - | struct64가 test-train 54 pairs를 flag했지만, flag된 train-train pairs의 \|dy\| 33.1 > random 19.7이고 시각 확인상 구조가 다름(low-contrast shading) ⇒ false positive, **leak 없음**. raw32/radial: flag 0. Hann: 7/8 bands에서 \|rho\| 증가(band0 .069→.131). |
| 2026-10-12 | P1 | F3 이후 anchor: const / ridge_cheap / lgbm_cheap | 17.665±1.329 / 14.853±1.267 / 14.993±1.310 | - | R² −0.01 / 0.281 / 0.269. 모든 kNN < const. 새 floor = 14.853. |
| 2026-10-12 | P2 | 타깃 변환 raw/log/boxcox(ridge, lgbm) | ridge 14.853/14.868/14.899; lgbm 14.993/14.995/14.997 | - | 차이는 fold std보다 훨씬 작음; raw가 우세. Box-Cox λ는 결정적이지 않았음. |
| 2026-10-12 | P3 | **submissions/001_cheap_gbdt.csv** = 28개 cheap stats에 Ridge(raw y), 500개로 refit, clipped | 14.853±1.267 | _pending_ | 예상 public LB 14.9(300 rows 기준 95% band: 13.5–16.3). 파일명은 요청에 따른 것; P2 winner는 GBDT가 아니라 Ridge. |
| 2026-10-12 | 1 | `features.py`: nuisance-aware features 114개(q_/r_/n_), 1500 imgs에 45.9s | - | - | 114개 중 84개 feature가 y보다 blur/noise와 더 강하게 상관됨. 모든 고전 grain-size 추정량은 noise-dominated(n_mli ρy −0.03 vs ρnoise 0.76). 최고: n_otsu_c0 ρ 0.44. |
| 2026-10-12 | 1 | stage1 feats 기반 ridge / lgbm; + cheap | 14.802±1.077 / 14.804±1.176; 14.608±1.089 / 14.574±1.129 | - | R² ≤ 0.31. 001 대비 paired Δ −0.28, p 0.10(유의하지 않음). |
| 2026-10-12 | 2 | T1 log-linear power law(5 latents) | 15.515±1.234 | - | R²(log y) CV 0.22; d exponent +0.02(−0.5가 아님). **Formula NOT recovered.** Additive Hall-Petch form 15.35. |
| 2026-10-12 | 2 | T2 3749개 H0 + C·Π 공식 열거 | best 15.531 | - | Ridge-cheap보다 나쁨. T3: formula residual ρ with contrast span 0.23, blur −0.17 ⇒ measurement bottleneck; GBDT residual vs nuisance \|ρ\| ≤ 0.06. |
| 2026-10-12 | 2 | `submissions/002_s1_lgbm.csv`(lgbm stage1+cheap) | 14.574±1.129 | _not uploaded_ | 예상 public LB 14.6(13.3–16.0). |
| 2026-10-12 | 2 | `submissions/003_blend_ridge_cheap_lgbm_s1.csv`(0.5·001 + 0.5·002, fixed weights) | 14.355±1.182 | _not uploaded_ | R² 0.329. 001 대비 paired Δ −0.498, p 7e-5, 14/15 folds. 현재까지 최고 CV. |
| 2026-10-12 | A | ceiling: feature-space twin test(s1+cheap, z / PCA-white95) | - | - | closest 1% pairs \|dy\| 15.2 / 14.7 vs GMD 19.7; clean-structural(23) 15.5/15.2; blur-dominated(84) 15.8/15.9; nuisance(7) 17.7/18.2 ⇒ features insufficient. |
| 2026-10-12 | A | blend 003의 residual | sd 14.35 | - | skew 0.33, exkurt 0.73(Shapiro p .002); pred tercile별 sd 11.6/15.1/15.9; max \|ρ(res, feat)\| 0.11(0 Bonferroni-sig); residual learnable R² −0.02 ⇒ measurement/information-limited. |
| 2026-10-12 | B | B1 MTF fit(σ̂ validated: +known blur를 ~0.15 px 이내 복원), B2 RL-deconv + re-segment(d_), B3 scale-space extrapolation(x_) | - | - | 60개 새 feature 중 1개가 blur domination을 깸(d_otsu_c0 ρy .365 vs ρblur .317); grain-size 추정량은 해당 없음. |
| 2026-10-12 | B | s1+cheap+new 기반 lgbm / ridge | 14.514±1.227 / 14.616±1.024 | - | 14.574 대비: Δ −0.060(p .37, 13/15 wins) / +0.042(p .72). 노이즈 범위 내 ⇒ primary로 채택하지 않고 blend 후보로 보존. |
| 2026-10-12 | C1 | top-20 predictive S1 features의 quadrant(128px) ICC | - | - | median ICC 0.603(<0.8): n_otsu_c0 .42, n_ws_eqd_mean .41; nuisance-like globals만 >0.8 ⇒ patch label sharing은 타당하지 않음; CNN은 전체 256 이미지를 학습. |
| 2026-10-12 | D | non-CNN family blend(eq vs cross-fitted NNLS) | eq(ridge_cheap+lgbm_s1+lgbm_s1new) 14.297±1.202 | - | 003 대비 Δ −0.057, p .13 ⇒ 노이즈 범위 내, 003 유지. |
| 2026-10-12 | E | refactor: `features.parquet` 컬럼 순서 변경(Part B) ⇒ lgbm_s1 OOF 14.574→14.568(ridge bit-identical); 002/003 scripts로 재생성 | 003: 14.354±1.199 | - | 첫 버전 대비 003 파일 변화 mean 0.36 / max 1.67(corr .9989). |
