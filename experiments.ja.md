> **Frozen snapshot (2026-10-12 14:00 KST). No longer maintained — the canonical, up-to-date log is [experiments.md](experiments.md) (English).**

# 実験ログ

言語: [English](experiments.md) | [한국어](experiments.ko.md) | 日本語

| 日付 | stage | 方法 | CV RMSE ± std | LB | 結論 |
|---|---|---|---|---|---|
| 2026-10-12 | 0 | ラベル/画像再構成(`src/stage0_recon.py`) | - | - | 連続ラベル(500/500 unique、小数 14 桁、grid なし)。log(y) ~ normal(Shapiro p 0.14 vs raw 4e-5)。完全/近傍重複なし(min 32px NN-MSE 0.58)、PNG メタデータなし、ファイルサイズ r=-0.03。潜在変数推定 + 公式復元として扱う。 |
| 2026-10-12 | 0v2 | 監査済み再構成(`src/stage0_recon.py v2`) | - | - | ラベル = 丸めなし float64(500/500 shortest-repr)。grid なし(すべての g で Rayleigh R<0.1)。Box-Cox λ=-0.69 [-1.46,0.08] = ターゲット変換候補のみ; "log-normal ⇒ multiplicative" は撤回。1500 ファイルすべて同じ PNG encoder signature。KS: shifted stats 0/28。D4 不変構造重複検索(positive-control power 100%): 0 pairs。どの画像 signature の kNN も constant より悪い ⇒ global similarity にはラベル情報がない。 |
| 2026-10-12 | 0v2 | **BASELINE FLOOR** const mean | 17.665 ± 1.329 | - | RepeatedKFold(5x3, seed 42)。単純な基準線。 |
| 2026-10-12 | 0v2 | **BASELINE FLOOR** 28 cheap global stats の Ridge | 15.156 ± 1.227 | - | Stage 1 はこれを上回る必要がある(R² ≈ 0.26)。 |
| 2026-10-12 | 0v2 | **BASELINE FLOOR** z-norm 32px box thumbnails の kNN-5 | 19.606 ± 1.464 | - | constant より悪い(struct64 18.65、radial 18.92 も悪い)。 |
| 2026-10-12 | 0-close | F1 control family 分離; F2 left-tail robust null(median−5·MAD); F3 Hann window; F4 regex=False | - | - | struct64 は 54 test-train pairs を flag したが、flag された train-train pairs の \|dy\| は 33.1 > random 19.7 で、目視確認でも構造が異なる(low-contrast shading) ⇒ false positive、**leak なし**。raw32/radial: flag 0。Hann: 7/8 bands で \|rho\| 増加(band0 .069→.131)。 |
| 2026-10-12 | P1 | F3 後の anchors: const / ridge_cheap / lgbm_cheap | 17.665±1.329 / 14.853±1.267 / 14.993±1.310 | - | R² −0.01 / 0.281 / 0.269。すべての kNN < const。新しい floor = 14.853。 |
| 2026-10-12 | P2 | ターゲット変換 raw/log/boxcox(ridge, lgbm) | ridge 14.853/14.868/14.899; lgbm 14.993/14.995/14.997 | - | 差は fold std より十分小さい; raw が勝ち。Box-Cox λ は決定的ではなかった。 |
| 2026-10-12 | P3 | **submissions/001_cheap_gbdt.csv** = 28 cheap stats に Ridge(raw y)、500 件で refit、clipped | 14.853±1.267 | _pending_ | 想定 public LB 14.9(300 rows の 95% band: 13.5–16.3)。ファイル名はリクエストどおり; P2 winner は GBDT ではなく Ridge。 |
| 2026-10-12 | 1 | `features.py`: nuisance-aware features 114 個(q_/r_/n_)、1500 imgs で 45.9s | - | - | 114 個中 84 個の feature が y より blur/noise と強く相関。すべての古典的 grain-size 推定量は noise-dominated(n_mli ρy −0.03 vs ρnoise 0.76)。Top: n_otsu_c0 ρ 0.44。 |
| 2026-10-12 | 1 | stage1 feats の ridge / lgbm; + cheap | 14.802±1.077 / 14.804±1.176; 14.608±1.089 / 14.574±1.129 | - | R² ≤ 0.31。001 比 paired Δ −0.28, p 0.10(有意ではない)。 |
| 2026-10-12 | 2 | T1 log-linear power law(5 latents) | 15.515±1.234 | - | R²(log y) CV 0.22; d exponent +0.02(−0.5 ではない)。**Formula NOT recovered.** Additive Hall-Petch form 15.35。 |
| 2026-10-12 | 2 | T2 3749 enumerated H0 + C·Π formulas | best 15.531 | - | Ridge-cheap より悪い。T3: formula residual ρ with contrast span 0.23, blur −0.17 ⇒ measurement bottleneck; GBDT residual vs nuisance \|ρ\| ≤ 0.06。 |
| 2026-10-12 | 2 | `submissions/002_s1_lgbm.csv`(lgbm stage1+cheap) | 14.574±1.129 | _not uploaded_ | 想定 public LB 14.6(13.3–16.0)。 |
| 2026-10-12 | 2 | `submissions/003_blend_ridge_cheap_lgbm_s1.csv`(0.5·001 + 0.5·002, fixed weights) | 14.355±1.182 | _not uploaded_ | R² 0.329。001 比 paired Δ −0.498, p 7e-5, 14/15 folds。現時点の最高 CV。 |
| 2026-10-12 | A | ceiling: feature-space twin test(s1+cheap, z / PCA-white95) | - | - | closest 1% pairs \|dy\| 15.2 / 14.7 vs GMD 19.7; clean-structural(23) 15.5/15.2; blur-dominated(84) 15.8/15.9; nuisance(7) 17.7/18.2 ⇒ features insufficient。 |
| 2026-10-12 | A | blend 003 の residual | sd 14.35 | - | skew 0.33, exkurt 0.73(Shapiro p .002); pred tercile ごとの sd 11.6/15.1/15.9; max \|ρ(res, feat)\| 0.11(0 Bonferroni-sig); residual learnable R² −0.02 ⇒ measurement/information-limited。 |
| 2026-10-12 | B | B1 MTF fit(σ̂ validated: +known blur を ~0.15 px 以内で復元), B2 RL-deconv + re-segment(d_), B3 scale-space extrapolation(x_) | - | - | 60 個の新 feature 中 1 個が blur domination を破る(d_otsu_c0 ρy .365 vs ρblur .317); grain-size 推定量ではない。 |
| 2026-10-12 | B | s1+cheap+new の lgbm / ridge | 14.514±1.227 / 14.616±1.024 | - | 14.574 比: Δ −0.060(p .37, 13/15 wins) / +0.042(p .72)。ノイズ範囲内 ⇒ primary には採用せず、blend candidate として保持。 |
| 2026-10-12 | C1 | top-20 predictive S1 features の quadrant(128px) ICC | - | - | median ICC 0.603(<0.8): n_otsu_c0 .42, n_ws_eqd_mean .41; nuisance-like globals のみ >0.8 ⇒ patch label sharing は妥当でない; CNN は full 256 images で学習。 |
| 2026-10-12 | D | non-CNN families の blend(eq vs cross-fitted NNLS) | eq(ridge_cheap+lgbm_s1+lgbm_s1new) 14.297±1.202 | - | 003 比 Δ −0.057, p .13 ⇒ ノイズ範囲内、003 を維持。 |
| 2026-10-12 | E | refactor: `features.parquet` の列順が変更(Part B) ⇒ lgbm_s1 OOF 14.574→14.568(ridge bit-identical); 002/003 を scripts で再生成 | 003: 14.354±1.199 | - | 初版比で 003 ファイル変化 mean 0.36 / max 1.67(corr .9989)。 |
