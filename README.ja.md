# 微細構造 → 硬さ回帰

言語: [English](README.md) | [한국어](README.ko.md) | 日本語

画像回帰の課題です(256×256 グレースケール微細構造 → 合成硬さプロキシ、評価指標 RMSE)。
実験ログ全体は [experiments.ja.md](experiments.ja.md) にあります([English](experiments.md), [한국어](experiments.ko.md))。
機械可読な結果は `reports/*.json` にあります。

## 再現方法(単一の実行手順)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install libomp            # macOS のみ: LightGBM の OpenMP ランタイム
.venv/bin/python src/run.py --mode full     # 全体を実行(CNN 学習を含む、Apple M5 で約 1.5 時間)
.venv/bin/python src/run.py --mode fast     # 同じ処理だが、キャッシュ済み CNN OOF/テスト予測を再利用
```

データは `./data/{train,test}/*.png`, `./data/train.csv`, `./data/sample_submission.csv` に配置してください。
すべてのパスはリポジトリルートからの相対パスです。出力は `submissions/*.csv`, `cache/`(特徴量、OOF), `reports/` に生成されます。

## 実行環境

| 項目 | バージョン |
|---|---|
| OS | macOS 26.3.1 (Darwin 25.3.0), Apple M5, 10 cores, 16 GB, PyTorch MPS backend |
| Python | 3.13.12 |
| core | numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, scikit-image 0.26.0, joblib 1.6.0, pillow 12.3.0, pyarrow 26.0.0 |
| models | lightgbm 4.7.0 (libomp 23.1.3), torch 2.14.1, torchvision 0.29.1, timm 1.0.30 |

すべてのパッケージは `requirements.txt` に固定されています(`pip freeze`)。

## 事前学習済み重みとライセンス

| 重み | 出典 | ライセンス |
|---|---|---|
| `timm/resnet18.a1_in1k` | `timm.create_model(..., pretrained=True)` 経由の Hugging Face Hub | Apache-2.0(モデルカード) |
| `timm/efficientnet_b0.ra_in1k`(ablation only) | timm 経由の Hugging Face Hub | Apache-2.0(モデルカード) |
| timm library | PyPI | Apache-2.0 |

その他の事前学習済み重み、外部データ、リモート API は使用していません。テスト画像は学習に一切使用していません。
画像ごとの特徴量は独立に計算され、学習されたすべての統計量(スケーラー、Box-Cox λ、CNN 入力正規化、
ターゲット標準化、ブレンド重み)は学習行のみで適合されています。

## 再現性

- すべてのシードは固定されています(`SEED = 42`)。CV プロトコルは `RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)` で、
  すべてのモデルが同じ画像フォールドを使用するため、OOF 予測をフォールド単位でブレンドし比較できます。
- Stage 0 の出力(`reports/stage0.json`)は `n_jobs` やキャッシュ設定に関係なくビット単位で同一です。LightGBM は
  `deterministic=True` と固定スレッド数で実行されます。
- 一部の PyTorch MPS カーネルはビット単位で決定論的ではないため、CNN OOF は実行ごとに末尾の桁が変わることがあります。

## パイプライン(`src/`)

| スクリプト | 目的 |
|---|---|
| `stage0_recon.py` | ラベル構造、陽性対照と堅牢な帰無分布による重複探索、PNG フォレンジック、ベースラインアンカー |
| `baseline_cheap.py` | アンカーテーブル(const / ridge / kNN / LightGBM)、CV によるターゲット変換選択、submission 001 |
| `features.py` | 画像ごとの 174 特徴量: nuisance(`q_`)、生/正規化推定量(`r_`/`n_`)、MTF 補正(`m_`)、デコンボリューション(`d_`)、スケールスペース(`x_`) |
| `stage1_report.py` | 推定量の妥当性(y との rho vs blur との rho)、Stage 1 CV、submission 002 |
| `stage2_formula.py` | べき乗則テスト、列挙式による公式探索、残差診断 |
| `ceiling.py` | 特徴空間の近傍ペアテスト、残差構造の探索 |
| `partb_report.py` | ブラー補正の評価と CV |
| `stationarity.py` | 四分割 ICC: crop は同じラベルとみなせるか |
| `cnn.py` | 全画像に対する timm CNN、D4 + nuisance 拡張、同一フォールド |
| `blend.py` | 等重み vs cross-fitted NNLS ブレンド、paired fold test、提出ファイル |
| `common.py` | パス、CV プロトコル、モデル、提出ファイル作成/検証 |

## 中心的な結論: モデル容量ではなく測定バイアス

画像には、ラベルと独立した強い nuisance が含まれています。具体的にはブラー、ノイズ(σ 4–20)、コントラスト(span 46–131)です。
古典的な微細構造推定量は、材料そのものではなく nuisance を測定してしまいます。114 個の構造特徴量のうち 84 個は、硬さよりもブラーまたはノイズと強く相関します。
ASTM E112 の平均線切片は y との rho が −0.03 ですが、ノイズとの rho は 0.76 です。
Hall-Petch べき乗則を適合すると、粒径指数は −0.5 ではなく +0.02 になります。
解析的なブラー補正(スペクトルの MTF デコンボリューション、セグメンテーション前の Richardson-Lucy、スケールスペース外挿)でも、
どの粒径推定量についても nuisance の支配を崩せませんでした。詳細は `experiments.ja.md` と PDF レポートを参照してください。
