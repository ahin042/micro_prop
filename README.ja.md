# 微細構造 → 硬さ予測

言語: [English](README.md) | [한국어](README.ko.md) | 日本語

合成微細構造に基づく材料物性予測 AI コンペ(2026年9月29日〜10月19日)のコードです。
課題: 合成材料の構造を写した 256×256 の白黒画像 1 枚から、硬さの値を予測します。
学習画像 500 枚(正解あり)、テスト画像 1000 枚です。スコアは RMSE で、低いほど良いです。

コンペのデータはこのリポジトリに**含まれていません**。自分で `data/` に置いてください(下を参照)。

## 結果

| ファイル | 内容 | CV | public LB |
|---|---|---|---|
| `004_…csv` | ResNet18 CNN + 手作り特徴に Ridge | 13.529 | 12.373 |
| `005_…csv` | 004 + DINOv2-S 特徴に Ridge (DINOv2 は学習しない) | 13.113 | 11.932 |
| `006_…csv` | 005 と同じだが CNN が画像を 2 倍の大きさで見る | 12.988 | 12.002 |
| `007_…csv` | 005 の CNN をファインチューニングした DINOv2-S に置き換え | 12.790 | 11.677 |
| `008_…csv` | 007 + ファインチューニングした DINOv2-B (4 つを同じ割合で平均) | **12.604** | **11.497** |

- どのファイルも構成モデルの単純平均です。混ぜる割合は学習していません。
- CV = 学習画像を 15 回(5-fold × 3 回、seed 42)に分けて測った RMSE の平均です。全モデルが同じ fold を使います。
- Public LB = テスト画像の 30% に対するスコアです。どのファイルでも public LB は CV より約 1.0〜1.2 低くなりました。
- ファイルの正式名、SHA-256、メモ: [reports/final_candidates.md](reports/final_candidates.md)、`reports/lb_tracking.json`。

## 実行方法

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # 正確なバージョン; 全一覧は requirements-lock.txt
brew install libomp                               # macOS のみ (LightGBM に必要)
.venv/bin/python src/run.py --mode fast           # 保存済みのモデル予測から CSV を作り直す (約 15 分)
.venv/bin/python src/run.py --mode full           # ニューラルネットもすべて学習し直す (数時間)
```

- データ: `data/train/*.png`、`data/test/*.png`、`data/train.csv`、`data/sample_submission.csv`。
- 重み: `weights/<名前>.safetensors`(下の表を参照)。すべてのスクリプトがファイルの SHA-256 を確認します。インターネットは不要です。
- `--target 004|005|006|007|008|all` で作るファイルを選びます(既定: all)。
- `HARDNESS_OUT=<フォルダ>` を指定すると、結果をリポジトリではなく新しいフォルダに書きます。
- Apple M5(16 GB)での full モードの時間: 004〜006 で約 3 時間 15 分、007 で約 2 時間、008 で約 4.5 時間が追加されます。

## 事前学習済みの重み

| 重み | 使ったファイル | ライセンス |
|---|---|---|
| `resnet18.a1_in1k` | 004〜008 (CNN と、特徴を計算する補助ネットワーク) | Apache-2.0 |
| `vit_small_patch14_dinov2.lvd142m` (DINOv2-S) | 005〜008 (005/006 は固定、007/008 はファインチューニング) | Apache-2.0 |
| `vit_base_patch14_dinov2.lvd142m` (DINOv2-B) | 008 (ファインチューニング) | Apache-2.0 |

## コード (`src/`)

| ファイル | 役割 |
|---|---|
| `run.py` | 004〜008 を 1 回のコマンドで作り直す |
| `common.py` | パス、fold、Ridge/LightGBM の補助、CSV の書き出し |
| `stage0_recon.py`、`features.py`、`p12_*.py`、`p13_*.py` | データの確認と手作りの画像特徴 |
| `p13_latnet.py` | 劣化した画像からも構造を読む補助ネットワーク |
| `cnn.py`、`cnn2.py` | ResNet18 CNN (004〜006) |
| `cnn3.py`、`cnn4.py` | DINOv2 のファインチューニング (007、008) |
| `p17_dino_probe.py`、`p13_build_004.py`、`p17_build.py` | 固定 DINOv2 特徴、モデルの平均、CSV ファイル |
| `writeup.py` | 解法 PDF (`reports/solution.pdf`) |
| そのほかの `p1*`/`p2*` スクリプト | そのほかの実験 (CSV を作り直すのには不要) |
