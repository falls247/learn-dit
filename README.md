# learn-dit — DiT と Flow Matching を動かして理解する

GR00T N1.7 の「動作生成」部分を理解するための教材です。
公開データだけを使い、小型の DiT（Diffusion Transformer）を Flow Matching で学習し、
生成結果を目で確認するところまでを、MNIST → PushT の順に体験します。

対象は **Python は書けるが学習実装は初めて** という方です。
数式より「どのテンソルがどう流れるか」を追えるように作っています。

- 題材1: **MNIST** — 数字ラベルを条件に 28×28 画像を生成する（VAE なし・画素空間）
- 題材2: **PushT** — 画像2時点と自機位置を条件に、未来16ステップの目標位置を生成する

対象外: GR00T 本体、実機、言語条件、有料クラウド、学習済みモデル配布、DDPM との比較、Notebook、Web UI。

---

## 1. セットアップ

前提: Python 3.12 / [uv](https://docs.astral.sh/uv/) / NVIDIA GPU（検証想定は RTX 4070 Ti SUPER）。
CUDA Toolkit の自動更新は行いません。既存環境をそのまま使います。

```bash
uv sync                    # MNIST 教材まで
uv sync --extra pusht      # PushT のデータ取得とシミュレーション評価も行う場合
uv run learn-dit doctor    # GPU 認識と CPU スモークテスト
```

`doctor` は次を報告します。GPU が無くても CPU スモークテストは通ります（学習は遅くなります）。

- Python / PyTorch / CUDA ビルドとデバイス名・VRAM・bf16 対応
- 小さな DiT で「1バッチ学習が進む」「4ステップ生成が有限値を返す」ことの確認

## 2. MNIST を一周する

```bash
uv run learn-dit prepare --task mnist          # torchvision で取得
uv run learn-dit inspect --task mnist          # 1バッチで何が起きているかを数値で見る
uv run learn-dit train   --task mnist          # 30分上限で学習
uv run learn-dit sample  --task mnist --checkpoint outputs/mnist/<run>/checkpoints/best.pt
uv run learn-dit evaluate --task mnist --checkpoint outputs/mnist/<run>/checkpoints/best.pt
```

`inspect` は「補間の端点」「速度の向き」「初期予測がゼロであること」「10ステップで損失が下がること」を
実際の数値と画像で出します。最初にここを読むのが近道です。

## 3. PushT へ進む

```bash
uv run learn-dit prepare --task pusht          # lerobot/pusht を取得し npz へ前処理
uv run learn-dit inspect --task pusht          # 分割・時刻対応・マスクの整合性を確認
uv run learn-dit train   --task pusht
uv run learn-dit evaluate --task pusht --checkpoint outputs/pusht/<run>/checkpoints/best.pt
```

`evaluate` はシミュレータ上で **未来16ステップを生成 → 先頭8ステップを実行 → 再観測** を繰り返し、
同じ20個の初期化 seed で未学習モデルと学習後モデルを比較して JSON と MP4 を残します。

## 4. 中断と再開

学習は時間上限（既定 1,800 秒）を更新ステップの間で確認して止まります。
`Ctrl-C` でも保存されます。どちらの場合も再開できます。

```bash
uv run learn-dit train --task mnist --resume outputs/mnist/<run>/checkpoints/last.pt
```

チェックポイントにはモデル・Optimizer・学習ステップ・乱数状態・設定・正規化情報が入ります。

## 5. 出力物

実験ごとに `outputs/<task>/<日時>/` に次が残ります。

| ファイル | 内容 |
| --- | --- |
| `config.json` | 使った設定 |
| `log.jsonl` | 学習・検証ログ |
| `loss_curve.png` | 損失曲線 |
| `checkpoints/last.pt`, `best.pt` | チェックポイント |
| `samples_steps*.png` | Euler 4/8/16/32 ステップの生成比較 |
| `intermediate.png` | 生成途中経過（t=0 → 1） |
| `evaluation.json` | 評価結果 |
| `trajectory_*.png`, `*.mp4` | PushT の軌跡と動画 |

## 6. 設定の変更

```bash
uv run learn-dit train --task mnist --set train.batch_size=64 --set train.time_budget_sec=300
```

既定値は `configs/mnist.yaml` と `configs/pusht.yaml` です。

## 7. テスト

```bash
uv run pytest
```

数学（補間端点・速度符号・Euler 更新）、データ整合性（正規化往復・時刻対応・分割独立性・系列マスク）、
保存と再開を検証します。

## 8. もっと詳しく

- [dev/docs/01_flow_matching.md](dev/docs/01_flow_matching.md) — Flow Matching を最小の式で
- [dev/docs/02_dit.md](dev/docs/02_dit.md) — DiT の中身と AdaLN
- [dev/docs/03_mnist.md](dev/docs/03_mnist.md) — 画像生成として動かす
- [dev/docs/04_pusht.md](dev/docs/04_pusht.md) — 観測条件付き動作生成へ
- [dev/docs/05_verification.md](dev/docs/05_verification.md) — 検証結果と未検証事項
- [dev/adr/](dev/adr/) — 設計判断の記録

## 9. 期待値について

30 分以内に高品質な生成や高い PushT 成功率が出ることは保証しません。
低い成績もそのまま記録し、`--resume` で学習を継ぎ足せることを重視しています。

## 参照

- Flow Matching: <https://arxiv.org/abs/2210.02747>
- DiT: <https://arxiv.org/abs/2212.09748>
- Isaac GR00T: <https://github.com/NVIDIA/Isaac-GR00T>
- PushT データ: <https://huggingface.co/datasets/lerobot/pusht>
- PushT 環境: <https://github.com/huggingface/gym-pusht>
