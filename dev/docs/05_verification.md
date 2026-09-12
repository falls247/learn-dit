# 05 検証記録（2026-09-12 時点）

SOW 第7章に沿って、**検証できたこと / できていないこと**を分けて記録します。
実装状況はこのファイルで管理し、SOW 本体は更新しません。

## 検証環境

| 項目 | 内容 |
| --- | --- |
| 実施環境 | 実装作業環境（ネットワーク無効・PyTorch 未導入） |
| GPU | **未確認**（RTX 4070 Ti SUPER での実測は未実施） |
| 実施日 | 2026-09-12 |

## 状態

| 受入条件 | 状態 | 備考 |
| --- | --- | --- |
| 補間端点・速度符号・Euler 更新の検証 | テスト実装済み・**未実行** | `tests/test_flow_matching.py` |
| 正規化往復・時刻対応・分割独立性・系列マスク | テスト実装済み・**未実行** | `tests/test_data.py` |
| 少数サンプルでの損失低下と重み更新 | テスト実装済み・**未実行** | `tests/test_models.py` |
| 保存後の推論と追加学習 | テスト実装済み・**未実行** | `tests/test_checkpoint.py` |
| MNIST の 0〜9 条件付き生成と途中経過 | 実装済み・**未実行** | `learn-dit sample --task mnist` |
| PushT の動作実行・動画・評価 JSON | 実装済み・**未実行** | ネットワークとシミュレータが必要 |
| 未学習/学習後の 20 seed 比較 | 実装済み・**未実行** | `learn-dit evaluate --task pusht` |
| 取得から保存モデル出力までの再実行 | 実装済み・**未実行** | 上記が通れば成立する想定 |
| GPU 検証・所要時間の報告 | **未実施** | 実行環境で埋める |

## 最初に実行する順序

```bash
uv sync
uv run pytest                        # 数学・データ・保存の検証
uv run learn-dit doctor              # GPU 認識と CPU スモークテスト
uv run learn-dit inspect --task mnist
uv run learn-dit train  --task mnist --set train.time_budget_sec=120   # まず短時間で通す
uv run learn-dit sample --task mnist --checkpoint outputs/mnist/<run>/checkpoints/last.pt
```

ここまで通ってから 1,800 秒の本番学習と PushT に進むのが安全です。

## 実行時に確認が必要な点（既知の要調整箇所）

1. **lerobot のバージョン差**
   `LeRobotDataset` の import 経路とフレームのキー名はバージョンで変わります。
   `src/learn_dit/data/pusht.py` は 2 つの import 経路と複数のキー候補を試し、
   合わなければ候補一覧を添えて失敗します。実測に合わせてここを直してください。
2. **画像キー名**
   `observation.image` を第一候補にしています。実データのキーは `prepare` の
   エラーメッセージに出ます。
3. **gym-pusht の観測形式**
   `obs_type="pixels_agent_pos"` を前提に `pixels` / `agent_pos` を読んでいます。
4. **動作の座標系**
   データセットの動作と環境の `action_space`（0〜512）が同じ尺度である前提です。
   `sample` の軌跡がデータセット軌跡と重なるかで確認できます。
5. **bf16 autocast**
   `train.amp=true` は CUDA かつ bf16 対応時のみ有効になります。不安定なら
   `--set train.amp=false` で切れます。

## 実測記入欄（実行後に埋める）

| 項目 | 値 |
| --- | --- |
| GPU 名 / VRAM | |
| MNIST 1,800 秒で到達したステップ数 | |
| MNIST 検証損失（最良） | |
| PushT 1,800 秒で到達したステップ数 | |
| PushT 成功率（未学習 / 学習後） | / |
| PushT 最大到達報酬（未学習 / 学習後） | / |
| 1 回の生成の平均推論時間（32 ステップ） | |
