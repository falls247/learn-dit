# 03 MNIST で動かす

## データ

torchvision の公式学習データ 60,000 件を、seed 42 で **55,000 / 5,000** に分割します。
画素は `[0,1]` → `[-1,1]` に変換します。ノイズ `N(0,I)` と尺度を合わせるためです。
VAE は使いません。28×28 をそのまま扱います。

分割は `split_indices()` 一箇所で行い、「同じ seed なら同じ分割」「train と val は重複しない」を
テストで固定しています。

## まず inspect

```bash
uv run learn-dit inspect --task mnist
```

出力（`inspect_report.json` と PNG）で次を確認できます。

- バッチの形と値域（`[-1,1]` に収まっているか）
- `t=0` の `x_t` がノイズと一致、`t=1` がデータと一致（最大絶対誤差が 0）
- 初期予測が 0（AdaLN-Zero）
- 同じ 1 バッチを 10 回更新すると損失が下がり、更新されたパラメータ数が出る
- `interpolation_t.png`: 同じ画像に対する `t=0 → 1` の見え方

## 学習

```bash
uv run learn-dit train --task mnist
```

AdamW / lr 1e-4 / バッチ 128 / seed 42、時間上限 1,800 秒。
上限は更新ステップの間で確認するので、保存で数秒超えることは許容しています。
取得・前処理・学習後の評価はこの 1,800 秒に含めません。

## 生成

```bash
uv run learn-dit sample --task mnist --checkpoint outputs/mnist/<run>/checkpoints/best.pt
```

- `samples_steps4/8/16/32.png`: 0〜9 の条件付き生成。**同じ初期ノイズ**でステップ数だけを変えています
- `intermediate.png`: `t=0 → 1` の途中経過
- `sample_summary.json`: ステップ数ごとの生成時間

`--checkpoint` を省くと未学習モデルで生成します。学習前後の比較に使えます。

## 見方のコツ

30 分では「数字らしい形が出始める」程度になることがあります。
きれいに出ないときは、まず損失曲線が下がっているか、`samples_steps32.png` が
ステップ数の少ない版より整っているかを見てください。
どちらも満たしていれば仕組みは動いていて、あとは学習量の問題です。
`--resume` で継ぎ足せます。
