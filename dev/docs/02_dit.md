# 02 DiT の中身

`src/learn_dit/models/dit.py` が本体です。MNIST と PushT で同じクラスを使います。

構成は SOW のとおり、**4 ブロック / 隠れ次元 128 / 4 ヘッド**です。

## トークン化

DiT は Transformer なので、入力を「トークン列」にします。

| 題材 | トークン | 数 | 1 トークンの次元 |
| --- | --- | --- | --- |
| MNIST | 4×4 パッチ | 49 (=7×7) | 16 (=4·4·1) |
| PushT | 未来の 1 時刻 | 16 | 2 (x, y) |

同じ Transformer が、画像では「空間の並び」、動作では「時間の並び」を扱います。
位置埋め込みは学習可能なパラメータです。

## 時刻の入れ方（AdaLN）

時刻 `t` は正弦波埋め込み → MLP で条件ベクトル `c` にします。
`c` から各ブロックの `shift` / `scale` / `gate` を作り、LayerNorm 後に掛けます。

```
h = norm(x) * (1 + scale) + shift
x = x + gate * Attention(h)
```

`shift / scale / gate` を作る線形層はゼロ初期化します（AdaLN-Zero）。
そのため **学習開始時点のモデル出力は厳密に 0** です。
`tests/test_models.py::test_mnist_model_shape_and_zero_init` で確認できます。
残差に何も足さない状態から始めるので、初期の学習が安定します。

## 条件の入れ方

- MNIST: 数字ラベルを埋め込み、時刻ベクトルに足す（`c = time(t) + label_emb`）
- PushT: 観測を Cross-Attention で参照する

Cross-Attention を有効にしたブロックは、Self-Attention → Cross-Attention → MLP の 3 段になり、
変調パラメータも 6 個から 9 個に増えます。

## 観測エンコーダ（PushT）

小型 CNN（stride 2 を 4 回）で 96×96 → 6×6 にし、36 トークン × 2 時点 = 72 トークン。
自機位置は MLP で 1 時点 1 トークンにして 2 トークン。
合計 74 トークンを `memory` として Cross-Attention から参照します。
どの時点の観測かは frame 埋め込みで区別します。

生成時は Euler の各ステップで観測が変わらないため、`encode_obs` を 1 回だけ実行して
`memory` を再利用します（`tests/test_models.py::test_pusht_memory_reuse_matches_full_forward`）。
32 ステップなら CNN の実行回数が 32 回から 1 回になります。
