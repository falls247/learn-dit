"""CLI。`uv run learn-dit <subcommand>` で呼ぶ。

  doctor    環境診断 (GPU 有無 / CPU スモークテスト)
  prepare   データ取得・前処理
  inspect   データと 1 バッチ学習の解説
  train     学習 (--resume で再開)
  sample    生成 (--checkpoint)
  evaluate  評価 (--checkpoint、PushT はシミュレーション実行)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from learn_dit.config import default_config_path, load_config
from learn_dit.utils.logging import run_dir_for, write_json


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=("mnist", "pusht"), default="mnist", help="題材")
    parser.add_argument("--config", default=None, help="設定 YAML (既定: configs/<task>.yaml)")
    parser.add_argument("--device", default=None, help="auto | cuda | cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None, help="実験出力の親ディレクトリ")
    parser.add_argument("--run-name", default=None, help="実験ディレクトリ名 (既定: 日時)")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="設定の上書き。例: --set train.batch_size=64 --set sample.steps=8",
    )


def _load(args: argparse.Namespace):
    path = Path(args.config) if args.config else default_config_path(args.task)
    overrides = list(args.overrides)
    overrides.append(f"task={args.task}")
    if args.device:
        overrides.append(f"device={args.device}")
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    if args.output_dir:
        overrides.append(f"output_dir={args.output_dir}")
    return load_config(path, overrides)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="learn-dit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="環境診断")
    _add_common(p_doctor)
    p_doctor.add_argument("--no-gpu", action="store_true", help="GPU スモークテストを省く")

    p_prepare = sub.add_parser("prepare", help="データ取得・前処理")
    _add_common(p_prepare)
    p_prepare.add_argument("--revision", default=None, help="PushT のデータリビジョン")
    p_prepare.add_argument("--limit-episodes", type=int, default=None, help="動作確認用に件数を制限")

    p_inspect = sub.add_parser("inspect", help="データと 1 バッチ学習の解説")
    _add_common(p_inspect)

    p_train = sub.add_parser("train", help="学習")
    _add_common(p_train)
    p_train.add_argument("--resume", default=None, help="再開するチェックポイント")
    p_train.add_argument("--no-download", action="store_true")

    p_sample = sub.add_parser("sample", help="生成")
    _add_common(p_sample)
    p_sample.add_argument("--checkpoint", default=None, help="未指定なら未学習モデルで生成する")

    p_eval = sub.add_parser("evaluate", help="評価")
    _add_common(p_eval)
    p_eval.add_argument("--checkpoint", default=None)
    p_eval.add_argument("--no-compare-untrained", action="store_true",
                        help="PushT で未学習モデルとの比較を省く")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load(args)

    if args.command == "doctor":
        from learn_dit.doctor import run_doctor

        report = run_doctor(gpu=not args.no_gpu)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        run_dir = run_dir_for(cfg.output_dir, "doctor", args.run_name)
        write_json(run_dir / "doctor.json", report)
        return 0

    if args.command == "prepare":
        if cfg.task == "mnist":
            from learn_dit.data.mnist import download_mnist

            root = download_mnist(cfg.data.root)
            print(f"MNIST を {root} に取得した。分割は seed={cfg.seed} で "
                  f"{60000 - cfg.data.val_size} / {cfg.data.val_size} 件。")
            return 0
        from learn_dit.data.pusht import prepare_pusht

        meta = prepare_pusht(
            cfg.data.root,
            repo_id=cfg.data.repo_id,
            revision=args.revision or cfg.data.revision,
            image_size=cfg.data.image_size,
            limit_episodes=args.limit_episodes,
        )
        print(json.dumps(meta, ensure_ascii=False, indent=2)[:2000])
        return 0

    if args.command == "inspect":
        from learn_dit.inspect import inspect_task

        run_dir = run_dir_for(cfg.output_dir, f"{cfg.task}_inspect", args.run_name)
        report = inspect_task(cfg, run_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n出力: {run_dir}")
        return 0

    if args.command == "train":
        from learn_dit.train import train

        run_dir = (
            Path(args.resume).parents[1] if args.resume and args.run_name is None
            else run_dir_for(cfg.output_dir, cfg.task, args.run_name)
        )
        summary = train(cfg, run_dir, resume=args.resume, download=not args.no_download)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "sample":
        from learn_dit.sample import run_sample

        run_dir = run_dir_for(cfg.output_dir, f"{cfg.task}_sample", args.run_name)
        result = run_sample(cfg, args.checkpoint, run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n出力: {run_dir}")
        return 0

    if args.command == "evaluate":
        from learn_dit.evaluate import run_evaluate

        run_dir = run_dir_for(cfg.output_dir, f"{cfg.task}_eval", args.run_name)
        result = run_evaluate(
            cfg, args.checkpoint, run_dir, compare_untrained=not args.no_compare_untrained
        )
        summary = {k: v for k, v in result.items() if k != "variants"}
        if "variants" in result:
            summary["variants"] = {
                name: {k: v for k, v in payload.items() if k != "episodes"}
                for name, payload in result["variants"].items()
            }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\n出力: {run_dir}")
        return 0

    raise SystemExit(f"未知のコマンド: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
