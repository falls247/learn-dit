"""実験ログ。標準出力と JSONL の両方に残す。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, run_dir: str | Path, filename: str = "log.jsonl") -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / filename
        self.started = time.time()

    def log(self, **record: Any) -> None:
        record.setdefault("elapsed_sec", round(time.time() - self.started, 3))
        with open(self.path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def info(self, message: str, **record: Any) -> None:
        print(message, flush=True)
        self.log(message=message, **record)

    def read_history(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, default=str)
    return path


def run_dir_for(output_dir: str | Path, task: str, name: str | None = None) -> Path:
    stamp = name or time.strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / task / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path
