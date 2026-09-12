"""torch が無い環境では torch 依存のテストだけを除外する。"""

import importlib.util

_TORCH_DEPENDENT = [
    "test_flow_matching.py",
    "test_models.py",
    "test_data.py",
    "test_checkpoint.py",
]

collect_ignore_glob = [] if importlib.util.find_spec("torch") else list(_TORCH_DEPENDENT)
