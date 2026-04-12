from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def external_paths() -> List[Path]:
    root = repo_root()
    return [root, root / "icoCNN-master"]


def register_external_paths() -> Iterable[Path]:
    for path in reversed(external_paths()):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return external_paths()
