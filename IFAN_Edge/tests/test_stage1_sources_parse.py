from __future__ import annotations

import py_compile
from pathlib import Path


def test_stage1_sources_parse() -> None:
    project_root = Path(__file__).resolve().parents[1]
    targets = (
        project_root / "ifan_edge" / "bridges" / "runtime.py",
        project_root / "ifan_edge" / "features" / "phat.py",
        project_root / "ifan_edge" / "features" / "lms.py",
        project_root / "ifan_edge" / "features" / "dual_preprocessor.py",
        project_root / "scripts" / "check_stage1_shapes.py",
        project_root / "scripts" / "visualize_stage1_features.py",
    )
    for target in targets:
        py_compile.compile(str(target), doraise=True)
