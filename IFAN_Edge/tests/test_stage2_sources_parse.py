from __future__ import annotations

import py_compile
from pathlib import Path


def test_stage2_sources_parse() -> None:
    project_root = Path(__file__).resolve().parents[1]
    targets = (
        project_root / "ifan_edge" / "features" / "phat.py",
        project_root / "ifan_edge" / "features" / "lms.py",
        project_root / "ifan_edge" / "features" / "dual_preprocessor.py",
        project_root / "ifan_edge" / "eval" / "stage2.py",
        project_root / "ifan_edge" / "models" / "placeholders.py",
        project_root / "scripts" / "profile_stage2_model.py",
        project_root / "scripts" / "check_stage2_forward.py",
    )
    for target in targets:
        py_compile.compile(str(target), doraise=True)
