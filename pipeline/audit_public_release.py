#!/usr/bin/env python3
"""Fail if a code-only release contains likely restricted or generated payloads."""
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git", ".venv", ".venv_epf", "__pycache__", ".ipynb_checkpoints",
    "logs", "outputs", "06_outputs", "dist",
}
ALLOWED_DATA = {
    ROOT / "reference_results" / "lear_qra_metrics.csv",
    ROOT / "reference_results" / "three_way_qw_metrics.csv",
    ROOT / "reference_results" / "three_way_qw_dm.csv",
}
REQUIRED_RELEASE_FILES = {
    ROOT / "LICENSE",
    ROOT / "CITATION.cff",
    ROOT / "README.md",
    ROOT / "requirements.txt",
    ROOT / "environment.yml",
    ROOT / "docs" / "DATA_ACCESS.md",
}
SENSITIVE_NAMES = re.compile(
    r"(DAM_outturn|model_table.*\.parquet$|predictions\.csv$|_predictions\.csv$|"
    r"natural_gas.*\.csv$|emissions_allowances.*\.csv$)", re.IGNORECASE
)
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|bearer)\s*[:=]\s*['\"][^'\"]+"),
    re.compile(r"/" + r"Users/[^/\s]+/"),
]


def included_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def main() -> None:
    failures: list[str] = []
    files = included_files()
    for path in sorted(REQUIRED_RELEASE_FILES):
        if not path.is_file():
            failures.append(f"required release file missing: {path.relative_to(ROOT)}")
    for path in files:
        if path in ALLOWED_DATA:
            continue
        if SENSITIVE_NAMES.search(path.name):
            failures.append(f"restricted/generated data filename: {path.relative_to(ROOT)}")
        if path.suffix.lower() in {".py", ".md", ".yaml", ".yml", ".tex", ".sh"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    failures.append(f"credential or absolute-path pattern: {path.relative_to(ROOT)}")
                    break
    if failures:
        print("PUBLIC RELEASE AUDIT FAILED")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print(f"PUBLIC RELEASE AUDIT PASSED: {len(files)} included files checked")


if __name__ == "__main__":
    main()
