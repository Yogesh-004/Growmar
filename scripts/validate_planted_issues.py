"""Compatibility wrapper: preserves `python scripts/validate_planted_issues.py`.

Canonical implementation: task1_data_audit/scripts/validate_planted_issues.py.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[1]
TASK_SRC = ROOT / "task1_data_audit" / "src"
for entry in (str(TASK_SRC), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

if __name__ == "__main__":
    target = ROOT / "task1_data_audit" / "scripts" / "validate_planted_issues.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
