"""Compatibility wrapper: preserves the historic `python scripts/run_audit.py` command.

Canonical implementation: task1_data_audit/scripts/run_audit.py (no duplicated logic).
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
    sys.argv[0] = str(ROOT / "task1_data_audit" / "scripts" / "run_audit.py")
    runpy.run_path(str(ROOT / "task1_data_audit" / "scripts" / "run_audit.py"), run_name="__main__")
