"""Pytest path bootstrap: allow `pytest -q` from the project root."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for entry in (ROOT / "task1_data_audit" / "src", ROOT / "task2_feature_pipeline" / "src", ROOT / "task3_attrition_baselines" / "src", ROOT / "task4_attendance_anomaly" / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
