"""Configuration loading for Task 4 anomaly detection."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from anomaly import ANOMALY_VERSION


@dataclass(frozen=True)
class Task4Config:
    package_version: str = "anomaly-1.0.0"
    dataset_file: str = "data/raw/CortexHR_AI_ML_Dataset.xlsx"
    attendance_sheet: str = "attendance_monthly_features"
    payroll_sheet: str = "payroll_monthly_features"
    labels_sheet: str = "attendance_anomaly_labels"
    daily_sheet: str = "attendance_daily_examples"
    random_state: int = 42
    injection_seed: int = 42
    train_start_month: str = "2025-04"
    train_end_month: str = "2025-12"
    calibration_end_month: str = "2026-05"
    injection_rate_per_category: float = 0.02
    iforest_n_estimators: int = 200
    iforest_max_samples: int = 256
    iforest_max_features: float = 1.0
    threshold_flag_rates: tuple = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
    max_false_alarm: float = 0.10
    max_flagged_fraction: float = 0.20
    artifacts_dir: str = "artifacts/task4"
    output_dir: str = "task4_attendance_anomaly/reports/generated"
    report_path: str = "task4_attendance_anomaly/reports/ATTENDANCE_ANOMALY_REPORT.md"
    pipeline_filename: str = "attendance_iforest.joblib"
    metadata_filename: str = "attendance_anomaly_metadata.json"

    @classmethod
    def from_json(cls, path: str | Path) -> "Task4Config":
        from dataclasses import fields as _fields

        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise ValueError("Task 4 config must be a JSON object.")
        known = {f.name for f in _fields(cls)}
        cfg = cls(**{k: v for k, v in raw.items() if k in known})
        if isinstance(cfg.threshold_flag_rates, list):
            object.__setattr__(cfg, "threshold_flag_rates", tuple(cfg.threshold_flag_rates))
        return cfg

    def __post_init__(self) -> None:
        if self.package_version != ANOMALY_VERSION:
            raise ValueError(
                f"Config version {self.package_version!r} does not match package {ANOMALY_VERSION!r}."
            )
