"""Configuration loading for the Task 3 baselines."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from baselines import BASELINES_VERSION


@dataclass(frozen=True)
class Task3Config:
    package_version: str = "baselines-1.0.0"
    dataset_file: str = "data/raw/CortexHR_AI_ML_Dataset.xlsx"
    attrition_sheet: str = "attrition_dataset"
    benchmark_sheet: str = "baseline_benchmarks"
    target_column: str = "left_within_12m"
    alternate_target_column: str = "voluntary_left_within_12m"
    split_manifest: str = "artifacts/task2/features/split_manifest.json"
    task2_config: str = "task2_feature_pipeline/configs/task2_config.json"
    random_state: int = 42
    cv_folds: int = 5
    cv_shuffle: bool = True
    decision_threshold: float = 0.5
    logreg_solver: str = "lbfgs"
    logreg_max_iter: int = 5000
    logreg_c: float = 1.0
    tree_max_depth: int = 4
    selection_primary: str = "recall_mean"
    selection_tiebreak: str = "f1_mean"
    artifacts_dir: str = "artifacts/task3"
    output_dir: str = "task3_attrition_baselines/reports/generated"
    report_path: str = "task3_attrition_baselines/reports/BASELINE_REPORT.md"
    selected_filename: str = "selected_config.json"

    @classmethod
    def from_json(cls, path: str | Path) -> "Task3Config":
        from dataclasses import fields as _fields

        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise ValueError("Task 3 config must be a JSON object.")
        known = {f.name for f in _fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def __post_init__(self) -> None:
        if self.package_version != BASELINES_VERSION:
            raise ValueError(
                f"Config version {self.package_version!r} does not match package {BASELINES_VERSION!r}."
            )
