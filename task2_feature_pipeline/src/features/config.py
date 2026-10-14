"""Configuration loading for the Task 2 feature package."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class Task2Config:
    package_version: str = "features-1.0.0"
    dataset_file: str = "data/raw/CortexHR_AI_ML_Dataset.xlsx"
    attrition_sheet: str = "attrition_dataset"
    target_column: str = "left_within_12m"
    alternate_target_column: str = "voluntary_left_within_12m"
    random_state: int = 42
    test_size: float = 0.2
    artifacts_dir: str = "artifacts/task2/features"
    output_dir: str = "data/processed/task2_features"
    pipeline_filename: str = "pipeline.joblib"
    metadata_filename: str = "features_metadata.json"
    split_manifest_filename: str = "split_manifest.json"
    numeric_imputation: str = "median"
    scaler: str = "RobustScaler"
    categorical_imputation: str = "most_frequent"
    encoder: str = "OneHotEncoder(handle_unknown=ignore)"
    structural_sentinel: float = 9999.0
    excluded_columns: tuple[str, ...] = field(default_factory=tuple)
    monthly_cutoff_month: str = "2026-04"
    monthly_pipeline_filename: str = "monthly_pipeline.joblib"
    monthly_metadata_filename: str = "monthly_features_metadata.json"
    monthly_manifest_filename: str = "monthly_split_manifest.json"

    def __post_init__(self) -> None:
        from features import FEATURES_VERSION as _pkg_version

        if self.package_version != _pkg_version:
            raise ValueError(
                f"Config version {self.package_version!r} does not match package {_pkg_version!r}."
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "Task2Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Task 2 config must be a JSON object.")
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in known}
        if isinstance(filtered.get("excluded_columns"), list):
            filtered["excluded_columns"] = tuple(filtered["excluded_columns"])
        return cls(**filtered)

    def resolve(self, root: str | Path, rel: str | Path) -> Path:
        """Resolve a project-relative path against the project root."""
        p = Path(rel)
        return Path(root) / p if not p.is_absolute() else p
