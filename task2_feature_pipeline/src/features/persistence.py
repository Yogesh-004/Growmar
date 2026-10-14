"""Pipeline persistence: joblib artifact plus versioned JSON metadata."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from . import engineering, rules
from . import pipeline as pipeline_mod


def metadata_record(
    *,
    version: str,
    pipeline_filename: str,
    source_sheets: list[str],
    input_columns: list[str],
    output_columns: list[str],
    train_config: dict[str, Any],
    dept_references: dict[str, float],
    assumptions: list[str],
) -> dict[str, Any]:
    """Assemble the metadata JSON saved beside the pipeline artifact."""
    return {
        "feature_package_version": version,
        "pipeline_artifact": pipeline_filename,
        "source_sheets": source_sheets,
        "input_schema": input_columns,
        "engineered_features": engineering.engineered_feature_definitions(),
        "output_columns": output_columns,
        "output_column_count": len(output_columns),
        "numeric_preprocessing": pipeline_mod.NUMERIC_STEPS,
        "categorical_preprocessing": pipeline_mod.CATEGORICAL_STEPS,
        "location_mapping": {
            "mapping": dict(rules.LOCATION_CANONICAL_MAP),
            "status": rules.LOCATION_MAP_STATUS,
        },
        "department_salary_references": dept_references,
        "training": train_config,
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "assumptions_and_limitations": assumptions,
    }


def save_artifact(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    artifacts_dir: str | Path,
    metadata_filename: str = "features_metadata.json",
) -> dict[str, Path]:
    """Persist the fitted pipeline and its metadata JSON; return their paths."""
    directory = Path(artifacts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pipe_path = directory / str(metadata["pipeline_artifact"])
    meta_path = directory / metadata_filename
    joblib.dump(pipeline, pipe_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"pipeline": pipe_path, "metadata": meta_path}


def load_artifact(
    artifacts_dir: str | Path,
    pipeline_filename: str = "pipeline.joblib",
    metadata_filename: str = "features_metadata.json",
) -> tuple[Pipeline, dict[str, Any]]:
    """Reload a fitted pipeline and validate it against its metadata version."""
    directory = Path(artifacts_dir)
    pipe_path = directory / pipeline_filename
    meta_path = directory / metadata_filename
    if not pipe_path.exists():
        raise FileNotFoundError(f"Pipeline artifact not found: {pipe_path}. Run fit first.")
    if not meta_path.exists():
        raise FileNotFoundError(f"Pipeline metadata not found: {meta_path}. Run fit first.")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    pipeline = joblib.load(pipe_path)
    if not isinstance(pipeline, Pipeline):
        raise ValueError(f"Artifact {pipe_path} is not a scikit-learn Pipeline.")
    return pipeline, metadata
