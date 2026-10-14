"""Schema validation and input loading (explicit sheets only, no URL access)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import rules


class SchemaError(ValueError):
    """Raised when an input frame does not match the expected feature schema."""


def expected_input_columns() -> list[str]:
    """Deterministic raw feature columns (base numeric + categorical)."""
    return list(rules.NUMERIC_FEATURES) + list(rules.CATEGORICAL_FEATURES)


def load_attrition_sheet(workbook: str | Path, sheet: str) -> pd.DataFrame:
    """Read exactly one worksheet; never the whole workbook at once."""
    path = Path(workbook)
    if not path.exists():
        raise FileNotFoundError(
            f"Workbook not found: {path}. Place it at data/raw/CortexHR_AI_ML_Dataset.xlsx."
        )
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except ValueError as exc:
        raise SchemaError(f"Worksheet {sheet!r} not readable: {exc}") from exc


def validate_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Check required feature columns exist; return them in deterministic order.

    Extra columns (including leakage fields, identifiers and targets) are never
    selected. Missing required columns raise SchemaError listing their names.
    """
    missing = [c for c in expected_input_columns() if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required feature column(s): {', '.join(missing)}")
    return df[expected_input_columns()].copy()


def build_supervised_frame(
    raw: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split a raw attrition frame into features (X), target (y) and keys.

    Keys carry the technical ``row_id`` when present (else the positional
    index) so later models can rejoin labels without touching employee
    identifiers or labels. Rows with missing labels are dropped per D03 and
    reported via the returned mask companion (callers must align keys).
    """
    if target_column not in raw.columns:
        raise SchemaError(f"Target column {target_column!r} is absent.")
    labelled = raw[raw[target_column].notna()].copy()
    keys = pd.DataFrame(
        {
            "row_id": labelled["row_id"] if "row_id" in labelled.columns else labelled.index.to_numpy(),
            "row_position": labelled.index.to_numpy(),
        }
    ).reset_index(drop=True)
    X = validate_feature_frame(labelled).reset_index(drop=True)
    y = labelled[target_column].reset_index(drop=True)
    # Defense in depth: targets/identifiers/leakage must never be features.
    leaked = [c for c in X.columns if c in rules.EXCLUDED_COLUMNS]
    if leaked:
        raise SchemaError(f"Excluded column(s) reached model inputs: {', '.join(leaked)}")
    return X, y, keys
