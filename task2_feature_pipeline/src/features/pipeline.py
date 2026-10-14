"""Reusable scikit-learn pipeline: rules + fitted transforms + output order."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

from . import engineering, rules
from .transformers import DeptMedianRatio

NUMERIC_STEPS = "median imputation (train-fitted) + RobustScaler (D04: robust to skew)"
CATEGORICAL_STEPS = "most-frequent imputation (train-fitted) + one-hot, unknown categories ignored"


def build_column_transformer(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Shared numeric/categorical preprocessing (train-fitted, deterministic)."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric, list(numeric_features)),
            ("cat", categorical, list(categorical_features)),
        ],
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def build_pipeline(sentinel: float = rules.STRUCTURAL_SENTINEL) -> Pipeline:
    """Assemble the full preprocessing pipeline in deterministic order.

    Stages: stateless Task 1 rules -> fitted department salary reference ->
    ColumnTransformer (numeric / categorical) with pandas output. All learned
    statistics (medians, modes, scaler params, dept references, categories)
    live inside the fitted object, so transformation of new rows is identical.
    """
    rules_step = FunctionTransformer(
        engineering.add_rule_based_features,
        kw_args={"sentinel": sentinel},
        validate=False,
    )
    dept_step = DeptMedianRatio()
    preprocessor = build_column_transformer(
        engineering.numeric_feature_names(), engineering.categorical_feature_names()
    )
    return Pipeline(
        [
            ("rules", rules_step),
            ("dept_salary_reference", dept_step),
            ("preprocessor", preprocessor),
        ]
    )


def transform_frame(pipeline: Pipeline, X: pd.DataFrame, output_columns: list[str] | None = None) -> pd.DataFrame:
    """Transform features and enforce the persisted output column order."""
    frame = pipeline.transform(X)
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    if output_columns is not None:
        missing = [c for c in output_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"Transformed frame lacks expected column(s): {', '.join(missing)}")
        frame = frame[output_columns]
    return frame


def input_columns() -> list[str]:
    """Raw columns the fitted pipeline consumes (validated schema)."""
    from .schema import expected_input_columns

    return expected_input_columns()
