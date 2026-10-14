"""Attendance feature frame, periods, and score convention (no labels as inputs)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ATTENDANCE_NUMERIC_FEATURES = (
    "attendance_rate",
    "late_arrivals",
    "avg_hours_worked",
    "overtime_hours",
    "present_days",
    "scheduled_days",
    "wfh_days",
    "unpaid_absence_days",
    "longest_unpaid_streak_days",
    "paid_leave_days",
    "half_days_unpaid",
    "late_arrivals_avg_3m",
    "overtime_hours_avg_3m",
    "attendance_rate_avg_3m",
    "months_of_history",
    "attendance_invalid",
)

PREPROCESSING_NOTE = "median imputation + RobustScaler, fitted on training months only (Task 2 convention)"


def attendance_feature_columns() -> list[str]:
    return list(ATTENDANCE_NUMERIC_FEATURES)


def build_preprocessor() -> ColumnTransformer:
    pre = ColumnTransformer(
        [("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]),
          attendance_feature_columns())],
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")
    return pre


def split_periods(
    merged: pd.DataFrame, train_start: str, train_end: str, calibration_end: str
) -> dict[str, pd.DataFrame]:
    """Partition monthly rows into train / calibration / final month ranges."""
    months = pd.to_datetime(merged["month"])
    lo = pd.Timestamp(f"{train_start}-01")
    mid = pd.Timestamp(f"{train_end}-01") + pd.offsets.MonthBegin(1)
    hi = pd.Timestamp(f"{calibration_end}-01") + pd.offsets.MonthBegin(1)
    parts = {
        "train": merged.loc[(months >= lo) & (months < mid)].reset_index(drop=True),
        "calibration": merged.loc[(months >= mid) & (months < hi)].reset_index(drop=True),
        "final": merged.loc[months >= hi].reset_index(drop=True),
    }
    for name, frame in parts.items():
        if frame.empty:
            raise ValueError(f"Period split left {name!r} empty; check month boundaries.")
    return parts


def period_bounds(parts: dict[str, pd.DataFrame]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, frame in parts.items():
        out[f"{name}_min"] = str(frame["month"].min())
        out[f"{name}_max"] = str(frame["month"].max())
        out[f"{name}_rows"] = str(len(frame))
    return out


def anomaly_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Higher anomaly_score means more anomalous (-decision_function).

    In scikit-learn, lower decision_function/score_samples values indicate more
    anomalous observations; negating gives the documented convention.
    """
    clf = model.named_steps["iforest"]
    return np.asarray(-clf.decision_function(model.named_steps["prep"].transform(X)), dtype=float)
