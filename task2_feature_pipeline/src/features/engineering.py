"""Fit-free feature engineering from Task 1 rules (deterministic, no learning)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import rules


def canonicalize_location(value: object) -> object:
    """Map a known variant spelling to its canonical form (provisional, D09)."""
    if isinstance(value, str) and value in rules.LOCATION_CANONICAL_MAP:
        return rules.LOCATION_CANONICAL_MAP[value]
    return value


def add_rule_based_features(df: pd.DataFrame, sentinel: float = rules.STRUCTURAL_SENTINEL) -> pd.DataFrame:
    """Add structural indicators, validity flags and canonical locations.

    Implements D01 (is_never_promoted / is_never_raised + sentinel fill),
    D02 (engagement_missing flag) and D09 (tenure/wfh/attendance validity
    flags, provisional location mapping). Pure function of each row: safe to
    apply identically to training and new data without refitting.
    """
    out = df.copy()
    for source_col, flag in rules.STRUCTURAL_INDICATORS.items():
        if source_col not in out.columns:
            continue
        missing = out[source_col].isna()
        out[flag] = missing.astype(int)
        out[source_col] = out[source_col].fillna(sentinel)
    if rules.ENGAGEMENT_COLUMN in out.columns:
        out[rules.ENGAGEMENT_MISSING_FLAG] = out[rules.ENGAGEMENT_COLUMN].isna().astype(int)
    if "tenure_months" in out.columns:
        v = pd.to_numeric(out["tenure_months"], errors="coerce")
        out["tenure_invalid"] = ((v < 0) & v.notna()).astype(int)
    if "wfh_share_3m" in out.columns:
        v = pd.to_numeric(out["wfh_share_3m"], errors="coerce")
        out["wfh_invalid"] = (((v < 0) | (v > 1)) & v.notna()).astype(int)
    if "attendance_rate_3m" in out.columns:
        v = pd.to_numeric(out["attendance_rate_3m"], errors="coerce")
        out["attendance_invalid"] = (((v < 0) | (v > 1)) & v.notna()).astype(int)
    if rules.LOCATION_COLUMN in out.columns:
        out[rules.LOCATION_COLUMN] = out[rules.LOCATION_COLUMN].map(canonicalize_location)
    return out


def numeric_feature_names() -> list[str]:
    """Deterministic numeric inputs after rule-based engineering (pre-scaling)."""
    return list(rules.NUMERIC_FEATURES) + list(rules.ENGINEERED_NUMERIC_FEATURES)


def categorical_feature_names() -> list[str]:
    """Deterministic categorical inputs."""
    return list(rules.CATEGORICAL_FEATURES)


def engineered_feature_definitions() -> list[dict[str, str]]:
    """Human-readable definitions for the feature dictionary and metadata."""
    return [
        {
            "name": "attendance_rate",
            "source": "attendance_rate_3m (audited 3-month field)",
            "formula": "validated as-is; attendance_invalid = 1 when outside [0, 1]",
            "unit": "proportion",
            "assumption": "Values above 1.0 are invalid until the definition is confirmed (D09).",
        },
        {
            "name": "late_count",
            "source": "late_arrivals_3m (audited 3-month count)",
            "formula": "used as-is (count of late arrivals in 3 months)",
            "unit": "count",
            "assumption": "Monthly employee grain already aggregated; no re-aggregation performed.",
        },
        {
            "name": "avg_overtime",
            "source": "avg_overtime_hours_per_month_3m (genuine overtime field)",
            "formula": "used as-is (mean overtime hours per month over 3 months)",
            "unit": "hours/month",
            "assumption": "Total hours worked (avg_hours_worked_3m) is NOT treated as overtime.",
        },
        {
            "name": "tenure_months",
            "source": "tenure_months (audited)",
            "formula": "used as-is; tenure_invalid = 1 when negative",
            "unit": "months",
            "assumption": "Negative values are quarantined via the flag, never silently corrected (D09).",
        },
        {
            "name": "salary_vs_department_median",
            "source": "annual_fixed_pay and department_id",
            "formula": "salary_vs_department_median = annual_fixed_pay / department_median(annual_fixed_pay)",
            "unit": "ratio",
            "assumption": "Department medians are fitted on training data only; unseen departments fall back to the global median.",
        },
        {
            "name": "months_since_revision",
            "source": "months_since_last_raise (audited months since last pay rise)",
            "formula": "used as-is; is_never_raised = 1 when last_raise_pct is missing (no raise ever)",
            "unit": "months",
            "assumption": "A pay rise is the salary-revision event. months_since_last_promotion is NOT substituted (different business meaning). Values present when raises_received == 0 need HR confirmation of their meaning.",
        },
    ]
