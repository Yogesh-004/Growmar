"""Standalone employee-month processing for attendance/payroll sheets.

Grain is one row per employee x month. This path never merges into the older
attrition observations: monthly history starts 2025-04 while most attrition
observations predate it, and monthly data extends past them (future-leakage
risk). All trailing windows use months at or before each row's month only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ATTENDANCE_SHEET = "attendance_monthly_features"
PAYROLL_SHEET = "payroll_monthly_features"

MONTHLY_COMPONENT_VERSION = "monthly-1.0.0"

KEY_COLUMNS = ["employee_id", "month"]

EXPECTED_ATTENDANCE_COLUMNS = (
    "attendance_feature_id",
    "employee_id",
    "month",
    "department_id",
    "job_level",
    "scheduled_days",
    "present_days",
    "wfh_days",
    "half_days_unpaid",
    "paid_leave_days",
    "unpaid_absence_days",
    "longest_unpaid_streak_days",
    "late_arrivals",
    "avg_hours_worked",
    "overtime_hours",
    "attendance_rate",
)

EXPECTED_PAYROLL_COLUMNS = (
    "payroll_item_id",
    "run_id",
    "employee_id",
    "month",
    "department_id",
    "role_id",
    "job_level",
    "overtime_eligible",
    "monthly_fixed_gross",
    "payable_days",
    "lop_days",
    "overtime_hours_paid",
    "overtime_pay",
    "variable_pay",
    "variable_pay_type",
    "gross_pay",
    "pf_employee",
    "esi_employee",
    "professional_tax",
    "tds",
    "other_deductions",
    "total_deductions",
    "net_pay",
)

# Technical line identifier disregarded by the duplicate rule (verified: the
# 15 duplicate employee-month groups are identical on all other fields).
PAYROLL_ID_COLUMN = "payroll_item_id"

# Trailing window: current month plus the two preceding months, min 1 present.
TRAILING_WINDOW = 3

MONTHLY_NUMERIC_FEATURES = (
    "scheduled_days",
    "present_days",
    "wfh_days",
    "half_days_unpaid",
    "paid_leave_days",
    "unpaid_absence_days",
    "longest_unpaid_streak_days",
    "late_arrivals",
    "avg_hours_worked",
    "overtime_hours",
    "attendance_rate",
    "late_arrivals_avg_3m",
    "overtime_hours_avg_3m",
    "attendance_rate_avg_3m",
    "months_of_history",
    "monthly_fixed_gross",
    "payable_days",
    "lop_days",
    "overtime_hours_paid",
    "overtime_pay",
    "variable_pay",
    "gross_pay",
    "pf_employee",
    "esi_employee",
    "professional_tax",
    "tds",
    "other_deductions",
    "total_deductions",
    "net_pay",
    "salary_vs_dept_median_monthly",
    "attendance_invalid",
    "job_level_mismatch",
)

MONTHLY_CATEGORICAL_FEATURES = (
    "department_id",
    "role_id",
    "overtime_eligible",
    "variable_pay_type",
)


class MonthlyDataError(ValueError):
    """Raised for schema violations, conflicting duplicates or bad joins."""


def load_monthly_sheets(workbook: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read exactly the two authorized monthly sheets (explicit selection)."""
    path = Path(workbook)
    if not path.exists():
        raise FileNotFoundError(
            f"Workbook not found: {path}. Place it at data/raw/CortexHR_AI_ML_Dataset.xlsx."
        )
    try:
        att = pd.read_excel(path, sheet_name=ATTENDANCE_SHEET)
        pay = pd.read_excel(path, sheet_name=PAYROLL_SHEET)
    except ValueError as exc:
        raise MonthlyDataError(f"Monthly worksheet not readable: {exc}") from exc
    for name, frame, expected in (
        (ATTENDANCE_SHEET, att, EXPECTED_ATTENDANCE_COLUMNS),
        (PAYROLL_SHEET, pay, EXPECTED_PAYROLL_COLUMNS),
    ):
        missing = [c for c in expected if c not in frame.columns]
        if missing:
            raise MonthlyDataError(f"{name} lacks column(s): {', '.join(missing)}")
    for frame, name in ((att, ATTENDANCE_SHEET), (pay, PAYROLL_SHEET)):
        frame["month"] = pd.to_datetime(frame["month"], errors="coerce")
        if frame["month"].isna().any():
            raise MonthlyDataError(f"{name} has unparseable month values.")
        if frame["employee_id"].isna().any():
            raise MonthlyDataError(f"{name} has missing employee_id values.")
    return att, pay


def deduplicate_payroll(pay: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop confirmed duplicate payroll lines, ignoring only payroll_item_id.

    A duplicate group must match on every other field; groups with conflicting
    non-key values raise MonthlyDataError (counts only, never identifiers).
    Returns the deduplicated frame plus an audit-log summary.
    """
    compare = [c for c in pay.columns if c != PAYROLL_ID_COLUMN]
    nunique = pay.groupby(["employee_id", "month"])[compare].nunique(dropna=False)
    conflicting = nunique[(nunique > 1).any(axis=1)]
    if len(conflicting):
        raise MonthlyDataError(
            f"{len(conflicting)} employee-month group(s) share a key but disagree on "
            "non-key fields; refusing to deduplicate. Resolve the source conflict first."
        )
    dup_mask = pay.duplicated(subset=compare, keep="first")
    removed = int(dup_mask.sum())
    return pay.loc[~dup_mask].reset_index(drop=True), {"removed_duplicate_lines": removed}


def validate_and_join(
    att: pd.DataFrame, pay: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """One-to-one inner join on (employee_id, month) with aggregate validation.

    Attendance keys missing from payroll, payroll-only keys, and job_level
    disagreements are counted (never listed). The payroll-only key is excluded
    from features and reported for the metadata log.
    """
    att_keys = set(zip(att["employee_id"].astype(str), att["month"].astype(str)))
    pay_keys = set(zip(pay["employee_id"].astype(str), pay["month"].astype(str)))
    merged = pd.merge(
        att, pay, on=["employee_id", "month"], how="inner",
        suffixes=("_att", "_pay"), validate="one_to_one",
    )
    stats = {
        "attendance_keys": len(att_keys),
        "payroll_keys": len(pay_keys),
        "matched_keys": len(att_keys & pay_keys),
        "attendance_only_keys": len(att_keys - pay_keys),
        "payroll_only_keys": len(pay_keys - att_keys),
        "department_agree": int((merged["department_id_att"] == merged["department_id_pay"]).sum()),
        "job_level_mismatches": int((merged["job_level_att"] != merged["job_level_pay"]).sum()),
        "merged_rows": len(merged),
    }
    return merged, stats


def add_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fit-free monthly engineering: flags, trailing windows, pay-type fill.

    Trailing 3-month means use each row's month plus the two preceding months
    only (never future months). Disputed job levels are never used as inputs:
    both source values stay out of features while job_level_mismatch flags the
    disagreement for source-owner confirmation.
    """
    out = df.copy()
    out = out.sort_values(["employee_id", "month"]).reset_index(drop=True)
    if "department_id_att" in out.columns:
        # department_id agrees across sources (verified at join); the single
        # canonical column feeds the salary reference and the encoder.
        out["department_id"] = out["department_id_att"]
    rate = pd.to_numeric(out["attendance_rate"], errors="coerce")
    out["attendance_invalid"] = (((rate < 0) | (rate > 1)) & rate.notna()).astype(int)
    out["job_level_mismatch"] = (
        (out["job_level_att"] != out["job_level_pay"])
        & out["job_level_att"].notna() & out["job_level_pay"].notna()
    ).astype(int)
    out["variable_pay_type"] = out["variable_pay_type"].fillna("none")
    grouped = out.groupby("employee_id", sort=False)
    for source, target in (
        ("late_arrivals", "late_arrivals_avg_3m"),
        ("overtime_hours", "overtime_hours_avg_3m"),
        ("attendance_rate", "attendance_rate_avg_3m"),
    ):
        values = pd.to_numeric(out[source], errors="coerce")
        out[target] = values.groupby(out["employee_id"], sort=False).transform(
            lambda s: s.rolling(TRAILING_WINDOW, min_periods=1).mean()
        )
    out["months_of_history"] = grouped.cumcount().to_numpy() + 1
    return out


def monthly_numeric_features() -> list[str]:
    return list(MONTHLY_NUMERIC_FEATURES)


def monthly_categorical_features() -> list[str]:
    return list(MONTHLY_CATEGORICAL_FEATURES)


def merged_input_columns() -> list[str]:
    """Post-join columns the monthly pipeline consumes (deterministic order).

    The merge suffixes only the overlapping non-key columns (department_id,
    job_level); all other columns keep their source names.
    """
    overlap = (set(EXPECTED_ATTENDANCE_COLUMNS) & set(EXPECTED_PAYROLL_COLUMNS)) - set(KEY_COLUMNS)
    att_part = [f"{c}_att" if c in overlap else c for c in EXPECTED_ATTENDANCE_COLUMNS]
    pay_part = [f"{c}_pay" if c in overlap else c for c in EXPECTED_PAYROLL_COLUMNS if c not in KEY_COLUMNS]
    return list(dict.fromkeys(att_part + pay_part))


def prepare_monthly_frame(att: pd.DataFrame, pay: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """Deduplicate payroll, join 1:1, return (joined, dedup_log, join_stats)."""
    pay_clean, dedup_log = deduplicate_payroll(pay)
    merged, join_stats = validate_and_join(att, pay_clean)
    return merged, dedup_log, join_stats


def chronological_split(
    merged: pd.DataFrame, cutoff_month: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Split monthly rows into earlier-months train and later-months holdout.

    Cutoff is a YYYY-MM month string; train holds month < cutoff, holdout holds
    month >= cutoff, so no future month ever trains. Raises MonthlyDataError
    when the cutoff would leave either side empty.
    """
    try:
        cutoff = pd.Timestamp(f"{cutoff_month}-01")
    except ValueError as exc:
        raise MonthlyDataError(f"Invalid monthly cutoff {cutoff_month!r}; use YYYY-MM.") from exc
    months = pd.to_datetime(merged["month"])
    train = merged.loc[months < cutoff].reset_index(drop=True)
    test = merged.loc[months >= cutoff].reset_index(drop=True)
    if train.empty or test.empty:
        raise MonthlyDataError(
            f"Cutoff {cutoff_month} leaves an unusable partition "
            f"(train {len(train)} rows, holdout {len(test)} rows)."
        )
    info: dict[str, object] = {
        "cutoff_month": cutoff_month,
        "train_month_min": str(train["month"].min()),
        "train_month_max": str(train["month"].max()),
        "test_month_min": str(test["month"].min()),
        "test_month_max": str(test["month"].max()),
        "train_rows": len(train),
        "test_rows": len(test),
        "no_future_leak": bool(train["month"].max() < cutoff <= test["month"].min()),
    }
    return train, test, info


def alignment_frame(merged: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Local row-matching manifest (kept out of the feature matrix and Git)."""
    keep = [c for c in ("attendance_feature_id", "payroll_item_id", "employee_id", "month",
                        "job_level_att", "job_level_pay", "department_id_att") if c in merged.columns]
    out = merged[keep].copy()
    out["split"] = split_name
    return out


def build_monthly_pipeline() -> "Pipeline":
    """Monthly preprocessing: rules -> dept pay reference -> column transform."""
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import FunctionTransformer

    from .pipeline import build_column_transformer
    from .transformers import DeptMedianRatio

    rules_step = FunctionTransformer(add_monthly_features, validate=False)
    dept_step = DeptMedianRatio(
        pay_column="monthly_fixed_gross",
        dept_column="department_id",
        output_column="salary_vs_dept_median_monthly",
    )
    preprocessor = build_column_transformer(monthly_numeric_features(), monthly_categorical_features())
    return SkPipeline(
        [
            ("rules", rules_step),
            ("dept_salary_reference", dept_step),
            ("preprocessor", preprocessor),
        ]
    )


def monthly_metadata(
    *,
    output_columns: list[str],
    train_info: dict[str, object],
    dedup_log: dict[str, int],
    join_stats: dict[str, int],
    dept_references: dict[str, float],
    train_rows: int,
    test_rows: int,
) -> dict[str, object]:
    """Metadata saved beside the monthly artifact (aggregates only)."""
    from datetime import datetime, timezone

    from . import FEATURES_VERSION
    from .pipeline import CATEGORICAL_STEPS, NUMERIC_STEPS

    return {
        "feature_package_version": FEATURES_VERSION,
        "monthly_component_version": MONTHLY_COMPONENT_VERSION,
        "pipeline_artifact": "monthly_pipeline.joblib",
        "source_sheets": [ATTENDANCE_SHEET, PAYROLL_SHEET],
        "input_schema": merged_input_columns(),
        "key_definition": list(KEY_COLUMNS),
        "duplicate_rule": (
            "Payroll lines duplicating (employee_id, month) are dropped only when identical "
            "on every field except payroll_item_id; conflicting groups raise an error."
        ),
        "duplicate_log": dedup_log,
        "join_validation": join_stats,
        "payroll_only_key_treatment": "excluded from features (no attendance row to pair); counted in join_validation",
        "engineered_features": monthly_feature_definitions(),
        "output_columns": output_columns,
        "output_column_count": len(output_columns),
        "numeric_preprocessing": NUMERIC_STEPS,
        "categorical_preprocessing": CATEGORICAL_STEPS,
        "training": {
            "strategy": "chronological holdout by month",
            "rows_train": train_rows,
            "rows_test": test_rows,
            **train_info,
        },
        "department_salary_references": dept_references,
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "assumptions_and_limitations": [
            "Trailing 3-month means use each row's month plus the two preceding months only.",
            "Disputed job_level values are excluded from inputs; job_level_mismatch flags them for source-owner confirmation.",
            "No salary-revision dates exist in payroll history, so months-since-revision cannot be derived here.",
            "Holdout transform of partial history uses available months only (months_of_history records depth).",
        ],
    }


def monthly_feature_definitions() -> list[dict[str, str]]:
    return [
        {"name": "attendance_rate", "source": "attendance_monthly_features.attendance_rate",
         "formula": "validated as-is; attendance_invalid = 1 when outside [0, 1]", "unit": "proportion",
         "assumption": "Monthly rate definition taken as given; >1 invalid until confirmed (D09)."},
        {"name": "late_arrivals", "source": "attendance_monthly_features.late_arrivals",
         "formula": "monthly count used as-is", "unit": "count/month",
         "assumption": "Already employee-month grain; no double counting."},
        {"name": "late_arrivals_avg_3m", "source": "same, trailing window",
         "formula": "mean of current + 2 preceding months (min 1 present)", "unit": "count/month",
         "assumption": "Strictly as-of; future months never contribute."},
        {"name": "overtime_hours", "source": "attendance_monthly_features.overtime_hours",
         "formula": "monthly value used as-is", "unit": "hours/month",
         "assumption": "Genuine overtime field; total hours never substituted."},
        {"name": "overtime_hours_avg_3m", "source": "same, trailing window",
         "formula": "mean of current + 2 preceding months (min 1 present)", "unit": "hours/month",
         "assumption": "Strictly as-of."},
        {"name": "attendance_rate_avg_3m", "source": "attendance_rate, trailing window",
         "formula": "mean of current + 2 preceding months (min 1 present)", "unit": "proportion",
         "assumption": "Strictly as-of."},
        {"name": "monthly pay fields", "source": "payroll_monthly_features (fixed gross, payable/LOP days, overtime paid/pay, variable pay, deductions, gross/net)",
         "formula": "monthly values used as-is", "unit": "as labelled (INR / days / hours)",
         "assumption": "One payroll line per employee-month after dedup rule."},
        {"name": "salary_vs_dept_median_monthly",
         "source": "monthly_fixed_gross, department_id (100% agreement across sources)",
         "formula": "monthly_fixed_gross / department median fitted on training months only; unseen departments use the global median",
         "unit": "ratio", "assumption": "Frozen references; no holdout participation."},
        {"name": "job_level_mismatch", "source": "job_level_att vs job_level_pay",
         "formula": "1 when both present and unequal", "unit": "flag",
         "assumption": "Disputed levels excluded from inputs until source-owner confirmation; both values preserved in alignment output."},
    ]
