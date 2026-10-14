"""Small, testable helpers used by the Task 1 audit script.

These functions summarize data only. They do not alter the source dataframe.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per column with missing counts and percentages."""
    n_rows = len(df)
    missing = df.isna().sum()
    result = pd.DataFrame(
        {
            "column": df.columns.astype(str),
            "dtype": [str(dtype) for dtype in df.dtypes],
            "row_count": n_rows,
            "missing_count": missing.to_numpy(),
            "missing_pct": (missing.to_numpy() / n_rows * 100) if n_rows else 0.0,
            "non_missing_count": (n_rows - missing).to_numpy(),
            "unique_non_missing": [int(df[c].nunique(dropna=True)) for c in df.columns],
            "audit_decision": "",
        }
    )
    return result.sort_values(["missing_pct", "column"], ascending=[False, True]).reset_index(drop=True)


def outlier_comparison(series: pd.Series, column_name: str) -> dict[str, object]:
    """Compare IQR fences and absolute z-score > 3 for numeric observations.

    Handles missing values (dropped) and zero-variance columns (no z-score
    flags) safely. Returns quartiles and IQR alongside fences so callers can
    report them without recomputing.
    """
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    result: dict[str, object] = {
        "column": column_name,
        "non_missing_numeric_count": int(values.size),
        "q1": np.nan,
        "median": np.nan,
        "q3": np.nan,
        "iqr": np.nan,
        "iqr_lower_fence": np.nan,
        "iqr_upper_fence": np.nan,
        "iqr_flag_count": 0,
        "iqr_flag_pct": 0.0,
        "zscore_threshold": 3.0,
        "zscore_flag_count": 0,
        "zscore_flag_pct": 0.0,
        "mean": float(values.mean()) if values.size else np.nan,
        "std_sample": float(values.std(ddof=1)) if values.size > 1 else np.nan,
        "min": float(values.min()) if values.size else np.nan,
        "max": float(values.max()) if values.size else np.nan,
    }
    if values.empty:
        return result

    q1, median, q3 = float(values.quantile(0.25)), float(values.median()), float(values.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_mask = (values < lower) | (values > upper)
    result["q1"] = q1
    result["median"] = median
    result["q3"] = q3
    result["iqr"] = float(iqr)
    result["iqr_lower_fence"] = float(lower)
    result["iqr_upper_fence"] = float(upper)
    result["iqr_flag_count"] = int(iqr_mask.sum())
    result["iqr_flag_pct"] = float(iqr_mask.mean() * 100)

    std = values.std(ddof=0)
    if np.isfinite(std) and std > 0:
        z_mask = ((values - values.mean()).abs() / std) > 3.0
        result["zscore_flag_count"] = int(z_mask.sum())
        result["zscore_flag_pct"] = float(z_mask.mean() * 100)
    return result


def class_distribution(series: pd.Series, positive_values: Iterable[str] = ()) -> tuple[pd.DataFrame, float | None]:
    """Return target counts/rates and positive-class rate when identifiable.

    Positive matching is case-insensitive after string conversion. If none of the
    declared positive values occur, positive_rate is None rather than guessed.
    """
    display = series.astype("string").fillna("<MISSING>")
    counts = display.value_counts(dropna=False)
    total = int(counts.sum())
    table = pd.DataFrame(
        {
            "class_value": counts.index.astype(str),
            "count": counts.to_numpy(dtype=int),
            "pct": (counts.to_numpy(dtype=float) / total * 100) if total else np.array([], dtype=float),
        }
    )
    normalized = {str(v).strip().casefold() for v in positive_values}
    observed = series.dropna().astype("string")
    matched = observed.astype(str).str.strip().str.casefold().isin(normalized) if normalized else pd.Series(False, index=observed.index)
    # The positive-class rate uses labelled rows only; missing labels remain visible
    # as a separate class in the distribution table, but do not enter this denominator.
    positive_rate = float(matched.mean() * 100) if normalized and bool(matched.any()) and len(observed) else None
    return table, positive_rate


def candidate_columns(existing_columns: Iterable[object], candidates: Iterable[str] | None) -> list[str]:
    """Return configured candidate columns that exist, preserving candidate order.

    Returns [] safely when candidates is None/empty or nothing matches, so a
    missing or ambiguous configuration never raises.
    """
    if not candidates:
        return []
    existing = {str(column).casefold(): str(column) for column in existing_columns}
    return [existing[c.casefold()] for c in candidates if c.casefold() in existing]


def infer_column_role(column: str, dtype_str: str, config: dict | None = None) -> str:
    """Provisional single-label role used for the data dictionary.

    Labels are provisional and must be confirmed with domain owners; an
    unfamiliar column is never silently assigned a business meaning.
    """
    cfg = config or {}
    lower = str(column).casefold()
    targets = {str(c).casefold() for c in (cfg.get("target_column_candidates") or [])}
    leakage = {str(c).casefold() for c in (cfg.get("leakage_candidate_columns") or [])}
    ids = {str(c).casefold() for c in (cfg.get("id_candidate_columns") or [])}
    if lower in targets:
        return "target (candidate — confirm)"
    if lower in leakage:
        return "potentially post-outcome (confirm prediction-time availability)"
    if lower in ids:
        return "identifier (exclude from features; privacy-sensitive)"
    dtype_lower = str(dtype_str).lower()
    if "datetime" in dtype_lower or "date" in dtype_lower:
        return "date/time (confirm role)"
    if "int" in dtype_lower or "float" in dtype_lower:
        return "numeric feature (provisional)"
    return "categorical/text feature (provisional)"


def build_column_profile(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """One row per column: types, counts, provisional role, confirmation flag."""
    cfg = config or {}
    rows = []
    n_rows = len(df)
    for col in df.columns:
        name = str(col)
        dtype_str = str(df[col].dtype)
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "column": name,
                "dtype": dtype_str,
                "role_provisional": infer_column_role(name, dtype_str, cfg),
                "row_count": n_rows,
                "non_missing_count": n_rows - missing,
                "missing_count": missing,
                "missing_pct": (missing / n_rows * 100) if n_rows else 0.0,
                "unique_non_missing": int(df[col].nunique(dropna=True)),
                "needs_domain_confirmation": "yes"
                if "provisional" in infer_column_role(name, dtype_str, cfg) or "confirm" in infer_column_role(name, dtype_str, cfg).lower()
                else "review",
                "definition_in_own_words": "",
            }
        )
    return pd.DataFrame(rows)


def duplicate_summary(
    df: pd.DataFrame,
    id_columns: Iterable[str] | None,
    exclude_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Exact-duplicate plus per-identifier duplicate counts (no row values).

    When ``exclude_columns`` is given (e.g. a technical ``row_id``), an extra
    business-record row counts rows duplicated across all remaining columns,
    distinguishing full-row duplication from business-record duplication.
    """
    ids = [c for c in (id_columns or []) if c in df.columns]
    rows = [{"check": "exact_duplicate_rows", "column": "<all columns>", "count": int(df.duplicated().sum())}]
    for col in ids:
        rows.append(
            {
                "check": "duplicate_non_missing_identifier_values",
                "column": col,
                "count": int(df[col].dropna().duplicated().sum()),
            }
        )
    if exclude_columns:
        biz = business_record_duplicates(df, exclude_columns=exclude_columns)
        rows.append(
            {
                "check": "duplicate_business_records_excl_" + "+".join(sorted(str(c) for c in exclude_columns)),
                "column": "<all columns except " + ", ".join(sorted(str(c) for c in exclude_columns)) + ">",
                "count": int(biz["extra_duplicate_rows"]),
            }
        )
    return pd.DataFrame(rows)


def business_record_duplicates(df: pd.DataFrame, exclude_columns: Iterable[str] | None = None) -> dict[str, object]:
    """Count rows duplicated across all columns except excluded technical keys.

    Returns aggregates only (no row values or identifiers): the number of
    compared columns, extra duplicate rows beyond the first per group, total
    rows sitting in duplicate groups, and the number of duplicate groups.
    """
    excl = {str(c).casefold() for c in (exclude_columns or [])}
    subset = [c for c in df.columns if str(c).casefold() not in excl]
    result: dict[str, object] = {
        "excluded_columns": sorted(excl),
        "compared_columns": len(subset),
        "extra_duplicate_rows": 0,
        "rows_in_duplicate_groups": 0,
        "duplicate_groups": 0,
    }
    if not subset or df.empty:
        return result
    try:
        extra = df.duplicated(subset=subset)
        in_groups = df.duplicated(subset=subset, keep=False)
    except (TypeError, ValueError):
        return result
    extra_count = int(extra.sum())
    grouped_count = int(in_groups.sum())
    result["extra_duplicate_rows"] = extra_count
    result["rows_in_duplicate_groups"] = grouped_count
    result["duplicate_groups"] = grouped_count - extra_count
    return result


def pay_comp_consistency(
    df: pd.DataFrame,
    pay_col: str = "annual_fixed_pay",
    ratio_col: str = "comp_ratio",
    group_col: str = "job_level",
    threshold: float = 3.0,
) -> dict[str, object]:
    """Flag rows whose pay-implied reference exceeds ``threshold`` x group median.

    The implied reference is ``pay / comp_ratio``. When ``comp_ratio`` means
    pay relative to a band, the reference should cluster within each group, so
    a large deviation suggests a keying typo (never proof on its own). Returns
    aggregates only: flagged count/pct, the threshold used, groups compared,
    the highest ratio among unflagged rows, and a ``skipped`` reason when the
    required columns are absent.
    """
    result: dict[str, object] = {
        "pay_column": pay_col,
        "ratio_column": ratio_col,
        "group_column": group_col,
        "threshold": float(threshold),
        "valid_rows": 0,
        "groups_compared": 0,
        "flagged_count": 0,
        "flagged_pct": 0.0,
        "max_unflagged_ratio": float("nan"),
        "skipped": "",
    }
    missing = [c for c in (pay_col, ratio_col, group_col) if c not in df.columns]
    if missing:
        result["skipped"] = f"required column(s) absent: {', '.join(missing)}"
        return result
    if df.empty:
        result["skipped"] = "empty dataframe"
        return result
    pay = pd.to_numeric(df[pay_col], errors="coerce")
    ratio = pd.to_numeric(df[ratio_col], errors="coerce")
    valid = pay.notna() & ratio.notna() & (ratio != 0)
    result["valid_rows"] = int(valid.sum())
    if not valid.any():
        result["skipped"] = "no valid pay/ratio pairs"
        return result
    ref = (pay / ratio).where(valid)
    try:
        medians = ref.groupby(df[group_col]).transform("median")
    except (TypeError, ValueError) as exc:
        result["skipped"] = f"grouping failed: {type(exc).__name__}"
        return result
    result["groups_compared"] = int(df[group_col][valid].nunique(dropna=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (ref / medians).where(valid & medians.notna() & (medians != 0))
    flagged = rel > threshold
    result["flagged_count"] = int(flagged.sum())
    result["flagged_pct"] = float(flagged.sum() / valid.sum() * 100)
    unflagged = rel[valid & ~flagged.fillna(False)]
    result["max_unflagged_ratio"] = float(unflagged.max()) if len(unflagged) else float("nan")
    return result


# --- Shared rule constants (single source of truth for script + details) ---
HOURS_PLAUSIBLE_RANGE = (6.0, 12.0)
PAID_LEAVE_HIGH_DAYS = 30
PAY_COMP_THRESHOLD = 3.0
LOCATION_VARIANTS = ["BENGALURU", "Bangalore", "bengaluru", "chennai", "delhi ncr", "Delhi-NCR", "Madras"]

# Plausible business ranges used to separate statistical flags from rule
# violations: column -> (low, high) inclusive bounds, None means open-ended.
BUSINESS_RANGES: dict[str, tuple[float | None, float | None]] = {
    "annual_fixed_pay": (0.0, None),
    "avg_hours_worked_3m": HOURS_PLAUSIBLE_RANGE,
    "tenure_months": (0.0, None),
}


def _row_ref(df: pd.DataFrame) -> pd.Series:
    """Technical source-row reference (row_id when present, else the index)."""
    if "row_id" in df.columns:
        return df["row_id"]
    return pd.Series(df.index, index=df.index, name="row_id")


def outlier_details(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """One row per observation flagged by IQR and/or |z| > 3 (aggregates + row refs).

    Uses the technical ``row_id`` for traceability, never employee identifiers.
    Includes every row flagged by *either* method with separate Boolean flags.
    Statistical flags are investigation candidates, not confirmed errors.
    """
    records: list[dict[str, object]] = []
    refs = _row_ref(df)
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").astype(float)
        valid = values.dropna()
        if valid.empty:
            continue
        q1, q3 = float(valid.quantile(0.25)), float(valid.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mean, std = float(valid.mean()), float(valid.std(ddof=0))
        low, high = BUSINESS_RANGES.get(str(column), (None, None))
        for idx, value in valid.items():
            iqr_flag = bool(value < lower or value > upper)
            z = (value - mean) / std if std and np.isfinite(std) and std > 0 else float("nan")
            z_flag = bool(np.isfinite(z) and abs(z) > 3.0)
            if not (iqr_flag or z_flag):
                continue
            if iqr_flag:
                direction = "below lower IQR fence" if value < lower else "above upper IQR fence"
            else:
                direction = "above mean (z-only flag)" if z > 0 else "below mean (z-only flag)"
            violates = bool((low is not None and value < low) or (high is not None and value > high))
            if violates:
                status = "suspected data error"
                explanation = f"Value {value:,.2f} is outside the plausible business range and was statistically flagged."
                action = "Verify against the source record; correct or quarantine for Task 2."
            else:
                status = "statistical outlier — investigate"
                explanation = "Statistically unusual but inside plausible ranges; may be a legitimate extreme."
                action = "Retain; handle in Task 2 with robust scaling, not deletion."
            records.append(
                {
                    "row_ref": refs.loc[idx],
                    "column": str(column),
                    "value": float(value),
                    "q1": q1,
                    "q3": q3,
                    "iqr": float(iqr),
                    "iqr_lower_fence": float(lower),
                    "iqr_upper_fence": float(upper),
                    "iqr_flagged": iqr_flag,
                    "z_score": float(z) if np.isfinite(z) else float("nan"),
                    "zscore_flagged": z_flag,
                    "direction": direction,
                    "business_range_violation": violates,
                    "explanation": explanation,
                    "recommended_action": action,
                    "finding_status": status,
                }
            )
    cols = ["row_ref", "column", "value", "q1", "q3", "iqr", "iqr_lower_fence", "iqr_upper_fence",
            "iqr_flagged", "z_score", "zscore_flagged", "direction", "business_range_violation",
            "explanation", "recommended_action", "finding_status"]
    return pd.DataFrame(records, columns=cols)


def issue_detail_records(df: pd.DataFrame) -> pd.DataFrame:
    """Record-level data-quality issues needing investigation (row_id refs only).

    Mirrors the aggregate ``invalid value`` checks with one row per affected
    record. Never includes employee identifiers.
    """
    records: list[dict[str, object]] = []
    refs = _row_ref(df)

    def add(mask: pd.Series, column: str, category: str, rule: str, reason: str,
            evidence: str, action: str, status: str) -> None:
        values = pd.to_numeric(df[column], errors="coerce") if column in df.columns else df[column]
        for idx in df.index[mask.fillna(False)]:
            raw = df.loc[idx, column]
            records.append(
                {
                    "row_ref": refs.loc[idx],
                    "column": str(column),
                    "value": raw if pd.notna(raw) else "",
                    "issue_category": category,
                    "rule": rule,
                    "reason": reason,
                    "evidence": evidence,
                    "recommended_action": action,
                    "status": status,
                }
            )

    n = len(df)
    if "tenure_months" in df.columns:
        v = pd.to_numeric(df["tenure_months"], errors="coerce")
        add(v < 0, "tenure_months", "negative tenure", "tenure_months < 0 months",
            "Tenure below zero is impossible.", f"{int((v < 0).sum())} of {n} rows affected.",
            "Verify against HR record; correct or quarantine.", "confirmed invalid")
    if "total_experience_years" in df.columns:
        v = pd.to_numeric(df["total_experience_years"], errors="coerce")
        add(v < 0, "total_experience_years", "negative experience", "total_experience_years < 0 years",
            "Experience below zero is impossible; observed magnitudes are small.",
            f"{int((v < 0).sum())} of {n} rows affected.",
            "Investigate source; flag for Task 2.", "suspicious")
    if "wfh_share_3m" in df.columns:
        v = pd.to_numeric(df["wfh_share_3m"], errors="coerce")
        add(v < 0, "wfh_share_3m", "wfh share below 0", "wfh_share_3m < 0",
            "A share cannot be negative; the sizeable affected mass suggests an encoding issue.",
            f"{int((v < 0).sum())} of {n} rows affected.",
            "Confirm definition; do not use raw until resolved.", "suspicious — requires confirmation")
        add(v > 1, "wfh_share_3m", "wfh share above 1", "wfh_share_3m > 1",
            "A share cannot exceed 1.", f"{int((v > 1).sum())} of {n} rows affected.",
            "Confirm definition; flag affected rows.", "suspicious — requires confirmation")
    if "attendance_rate_3m" in df.columns:
        v = pd.to_numeric(df["attendance_rate_3m"], errors="coerce")
        add(v > 1, "attendance_rate_3m", "attendance above 1", "attendance_rate_3m > 1.0",
            "A rate above 1.0 is suspicious unless the definition allows it.",
            f"{int((v > 1).sum())} of {n} rows affected.",
            "Confirm definition; flag affected rows.", "suspicious — requires confirmation")
    if "avg_hours_worked_3m" in df.columns:
        v = pd.to_numeric(df["avg_hours_worked_3m"], errors="coerce")
        lo, hi = HOURS_PLAUSIBLE_RANGE
        add((v < lo) | (v > hi), "avg_hours_worked_3m", "implausible work hours",
            f"avg_hours_worked_3m outside {lo:g}-{hi:g} hours/day (illustrative range)",
            "Far from the ~8.7-hour cluster; the low extreme is business-implausible.",
            f"{int(((v < lo) | (v > hi)).sum())} of {n} rows affected.",
            "Investigate the extreme record(s); retain the rest.", "suspicious")
    pay_flags = pay_comp_consistency(df)
    if not pay_flags.get("skipped") and {"annual_fixed_pay", "comp_ratio", "job_level"} <= set(df.columns):
        pay = pd.to_numeric(df["annual_fixed_pay"], errors="coerce")
        ratio = pd.to_numeric(df["comp_ratio"], errors="coerce")
        valid = pay.notna() & ratio.notna() & (ratio != 0)
        ref = (pay / ratio).where(valid)
        medians = ref.groupby(df["job_level"]).transform("median")
        rel = (ref / medians).where(valid & medians.notna() & (medians != 0))
        thr = float(pay_flags.get("threshold", PAY_COMP_THRESHOLD))
        add(rel > thr, "annual_fixed_pay", "salary/comp-ratio inconsistency",
            f"pay-implied reference (pay/comp_ratio) > {thr:g}x job-level median (provisional interpretation)",
            "Pattern fits decimal-place keying typos; comp_ratio definition is provisional.",
            f"{int((rel > thr).fillna(False).sum())} rows exceed {thr:g}x; highest other ratio "
            f"{pay_flags['max_unflagged_ratio']:.2f}x.",
            "Verify against payroll; never auto-divide values.", "suspicious — requires confirmation")
    if "paid_leave_days_6m" in df.columns:
        v = pd.to_numeric(df["paid_leave_days_6m"], errors="coerce")
        add(v > PAID_LEAVE_HIGH_DAYS, "paid_leave_days_6m", "very high paid leave",
            f"paid_leave_days_6m > {PAID_LEAVE_HIGH_DAYS} days in 6 months",
            "Very high leave balances are rare.", f"{int((v > PAID_LEAVE_HIGH_DAYS).sum())} of {n} rows affected.",
            "Confirm with HR; retain pending confirmation.", "suspicious — requires confirmation")
    if "work_location" in df.columns:
        mask = df["work_location"].isin(LOCATION_VARIANTS)
        for idx in df.index[mask.fillna(False)]:
            records.append(
                {
                    "row_ref": refs.loc[idx],
                    "column": "work_location",
                    "value": str(df.loc[idx, "work_location"]),
                    "issue_category": "non-canonical location spelling",
                    "rule": "value is a known case/punctuation/alternate-name variant",
                    "reason": "Fragments one city into several spellings.",
                    "evidence": f"{int(mask.sum())} rows use 7 known variants.",
                    "recommended_action": "Normalize with the HR-approved mapping in Task 2.",
                    "status": "requires confirmation (canonical list)",
                }
            )
    if "resignation_date" in df.columns and "observation_date" in df.columns:
        obs = pd.to_datetime(df["observation_date"], errors="coerce")
        res = pd.to_datetime(df["resignation_date"], errors="coerce")
        mask = res.notna() & obs.notna() & (res < obs)
        for idx in df.index[mask.fillna(False)]:
            records.append(
                {
                    "row_ref": refs.loc[idx],
                    "column": "resignation_date",
                    "value": str(df.loc[idx, "resignation_date"]),
                    "issue_category": "resignation before observation",
                    "rule": "resignation_date < observation_date",
                    "reason": "Contradicts a clean prediction timeline.",
                    "evidence": f"{int(mask.sum())} dated resignations affected.",
                    "recommended_action": "Clarify dating semantics before any date-derived feature.",
                    "status": "requires confirmation",
                }
            )
    cols = ["row_ref", "column", "value", "issue_category", "rule", "reason",
            "evidence", "recommended_action", "status"]
    return pd.DataFrame(records, columns=cols)


def build_leakage_register(
    columns_in_order: Iterable[object],
    config: dict | None = None,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Build one leakage-review row for EVERY column (completeness guaranteed).

    Each row documents the suspected mechanism, a risk level, whether the
    column should be excluded at the intended prediction time, the reason,
    and what confirmation is still needed. Target leakage is distinguished
    from identifier/privacy concerns and from ordinary feature-quality notes.
    """
    cfg = config or {}
    leakage_set = {str(c).casefold() for c in (cfg.get("leakage_candidate_columns") or [])}
    id_set = {str(c).casefold() for c in (cfg.get("id_candidate_columns") or [])}
    target_set = {str(c).casefold() for c in (cfg.get("target_column_candidates") or [])}
    if target_column is not None:
        target_norm = str(target_column).casefold()
    else:
        target_norm = ""

    # Tailored notes for the known attrition schema; unknown columns fall back
    # to a generic "no evident risk, confirm availability" assessment so the
    # register stays complete for synthetic fixtures and schema changes.
    known: dict[str, tuple[str, str, str, str, str]] = {
        # column -> (mechanism, risk_level, exclude, reason, confirmation_needed)
        "exit_date": (
            "Post-outcome date recorded only after leaving",
            "Confirmed risk",
            "Exclude",
            "Present only when the outcome occurred; unknowable at prediction time.",
            "Confirm prediction timestamp precedes any exit recording.",
        ),
        "resignation_date": (
            "Post-outcome date recorded only after resigning",
            "Confirmed risk",
            "Exclude",
            "Present only for voluntary leavers; directly reveals the outcome.",
            "Confirm prediction timestamp precedes resignation.",
        ),
        "exit_reason": (
            "Free-text/category recorded at or after exit",
            "Confirmed risk",
            "Exclude",
            "Recorded only when someone left; missing otherwise.",
            "Confirm no pre-exit proxy with the same name is intended.",
        ),
        "exit_interview_done": (
            "Exit-interview flag recorded after exit",
            "Confirmed risk",
            "Exclude",
            "Only defined for leavers; missing for stayers.",
            "Confirm interview timing relative to prediction date.",
        ),
        "fnf_settlement_amount": (
            "Final settlement amount paid after exit",
            "Confirmed risk",
            "Exclude",
            "Full-and-final payout exists only post-exit.",
            "Confirm payroll timing; never use as input.",
        ),
        "final_settlement": (
            "Final settlement field paid after exit",
            "Confirmed risk",
            "Exclude",
            "Post-outcome payroll artifact by name.",
            "Confirm timing if this column ever appears.",
        ),
        "final_settlement_amount": (
            "Final settlement amount paid after exit",
            "Confirmed risk",
            "Exclude",
            "Post-outcome payroll artifact by name.",
            "Confirm timing if this column ever appears.",
        ),
    }

    rows = []
    for col in columns_in_order:
        name = str(col)
        lower = name.casefold()
        if target_norm and lower == target_norm:
            rows.append(
                {
                    "column": name,
                    "suspected_mechanism": "Prediction target itself",
                    "risk_level": "Target — not a feature",
                    "exclude_from_model": "Exclude as input (this is the label)",
                    "reason": "The label must never be used as a predictive input.",
                    "confirmation_needed": "Confirm positive-label semantics from the source spec.",
                }
            )
        elif lower in known:
            mech, risk, excl, reason, conf = known[lower]
            rows.append(
                {
                    "column": name,
                    "suspected_mechanism": mech,
                    "risk_level": risk,
                    "exclude_from_model": excl,
                    "reason": reason,
                    "confirmation_needed": conf,
                }
            )
        elif lower in leakage_set:
            rows.append(
                {
                    "column": name,
                    "suspected_mechanism": "Configured leakage candidate by name",
                    "risk_level": "Potential risk",
                    "exclude_from_model": "Exclude pending confirmation",
                    "reason": "Listed in config as requiring prediction-time review.",
                    "confirmation_needed": "Verify availability at the intended prediction time.",
                }
            )
        elif lower in target_set:
            rows.append(
                {
                    "column": name,
                    "suspected_mechanism": "Alternative outcome label",
                    "risk_level": "Confirmed risk if modelling the primary target",
                    "exclude_from_model": "Exclude",
                    "reason": "A second label for the same event; using it reveals the outcome.",
                    "confirmation_needed": "Decide which label is the Task 3 target; exclude the other.",
                }
            )
        elif lower in id_set:
            rows.append(
                {
                    "column": name,
                    "suspected_mechanism": "Row/employee identifier (privacy, not outcome leakage)",
                    "risk_level": "Identifier — no outcome leakage, exclude for generalization/privacy",
                    "exclude_from_model": "Exclude",
                    "reason": "Identifiers do not generalize and expose row-level identity.",
                    "confirmation_needed": "Confirm join-key needs for Task 2; otherwise drop from features.",
                }
            )
        elif lower in ("observation_date",):
            rows.append(
                {
                    "column": name,
                    "suspected_mechanism": "Prediction-time reference date",
                    "risk_level": "Requires confirmation",
                    "exclude_from_model": "Use as reference only, not as a raw model input",
                    "reason": "Defines the prediction point; needed for splits and prior-window checks.",
                    "confirmation_needed": "Confirm this is the date at which predictions would be made.",
                }
            )
        elif lower in ("source",):
            rows.append(
                {
                    "column": name,
                    "suspected_mechanism": "Provenance/batch indicator",
                    "risk_level": "No evident leakage risk",
                    "exclude_from_model": "Decide after confirmation (likely exclude or use for split analysis only)",
                    "reason": "Describes data origin, not the outcome; may encode time/batch shift.",
                    "confirmation_needed": "Confirm whether source mixes time periods; do not let it proxy the label.",
                }
            )
        elif lower in ("dept_attrition_rate_prior_12m",):
            rows.append(
                {
                    "column": name,
                    "suspected_mechanism": "Department aggregate labelled 'prior'",
                    "risk_level": "Potential risk — requires confirmation",
                    "exclude_from_model": "Retain pending confirmation",
                    "reason": "Name claims the window ends at observation time; if true it is legitimate.",
                    "confirmation_needed": "Verify the 12-month window strictly precedes observation_date.",
                }
            )
        else:
            rows.append(
                {
                    "column": name,
                    "suspected_mechanism": "No post-outcome mechanism evident from name/pattern",
                    "risk_level": "No evident leakage risk",
                    "exclude_from_model": "Retain pending confirmation",
                    "reason": "Appears to be a pre-prediction attribute; confirm availability at prediction time.",
                    "confirmation_needed": "Confirm measurement timing relative to observation_date.",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "column",
            "suspected_mechanism",
            "risk_level",
            "exclude_from_model",
            "reason",
            "confirmation_needed",
        ],
    )


VALIDATION_COLUMNS = [
    "issue_category",
    "expected_planted_count",
    "detected_count",
    "matched_count",
    "missed_count",
    "additional_flagged_count",
    "validation_status",
    "explanation",
]

VALIDATION_STATUSES = {"detected", "detected-after-correction", "partially-detected", "missed", "unconfirmable"}


def summarize_planted_validation(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Build the post-audit validation table with a fixed, aggregate-only schema.

    Each input row needs ``issue_category``, counts, a ``validation_status``
    from ``VALIDATION_STATUSES``, and an ``explanation``. Raises ``ValueError``
    on unknown statuses or missing fields so bad validation output fails loudly.
    """
    records = list(rows)
    frame = pd.DataFrame(records)
    missing_cols = [c for c in VALIDATION_COLUMNS if c not in frame.columns]
    if missing_cols:
        raise ValueError(f"validation rows lack column(s): {', '.join(missing_cols)}")
    unknown = set(frame["validation_status"].astype(str)) - VALIDATION_STATUSES
    if unknown:
        raise ValueError(f"unknown validation_status value(s): {sorted(unknown)}")
    for col in [
        "expected_planted_count",
        "detected_count",
        "matched_count",
        "missed_count",
        "additional_flagged_count",
    ]:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(int)
    return frame[VALIDATION_COLUMNS].reset_index(drop=True)


def render_validation_section(validation: pd.DataFrame | None) -> str:
    """Render the report appendix for post-audit validation (no row-level data)."""
    header = "## Post-Audit Validation Against Planted Issues\n"
    scope_note = (
        "\nThe independent audit above was completed before the planted answer key "
        "(`answer_key_planted_dirt`) was read. This section compares that independent "
        "assessment against the planted issues afterwards. Original findings are unchanged; "
        "corrections made because of validation are listed explicitly below.\n"
    )
    if validation is None or validation.empty:
        return (
            header + scope_note
            + "\nPost-audit validation has not been run. Run "
            "`python scripts/validate_planted_issues.py` (reads only `attrition_dataset` "
            "and the authorized `answer_key_planted_dirt` sheet) to generate "
            "`planted_issue_validation.csv`; re-running the audit then embeds the results here.\n"
        )
    lines = [
        "| Issue category | Expected | Detected | Matched | Missed | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in validation.itertuples():
        lines.append(
            f"| {r.issue_category} | {int(r.expected_planted_count)} | {int(r.detected_count)} | "
            f"{int(r.matched_count)} | {int(r.missed_count)} | {r.validation_status} |"
        )
    detail = "\n".join(f"- **{r.issue_category}**: {r.explanation}" for r in validation.itertuples())
    return (
        header + scope_note
        + "\n" + "\n".join(lines)
        + "\n\nFull aggregate table: `planted_issue_validation.csv` (no employee-level data).\n"
        + "\n" + detail + "\n"
    )
def resolve_target_column(df: pd.DataFrame, config: dict | None = None) -> str | None:
    """First configured target candidate present in the dataframe, else None."""
    cfg = config or {}
    found = candidate_columns(df.columns, cfg.get("target_column_candidates"))
    return found[0] if found else None


def output_paths(output_dir: Path) -> dict[str, Path]:
    """Canonical artifact locations shared by the script and tests."""
    output_dir = Path(output_dir)
    return {
        "data_dictionary": output_dir / "data_dictionary.csv",
        "column_profile": output_dir / "column_profile.csv",
        "missing_values": output_dir / "missing_values.csv",
        "missingness_alias": output_dir / "missingness.csv",
        "outlier_summary": output_dir / "outlier_summary.csv",
        "outlier_alias": output_dir / "outlier_comparison.csv",
        "target_distribution": output_dir / "target_distribution.csv",
        "leakage_register": output_dir / "leakage_register.csv",
        "leakage_alias": output_dir / "leakage_review.csv",
        "decision_register": output_dir / "decision_register.csv",
        "duplicate_review": output_dir / "duplicate_review.csv",
        "invalid_values": output_dir / "invalid_values.csv",
        "outlier_details": output_dir / "outlier_details.csv",
        "issue_details": output_dir / "data_quality_issue_details.csv",
        "planted_validation": output_dir / "planted_issue_validation.csv",
        "final_report": output_dir / "TASK1_AUDIT_REPORT.md",
        "audit_summary": output_dir / "audit_summary.md",
        "figures_dir": output_dir / "figures",
    }
