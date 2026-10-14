"""Run a non-destructive Task 1 data audit against one configured Excel sheet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _entry in (PROJECT_ROOT / "task1_data_audit" / "src", PROJECT_ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from audit.audit_utils import (  # noqa: E402
    HOURS_PLAUSIBLE_RANGE,
    LOCATION_VARIANTS,
    PAID_LEAVE_HIGH_DAYS,
    PAY_COMP_THRESHOLD,
    build_column_profile,
    build_leakage_register,
    business_record_duplicates,
    candidate_columns,
    class_distribution,
    duplicate_summary,
    issue_detail_records,
    missingness_summary,
    outlier_comparison,
    outlier_details,
    pay_comp_consistency,
    render_validation_section,
    resolve_target_column,
)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Task 1 config must be a JSON object.")
    return cfg


def fmt_number(value: object, decimals: int = 2) -> str:
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def build_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate suspicious/invalid-value checks (counts only, no row data)."""
    rows: list[dict[str, object]] = []

    def add(column: str, check: str, count: int, pct_den: int, details: str, action: str) -> None:
        rows.append(
            {
                "column": column,
                "check": check,
                "suspicious_count": int(count),
                "suspicious_pct": float(count / pct_den * 100) if pct_den else 0.0,
                "details": details,
                "recommended_action": action,
            }
        )

    n = len(df)
    if "tenure_months" in df.columns:
        v = pd.to_numeric(df["tenure_months"], errors="coerce")
        add("tenure_months", "negative tenure", int((v < 0).sum()), n,
            "Tenure below 0 months is impossible; observed minima are negative single digits.",
            "Investigate source; correct against verified HR record or flag; never silently drop.")
    if "total_experience_years" in df.columns:
        v = pd.to_numeric(df["total_experience_years"], errors="coerce")
        add("total_experience_years", "negative experience", int((v < 0).sum()), n,
            "Experience below 0 years is impossible; observed negatives are small magnitudes near zero.",
            "Investigate; likely data-entry/clipping issue; flag for Task 2.")
    if "wfh_share_3m" in df.columns:
        v = pd.to_numeric(df["wfh_share_3m"], errors="coerce")
        add("wfh_share_3m", "share below 0", int((v < 0).sum()), n,
            "A share cannot be negative; sizeable negative mass suggests encoding or scaling error.",
            "Investigate source definition; do not use raw until resolved; consider clipping only after confirmation.")
        add("wfh_share_3m", "share above 1", int((v > 1).sum()), n,
            "A share cannot exceed 1; a small count exceeds 1.",
            "Investigate; same Task 2 hold as above.")
    if "attendance_rate_3m" in df.columns:
        v = pd.to_numeric(df["attendance_rate_3m"], errors="coerce")
        add("attendance_rate_3m", "rate above 1", int((v > 1).sum()), n,
            "Rate above 1.0 is suspicious unless overtime/definition allows it; very few rows.",
            "Confirm definition; flag affected rows for Task 2.")
    if "avg_hours_worked_3m" in df.columns:
        v = pd.to_numeric(df["avg_hours_worked_3m"], errors="coerce")
        lo, hi = HOURS_PLAUSIBLE_RANGE
        add("avg_hours_worked_3m", f"outside plausible {lo:g}-{hi:g} hours/day", int(((v < lo) | (v > hi)).sum()), n,
            "One extremely low value near 1.8 hours; remainder inside 6-12. Plausible range is illustrative.",
            "Investigate the extreme row(s) via source; retain others.")
    if "annual_fixed_pay" in df.columns:
        v = pd.to_numeric(df["annual_fixed_pay"], errors="coerce")
        add("annual_fixed_pay", "non-positive pay", int((v <= 0).sum()), n,
            "Non-positive pay would be impossible; none observed is a good sign.",
            "Retain; re-check after any new extract.")
    pay_check = pay_comp_consistency(df, threshold=PAY_COMP_THRESHOLD)
    if not pay_check.get("skipped"):
        flagged = int(pay_check["flagged_count"])
        max_ok = pay_check["max_unflagged_ratio"]
        max_ok_txt = fmt_number(max_ok, 2) if pd.notna(max_ok) else "N/A"
        add("annual_fixed_pay", f"pay/comp_ratio inconsistency (>{PAY_COMP_THRESHOLD:g}x job-level median reference)", flagged, n,
            f"{flagged} row(s) whose pay-implied reference (pay/comp_ratio) exceeds {PAY_COMP_THRESHOLD:g}x the job-level median "
            f"(highest non-flagged ratio is {max_ok_txt}x). Pattern fits decimal-place keying typos; comp_ratio definition is provisional.",
            "Investigate against verified payroll; correct from source or flag; never auto-divide values.")
    if "paid_leave_days_6m" in df.columns:
        v = pd.to_numeric(df["paid_leave_days_6m"], errors="coerce")
        add("paid_leave_days_6m", f"above {PAID_LEAVE_HIGH_DAYS} days in 6 months", int((v > PAID_LEAVE_HIGH_DAYS).sum()), n,
            "Very high leave balances are rare and need HR confirmation.",
            "Investigate; retain pending confirmation.")
    if "work_location" in df.columns:
        nunique = int(df["work_location"].nunique(dropna=True))
        # Observed non-canonical variants (case/city spellings) counted as rows affected.
        variants = list(LOCATION_VARIANTS)
        try:
            variant_rows = int(df["work_location"].isin(variants).sum())
        except Exception:
            variant_rows = 0
        add("work_location", "non-canonical location spellings", variant_rows, n,
            f"{variant_rows} rows use 7 non-canonical variants; {nunique} distinct spellings total "
            "(e.g. Bengaluru/Bangalore/bengaluru; Chennai/chennai; Delhi variants; Madras). Canonical list needs HR confirmation.",
            "Standardize spelling in Task 2 with a recorded mapping; confirm canonical list with HR.")
    if "resignation_date" in df.columns and "observation_date" in df.columns:
        tmp = df[["observation_date", "resignation_date"]].dropna(subset=["resignation_date"])
        try:
            obs = pd.to_datetime(tmp["observation_date"], errors="coerce")
            res = pd.to_datetime(tmp["resignation_date"], errors="coerce")
            cnt = int(((res < obs)).sum())
        except Exception:
            cnt = 0
        add("resignation_date", "resignation before observation_date", cnt, int(tmp.shape[0]),
            "A resignation dated before the observation point contradicts a clean prediction timeline.",
            "Investigate dating semantics before using any date-derived feature.")
    return pd.DataFrame(rows, columns=["column", "check", "suspicious_count", "suspicious_pct", "details", "recommended_action"])


def build_decision_register(
    missing: pd.DataFrame,
    outliers: pd.DataFrame,
    target_col: str | None,
    duplicate_df: pd.DataFrame,
) -> pd.DataFrame:
    """Explicit Task 1 -> Task 2 rules (recommendations, not silent fixes)."""
    rows = [
        {
            "decision_id": "D01",
            "status": "Binding Task 2 rule",
            "area": "missingness",
            "decision": "Preserve structural missingness for months_since_last_promotion and last_raise_pct; add missingness indicators in Task 2",
            "reason": "These fields are missing exactly when no promotion/raise occurred, so absence carries meaning.",
            "task2_rule": "Impute only for modelling convenience (e.g. large sentinel) AND keep is_never_promoted / is_never_raised flags; fit on train only.",
        },
        {
            "decision_id": "D02",
            "status": "Binding Task 2 rule",
            "area": "missingness",
            "decision": "Impute commute_distance_km, last_performance_rating, highest_education with train-fitted imputers; investigate engagement missingness",
            "reason": "Low-to-moderate missingness (0.6-7.6%) without structural pattern; engagement at 22% needs source check.",
            "task2_rule": "Median/mode imputation fitted on train; add missingness indicator for engagement; document rates.",
        },
        {
            "decision_id": "D03",
            "status": "Binding Task 2 rule",
            "area": "missingness",
            "decision": "Do not drop any column for missingness alone; never drop rows except for missing target",
            "reason": "Even 61% and 85% missing columns have business meaning (no-event vs post-event).",
            "task2_rule": "Keep columns through Task 2 design; drop rows only if the target label itself were missing.",
        },
        {
            "decision_id": "D04",
            "status": "Binding Task 2 rule",
            "area": "outliers",
            "decision": "Retain all statistical outliers through Task 1; flag and investigate before any Task 2 treatment",
            "reason": "IQR and z-score flag legitimate extremes in skewed pay/hours/tenure data.",
            "task2_rule": "No deletion/capping in audit; in Task 2 prefer robust scaling and flagged-indicator columns; cap only with documented HR limits.",
        },
        {
            "decision_id": "D05",
            "status": "Working target - Task 3 confirmation needed",
            "area": "target",
            "decision": f"Use {target_col or 'confirmed target'} as the primary Task 3 label after confirming semantics",
            "reason": "Binary 0/1 field with documented leaver meaning; class imbalance requires stratified handling.",
            "task2_rule": "Stratified splits; report precision/recall/AUC rather than accuracy alone.",
        },
        {
            "decision_id": "D06",
            "status": "Binding Task 2 rule (exclude)",
            "area": "leakage",
            "decision": "Exclude all post-outcome exit/settlement fields from model inputs",
            "reason": "These fields exist only after leaving and would not be known at prediction time.",
            "task2_rule": "Drop exit_date, resignation_date, exit_reason, exit_interview_done, fnf_settlement_amount before training.",
        },
        {
            "decision_id": "D07",
            "status": "Binding Task 2 rule (exclude)",
            "area": "leakage",
            "decision": "Exclude identifiers and the non-primary outcome label from features",
            "reason": "IDs harm generalization and privacy; the second label reveals the same event.",
            "task2_rule": "Drop row_id/employee_id and the alternative target column from X.",
        },
        {
            "decision_id": "D08",
            "status": "Binding Task 2 rule",
            "area": "duplicates",
            "decision": "Keep all rows in Task 1; resolve repeated employee identifiers in Task 2 with a documented key rule",
            "reason": "No full-row duplicates, but business-record duplicates exist once the technical row_id is excluded; repeated employee IDs need timeline-aware dedup, not deletion.",
            "task2_rule": "Define one-row-per-employee rule (e.g. latest observation; business key = all fields except row_id) before modelling; keep audit trail.",
        },
        {
            "decision_id": "D09",
            "status": "Provisional - HR confirmation needed",
            "area": "data-quality",
            "decision": "Standardize work_location spellings and validate impossible ranges in Task 2",
            "reason": "Case/city variants fragment categories; negatives and out-of-range shares are data errors until proven otherwise.",
            "task2_rule": "Recorded category mapping; range checks with quarantine/flag, not silent deletion.",
        },
        {
            "decision_id": "D10",
            "status": "Provisional - source confirmation needed",
            "area": "temporal",
            "decision": "Confirm observation/prediction timestamp and prior-window definitions before modelling",
            "reason": "Leakage decisions depend on what is known at prediction time.",
            "task2_rule": "Freeze prediction date; verify dept prior rate window ends at observation; time-based validation split.",
        },
    ]
    _ = (missing, outliers, duplicate_df)  # evidence inputs recorded in sibling CSVs
    return pd.DataFrame(rows, columns=["decision_id", "area", "decision", "reason", "task2_rule", "status"])


def plot_distribution_with_fences(values: pd.Series, column: str, path: Path, unit: str) -> None:
    """Histogram plus IQR fences and plausible-range context (safe on bad data)."""
    v = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if v.empty:
        return
    q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(v.to_numpy(), bins=40)
    axes[0].axvline(lower, linestyle="--", label=f"IQR lower {lower:,.2f}")
    axes[0].axvline(upper, linestyle="--", label=f"IQR upper {upper:,.2f}")
    axes[0].set_title(f"{column} distribution ({unit})")
    axes[0].set_xlabel(f"{column} ({unit})")
    axes[0].set_ylabel("Rows")
    axes[0].legend(fontsize=8)
    try:
        axes[1].boxplot(v.to_numpy(), orientation="vertical", tick_labels=[column])
    except TypeError:  # older matplotlib: `vert` + `labels`
        try:
            axes[1].boxplot(v.to_numpy(), vert=True, tick_labels=[column])
        except TypeError:
            axes[1].boxplot(v.to_numpy(), vert=True, labels=[column])
    axes[1].set_title(f"{column} boxplot with IQR fences")
    axes[1].set_ylabel(unit)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="task1_data_audit/configs/task1_config.json", help="Path to JSON config, relative to project root by default")
    parser.add_argument("--input", default=None, help="Override dataset workbook path")
    parser.add_argument("--sheet", default=None, help="Override source worksheet name")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 2
    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"ERROR: invalid config {config_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    input_value = args.input or config.get("dataset_file")
    if not input_value:
        print("ERROR: dataset_file is missing in config and no --input was given.", file=sys.stderr)
        return 2
    input_path = Path(input_value)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    sheet_name = args.sheet or config.get("sheet_name")
    if not sheet_name:
        print("ERROR: sheet_name is missing in config and no --sheet was given.", file=sys.stderr)
        return 2
    output_dir = Path(config.get("output_dir", "task1_data_audit/reports/generated"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    figures_dir = output_dir / "figures"

    if not input_path.exists():
        print("DATASET NOT FOUND")
        print(f"Expected workbook: {input_path}")
        print("Place your workbook in data/raw/ and use the configured filename, or pass --input.")
        return 2

    # Read only the configured source sheet. This code does not visit or follow workbook links.
    try:
        df = pd.read_excel(input_path, sheet_name=sheet_name)
    except ValueError as exc:
        print(f"ERROR: Could not read configured worksheet {sheet_name!r}: {exc}", file=sys.stderr)
        print("Confirm the sheet name in task1_data_audit/configs/task1_config.json. Other sheets are not inspected.", file=sys.stderr)
        return 2
    except Exception as exc:  # provide a helpful failure without hiding the actual reason
        print(f"ERROR: Workbook read failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = df.shape

    # --- Profiles (shared helpers; notebook imports the same functions) ---
    profile = build_column_profile(df, config)
    profile.to_csv(output_dir / "column_profile.csv", index=False)

    data_dict = missingness_summary(df)
    data_dict["role_provisional"] = [profile.set_index("column").loc[str(c), "role_provisional"] if str(c) in set(profile["column"]) else "" for c in data_dict["column"]]
    data_dict["definition_in_own_words"] = ""
    data_dict.to_csv(output_dir / "data_dictionary.csv", index=False)

    missing_all = data_dict[["column", "dtype", "role_provisional", "row_count", "missing_count", "missing_pct", "non_missing_count", "unique_non_missing", "audit_decision"]].copy()
    missing_all.to_csv(output_dir / "missing_values.csv", index=False)
    missing_nonzero = missing_all.loc[missing_all["missing_count"] > 0].copy()
    missing_nonzero.to_csv(output_dir / "missingness.csv", index=False)

    id_cols = candidate_columns(df.columns, config.get("id_candidate_columns", []))
    duplicates = duplicate_summary(df, id_cols, exclude_columns=["row_id"])
    duplicates.to_csv(output_dir / "duplicate_review.csv", index=False)
    duplicate_row_count = int(df.duplicated().sum())
    business_dups = business_record_duplicates(df, exclude_columns=["row_id"])

    configured_outliers = candidate_columns(df.columns, config.get("outlier_columns", []))
    outlier_records = []
    for col in configured_outliers:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.to_numeric(df[col], errors="coerce").notna().any():
            outlier_records.append(outlier_comparison(df[col], col))
    outliers = pd.DataFrame(outlier_records)
    if outliers.empty:
        outliers = pd.DataFrame(columns=["column", "non_missing_numeric_count", "q1", "median", "q3", "iqr",
                                         "iqr_lower_fence", "iqr_upper_fence", "iqr_flag_count", "iqr_flag_pct",
                                         "zscore_threshold", "zscore_flag_count", "zscore_flag_pct",
                                         "mean", "std_sample", "min", "max"])
    outliers.to_csv(output_dir / "outlier_summary.csv", index=False)
    outliers.to_csv(output_dir / "outlier_comparison.csv", index=False)

    target_col = resolve_target_column(df, config)
    target_table = pd.DataFrame(columns=["class_value", "count", "pct"])
    positive_rate: float | None = None
    if target_col:
        target_table, positive_rate = class_distribution(df[target_col], config.get("positive_target_values", []))
        target_table.to_csv(output_dir / "target_distribution.csv", index=False)
    else:
        pd.DataFrame([{"status": "No configured target candidate was found.", "action": "Set target_column_candidates in task1_data_audit/configs/task1_config.json after checking the source columns."}]).to_csv(output_dir / "target_distribution.csv", index=False)

    leakage_full = build_leakage_register(list(df.columns), config, target_column=target_col)
    leakage_full.to_csv(output_dir / "leakage_register.csv", index=False)
    leakage_candidates_present = candidate_columns(df.columns, config.get("leakage_candidate_columns", []))
    leakage_alias = leakage_full[leakage_full["column"].isin(leakage_candidates_present)].copy()
    if leakage_alias.empty:
        leakage_alias = pd.DataFrame(columns=list(leakage_full.columns))
    leakage_alias.to_csv(output_dir / "leakage_review.csv", index=False)

    invalids = build_invalid_values(df)
    invalids.to_csv(output_dir / "invalid_values.csv", index=False)

    outlier_detail_rows = outlier_details(df, configured_outliers)
    outlier_detail_rows.to_csv(output_dir / "outlier_details.csv", index=False)
    issue_details = issue_detail_records(df)
    issue_details.to_csv(output_dir / "data_quality_issue_details.csv", index=False)

    decisions = build_decision_register(missing_all, outliers, target_col, duplicates)
    decisions.to_csv(output_dir / "decision_register.csv", index=False)

    # --- Figures: never print raw row values or employee-level records. ---
    if n_cols:
        top_missing = data_dict.loc[data_dict["missing_count"] > 0].head(20).sort_values("missing_pct")
        if not top_missing.empty:
            fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(top_missing))))
            ax.barh(top_missing["column"], top_missing["missing_pct"])
            ax.set_xlabel("Missing values (%)")
            ax.set_title("Columns with the most missing values (top 20)")
            ax.set_xlim(0, 100)
            fig.tight_layout()
            fig.savefig(figures_dir / "missingness_top20.png", dpi=160)
            plt.close(fig)

    if target_col and not target_table.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(target_table["class_value"].astype(str), target_table["count"])
        ax.set_title(f"Target distribution: {target_col}")
        ax.set_xlabel("Class (1 = left within 12 months)")
        ax.set_ylabel("Rows")
        fig.tight_layout()
        fig.savefig(figures_dir / "target_distribution.png", dpi=160)
        plt.close(fig)

    if not outliers.empty and len(outliers):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(outliers))
        width = 0.36
        ax.bar(x - width / 2, outliers["iqr_flag_count"], width, label="IQR")
        ax.bar(x + width / 2, outliers["zscore_flag_count"], width, label="|z-score| > 3")
        ax.set_xticks(x, outliers["column"], rotation=20, ha="right")
        ax.set_ylabel("Flagged observations")
        ax.set_title("Outlier screening comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "outlier_method_comparison.png", dpi=160)
        plt.close(fig)

    salary_col = str(config.get("salary_column", "annual_fixed_pay"))
    hours_col = str(config.get("hours_column", "avg_hours_worked_3m"))
    tenure_col = str(config.get("tenure_column", "tenure_months"))
    if salary_col in df.columns:
        plot_distribution_with_fences(df[salary_col], salary_col, figures_dir / "dist_salary.png", "INR per year")
    if hours_col in df.columns:
        plot_distribution_with_fences(df[hours_col], hours_col, figures_dir / "dist_hours.png", "hours per day")
    if tenure_col in df.columns:
        plot_distribution_with_fences(df[tenure_col], tenure_col, figures_dir / "dist_tenure.png", "months")

    # --- Machine draft summary ---
    top_missing_text = "\n".join(
        f"- `{row.column}`: {int(row.missing_count):,} missing ({float(row.missing_pct):.2f}%). Decision to be documented."
        for row in data_dict.loc[data_dict["missing_count"] > 0].head(15).itertuples()
    ) or "- No missing values detected."
    outlier_text = "\n".join(
        f"- `{r.column}`: IQR flagged {int(r.iqr_flag_count):,}; |z-score| > 3 flagged {int(r.zscore_flag_count):,}. These are review flags, not confirmed errors."
        for r in outliers.itertuples()
    ) or "- No configured outlier columns were found."
    leak_text = "\n".join(f"- `{c}`: verify whether available at the prediction time." for c in leakage_candidates_present) or "- No configured candidate columns were found. Check the candidate list after reviewing the data dictionary."
    if target_col:
        target_line = f"- Target column detected: `{target_col}`."
        if positive_rate is not None:
            target_line += f"\n- Rate matching configured positive label values: {positive_rate:.2f}%. Confirm the configured positive values match the actual label semantics."
        else:
            target_line += "\n- The class distribution was saved, but a positive/attrition rate was not inferred. Verify the positive label values in the config."
    else:
        target_line = "- No target column matched the configured candidates. Update the candidate list only after confirming the actual target column."

    summary = f"""# Task 1: Data Audit & Exploration (Initial Automated Profile)

## Scope

- Input workbook: `{input_path.name}`
- Worksheet read: `{sheet_name}` only
- Rows: **{n_rows:,}**
- Columns: **{n_cols:,}**
- Exact duplicate rows: **{duplicate_row_count:,}**
- This report is a profiling draft, not the final human-reviewed audit.
- The source workbook was not modified. No rows or columns were dropped, filled, capped, or rewritten.
- Workbook links are not opened or followed by this script. The configured planted-answer-key sheet is not read.

## Missingness

{top_missing_text}

Complete inventory: `data_dictionary.csv`; decision fields: `missingness.csv`.

## Outlier screening

The IQR method flags values outside Q1 - 1.5xIQR or Q3 + 1.5xIQR. The z-score screen flags absolute z-scores above 3.0. Both are screening methods and can flag legitimate extreme values.

{outlier_text}

## Target balance

{target_line}

The class counts and percentages are in `target_distribution.csv`.

## Potential leakage fields

These are candidates requiring manual review. Exclude a field from model inputs when it would not be known at the prediction time or directly/indirectly reveals the outcome.

{leak_text}

Full per-column review: `leakage_register.csv`.

## Duplicate review

See `duplicate_review.csv`. Duplicate identifiers require investigation in context; they are not automatically deleted.

## Human decisions still required

1. Complete `definition_in_own_words` in `data_dictionary.csv`.
2. Document a choice and rationale for each missing column in `missingness.csv`.
3. Inspect each flagged outlier before deciding to retain, flag, correct, or cap it.
4. Confirm target column and positive label semantics from the source specification.
5. Finalize the leakage list based on when predictions are intended to be made.
6. Review suspicious ranges and identifier duplication, and add any findings to the final report.

## Plain-language conclusion (complete after review)

Write 1-2 paragraphs describing the most important data-quality risks, what can safely be used, what needs validation, and the decisions that will become rules in Task 2.
"""
    (output_dir / "audit_summary.md").write_text(summary, encoding="utf-8")

    # --- Final 2-3 page plain-language report with measured numbers ---
    # Post-audit validation output is optional: the normal audit never reads the
    # answer key. When `planted_issue_validation.csv` exists (written by the
    # separate validation script), its aggregate results are embedded below.
    validation_path = output_dir / "planted_issue_validation.csv"
    validation_df = None
    if validation_path.exists():
        try:
            validation_df = pd.read_csv(validation_path)
        except Exception as exc:
            print(f"WARNING: could not read {validation_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            validation_df = None
    for figure_prefix, dest in (
        ("figures", output_dir / "TASK1_AUDIT_REPORT.md"),
        ("generated/figures", PROJECT_ROOT / "task1_data_audit" / "reports" / "TASK1_AUDIT_REPORT.md"),
    ):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            build_final_report(
                input_name=input_path.name,
                sheet_name=sheet_name,
                n_rows=n_rows,
                n_cols=n_cols,
                duplicate_row_count=duplicate_row_count,
                business_dups=business_dups,
                data_dict=data_dict,
                outliers=outliers,
                target_col=target_col,
                target_table=target_table,
                positive_rate=positive_rate,
                leakage_full=leakage_full,
                duplicates=duplicates,
                invalids=invalids,
                decisions=decisions,
                salary_col=salary_col,
                hours_col=hours_col,
                tenure_col=tenure_col,
                figure_prefix=figure_prefix,
                validation_df=validation_df,
            ),
            encoding="utf-8",
        )
    top_report = PROJECT_ROOT / "task1_data_audit" / "reports" / "TASK1_AUDIT_REPORT.md"

    print(f"Audit profile complete for {input_path.name} :: {sheet_name}")
    print(f"Rows: {n_rows:,} | Columns: {n_cols:,} | Exact duplicate rows: {duplicate_row_count:,}")
    print(f"Outputs: {output_dir}")
    print(f"Final report: {top_report}")
    print("Note: the script profiles one configured sheet only and does not modify source data.")
    return 0


def build_final_report(
    *,
    input_name: str,
    sheet_name: str,
    n_rows: int,
    n_cols: int,
    duplicate_row_count: int,
    business_dups: dict,
    data_dict: pd.DataFrame,
    outliers: pd.DataFrame,
    target_col: str | None,
    target_table: pd.DataFrame,
    positive_rate: float | None,
    leakage_full: pd.DataFrame,
    duplicates: pd.DataFrame,
    invalids: pd.DataFrame,
    decisions: pd.DataFrame,
    salary_col: str,
    hours_col: str,
    tenure_col: str,
    figure_prefix: str = "figures",
    validation_df: pd.DataFrame | None = None,
) -> str:
    """Compose the plain-language audit report (no row-level data)."""
    F = figure_prefix
    miss = data_dict.set_index("column")

    def _miss(col: str) -> str:
        r = miss.loc[col]
        return f"{int(r['missing_count']):,} ({float(r['missing_pct']):.2f}%)"

    post_cols = ["exit_date", "resignation_date", "exit_reason", "exit_interview_done", "fnf_settlement_amount"]
    post_pcts = [float(miss.loc[c, "missing_pct"]) for c in post_cols if c in miss.index]
    post_range = f"{min(post_pcts):.1f}–{max(post_pcts):.1f}%" if post_pcts else "N/A"

    miss_table = "\n".join([
        f"| Exit and settlement fields (5 post-outcome columns) | {post_range} | Not predictors — excluded (see leakage) |",
        f"| `months_since_last_promotion` | {_miss('months_since_last_promotion')} | Meaningful gap (no promotion yet) — keep plus flag |"
        if "months_since_last_promotion" in miss.index else "",
        f"| `last_engagement_score` | {_miss('last_engagement_score')} | Fill in Task 2 + flag; check source |"
        if "last_engagement_score" in miss.index else "",
        f"| `last_raise_pct` | {_miss('last_raise_pct')} | Meaningful gap (no raise yet) — keep plus flag |"
        if "last_raise_pct" in miss.index else "",
        f"| `last_performance_rating` | {_miss('last_performance_rating')} | Fill in Task 2 |"
        if "last_performance_rating" in miss.index else "",
        f"| `commute_distance_km` | {_miss('commute_distance_km')} | Fill in Task 2 |"
        if "commute_distance_km" in miss.index else "",
        f"| `highest_education` | {_miss('highest_education')} | Fill in Task 2 |"
        if "highest_education" in miss.index else "",
    ])
    miss_table = "\n".join(line for line in miss_table.split("\n") if line.strip())

    out_lines = []
    for r in outliers.itertuples():
        out_lines.append(
            f"| `{r.column}` | {int(r.iqr_flag_count):,} ({float(r.iqr_flag_pct):.1f}%) | "
            f"{int(r.zscore_flag_count):,} ({float(r.zscore_flag_pct):.1f}%) |"
        )
    out_table = "\n".join(out_lines) if out_lines else "| — | — | — |"

    if target_col and not target_table.empty and positive_rate is not None:
        counts = {str(r.class_value): int(r.count) for r in target_table.itertuples()}
        leavers = counts.get("1", 0)
        stayers = counts.get("0", 0)
        target_head = f"{leavers:,} of {n_rows:,} employees left ({positive_rate:.2f}%)"
        target_para = (
            f"Our working target is `{target_col}`: 1 means the employee left within 12 months after "
            f"the observation date. {target_head}; {stayers:,} stayed. Roughly one in seven employees left. "
            f"A model that always predicts 'stayed' would look about {(100 - positive_rate):.1f}% accurate "
            "while finding zero leavers — so accuracy alone would mislead us, and later evaluation must use "
            "recall, precision and ranking quality with stratified splits. "
            "`left_within_12m` is our working target only; formal Task 3 approval is still open."
        )
    else:
        target_para = "No target column could be confirmed, so no prevalence is reported rather than guessed."

    excluded = leakage_full[leakage_full["exclude_from_model"].str.contains("Exclude", na=False)]
    post_outcome = [c for c in excluded["column"].tolist() if str(c).casefold() in {"exit_date", "resignation_date", "exit_reason", "exit_interview_done", "fnf_settlement_amount", "final_settlement", "final_settlement_amount"}]
    post_list = ", ".join(f"`{c}`" for c in post_outcome) if post_outcome else "—"

    def _inv(column: str, needle: str) -> str:
        hit = invalids[(invalids["column"] == column) & invalids["check"].str.contains(needle, na=False)]
        if hit.empty:
            return "0"
        r = hit.iloc[0]
        return f"{int(r['suspicious_count']):,} ({float(r['suspicious_pct']):.2f}%)"

    biz_extra = int(business_dups.get("extra_duplicate_rows", 0))
    biz_groups = int(business_dups.get("duplicate_groups", 0))

    dec_lines = []
    for r in decisions.itertuples():
        dec_lines.append(
            f"| {r.decision_id} | {r.area} | {r.decision} | {r.reason} | {r.task2_rule} | {r.status} |"
        )
    dec_table = "\n".join(dec_lines) if dec_lines else "| — | — | — | — | — | — |"

    validation_section = render_validation_section(validation_df)

    return f"""# Aura Workforce — Task 1: Data Audit and Exploration

**Source:** `{input_name}` · **Worksheet:** `{sheet_name}` only · **Rows:** {n_rows:,} · **Columns:** {n_cols:,}

> Plain-language summary of measured findings. Supporting evidence lives in the CSV files and charts beside this report. No employee-level records are disclosed and the workbook was not modified.

## 1. Dataset overview

Each row is one employee observed on one observation date; the label records whether that person left during the following 12 months (meaning inferred from column patterns — please confirm). Only `{sheet_name}` was read explicitly; links were not opened and the answer key stayed sealed during the independent audit (see the validation appendix for the later authorized comparison). The sheet holds {n_cols} columns: employee attributes, work patterns, two ID columns, the label, and five post-outcome fields. Full inventory: `column_profile.csv`, `data_dictionary.csv`.

![Missing values]({F}/missingness_top20.png)

## 2. Executive summary

We inspected {n_rows:,} employee records across {n_cols} columns. The headline issues: (a) exit and settlement fields exist only for leavers and must never be model inputs — that is **data leakage**, using information unavailable at prediction time; (b) some gaps are meaningful (no promotion or raise yet) while others need filling in Task 2; (c) pay is skewed with a few typo-like extremes, plus small counts of impossible negatives and out-of-range shares needing source checks; (d) location spellings vary and a few records repeat under a fresh technical ID. Our decisions below handle each case. Subject to them — exclusions, train-only imputation, location mapping and timeline-aware dedup — the data can proceed to Task 2. An **outlier** here means "worth investigating", not "definitely wrong".

## 3. Missing values and decisions

Gaps cluster in a few columns rather than spreading everywhere. "Impute" means filling blanks with a typical value learned from training data; a "was-missing flag" is an extra yes/no column recording that the value was absent, so the model can learn when absence itself matters.

| Column(s) | Missing | Decision |
|---|---:|---|
{miss_table}

Post-outcome gaps (~85% blank because stayers have no exit data) are excluded as features, not filled. Structural gaps (no promotion/raise yet) are kept with a was-missing flag, since absence carries meaning. Ordinary gaps (engagement, performance, commute, education) are filled in Task 2 using training-data statistics, with an extra flag for engagement at 22%. No rows are dropped for missing features. Full inventory: `missing_values.csv`.

### Think for yourself: What should we do when a column has 40% missing values?

Do not drop it, fill it, or flag it on the percentage alone — first ask what the column means, why values are missing, whether it will exist when real predictions are made, and whether it helps. If it is a valid, useful predictor, our proposal is: **keep it, add a `was_missing` flag where justified, and fill blanks with a method fitted on training data only** (for example the median for numbers, the most common value for categories). If the gap itself is meaningful — as with "no promotion yet" — preserve that meaning instead of averaging it away. Drop the column only for a documented reason: it reveals the future outcome, will not exist at prediction time, is unreliable, or adds too little to justify itself. Note this 40% case is hypothetical — no column in this dataset sits at exactly 40% (ours jump from 22% to 61%), so treat this as the general rule behind decisions D01–D03.

## 4. Outlier comparison

Two screens ran on `{salary_col}` (pay/year), `{hours_col}` (hours/day) and `{tenure_col}` (months). The IQR rule flags values far outside the middle half of the data; the z-score flags values unusually far from the average relative to spread. They disagree most when data is skewed: a long pay tail inflates the average and spread, so z-score stays quiet while IQR speaks up.

| Variable | IQR flags | Z-score flags |
|---|---:|---:|
{out_table}

Pay flags are mostly legitimate high earners; hours cluster near 8.7 with one implausible ~1.8-hour value; tenure includes four impossible negatives. A flag means "investigate", never automatic deletion — the four negatives and similar rule-breakers are **suspected data errors**, the rest are ordinary statistical extremes. Every flagged record, with thresholds, direction and business-range verdict, is in `outlier_details.csv`.

![Outlier comparison]({F}/outlier_method_comparison.png)

![Salary]({F}/dist_salary.png) ![Hours]({F}/dist_hours.png) ![Tenure]({F}/dist_tenure.png)

## 5. Attrition and class imbalance

{target_para}

![Target distribution]({F}/target_distribution.png)

## 6. Data leakage

Leakage means letting the model peek at information that reveals the answer or would not exist yet when predicting. The confirmed post-outcome fields — {post_list} — are recorded only for leavers and must be excluded from inputs. Identifiers (`row_id`, `employee_id`) are excluded for privacy and because they cannot generalize, and the alternate outcome label is excluded since it describes the same event. All {n_cols} columns were reviewed one by one; the full register is in `leakage_register.csv`.

## 7. Data-quality findings and decisions

- Duplicates: {duplicate_row_count} full-row match(es), but {biz_extra} extra row(s) in {biz_groups} group(s) once the technical `row_id` is set aside — copies with a fresh ID hide from full-row comparison. Task 2 dedups on the business key (D08).
- Negative tenure: {_inv('tenure_months', 'negative tenure')} rows — impossible; verify against HR, quarantine (D09, confirmed).
- Negative experience: {_inv('total_experience_years', 'negative experience')} rows — verify source, flag (D09, provisional).
- Work-from-home share below 0: {_inv('wfh_share_3m', 'below 0')} rows; above 1: {_inv('wfh_share_3m', 'above 1')} rows — needs definition check before any use (D09, requires confirmation).
- Attendance above 1.0: {_inv('attendance_rate_3m', 'rate above 1')} rows — confirm definition, flag rows (D09).
- Implausible hours: {_inv('avg_hours_worked_3m', 'plausible')} row — investigate the extreme, keep the rest (D04).
- Pay-vs-ratio inconsistency: {_inv('annual_fixed_pay', 'inconsistency')} rows — matches decimal-place-typo pattern; verify against payroll, never auto-correct (D04/D09, provisional).
- Location variants: {_inv('work_location', 'spellings')} rows — normalize with the HR-approved mapping in Task 2 (D09).
- Resignation before observation: {_inv('resignation_date', 'before observation')} of dated cases — clarify dating before any date feature (D10).
- No non-positive pay found — a good sign. Record-level evidence: `data_quality_issue_details.csv`, `duplicate_review.csv`, `invalid_values.csv`.

| ID | Area | Action | Why | Task 2 rule | Status |
|---|---|---|---|---|---|
{dec_table}

## 8. Handoff to Task 2

The table above is the handoff: Task 2 must implement these rules in a reusable, versioned pipeline — missing-value handling with was-missing flags, leakage exclusions, outlier/invalid handling with robust scaling and quarantine, business-key dedup, approved location mapping — fitting every learned step (imputers, scalers, encoders) on training data only and applying it unchanged to validation, test and future data. Task 2 itself is not built here.

## 9. Open questions

Working target vs formal Task 3 approval; canonical location list; causes of negatives, out-of-range shares, attendance above 1 and pre-observation resignations; department prior-rate window; whether engagement gaps are systematic; `comp_ratio` definition. All need HR/source confirmation.

{validation_section}
*Artifacts: `column_profile.csv`, `data_dictionary.csv`, `missing_values.csv`, `outlier_summary.csv`, `outlier_details.csv`, `target_distribution.csv`, `leakage_register.csv`, `decision_register.csv`, `duplicate_review.csv`, `invalid_values.csv`, `data_quality_issue_details.csv`, `planted_issue_validation.csv` (after validation), `audit_summary.md`, `figures/`.*
"""


def missing_decision(column: str) -> str:
    """Short human decision label used inside the final report table."""
    mapping = {
        "months_since_last_promotion": "Preserve + indicator (no promotion yet)",
        "last_raise_pct": "Preserve + indicator (no raise yet)",
        "exit_date": "Exclude as feature (post-outcome)",
        "resignation_date": "Exclude as feature (post-outcome)",
        "exit_reason": "Exclude as feature (post-outcome)",
        "exit_interview_done": "Exclude as feature (post-outcome)",
        "fnf_settlement_amount": "Exclude as feature (post-outcome)",
        "last_engagement_score": "Impute (train-fit) + indicator; investigate",
        "last_performance_rating": "Impute (train-fit)",
        "commute_distance_km": "Impute (train-fit)",
        "highest_education": "Impute (train-fit, mode)",
    }
    return mapping.get(column, "Impute (train-fit) or investigate; see decision register")


if __name__ == "__main__":
    raise SystemExit(main())
