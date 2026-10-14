"""Post-audit validation of Task 1 findings against planted issues.

Reads only two explicitly authorized worksheets: the configured
``attrition_dataset`` sheet and ``answer_key_planted_dirt``. The answer key is
read here ONLY because the independent audit is already complete; the normal
audit (``scripts/run_audit.py``) never reads it.

Outputs aggregate counts only. Employee IDs are used in memory to match
planted cases to detected rows and are never written to disk or printed.

After writing ``planted_issue_validation.csv``, this script re-runs the normal
audit so the generated report embeds the validation appendix. Re-running the
audit alone (without this script) never requires the answer key.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _entry in (PROJECT_ROOT / "task1_data_audit" / "src", PROJECT_ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from audit.audit_utils import (  # noqa: E402
    business_record_duplicates,
    outlier_comparison,
    pay_comp_consistency,
    summarize_planted_validation,
)

DATA_SHEET_DEFAULT = "attrition_dataset"
ANSWER_SHEET_DEFAULT = "answer_key_planted_dirt"
# Formatting variants the independent audit already tracks (no answer-key use).
LOCATION_VARIANTS = ["BENGALURU", "Bangalore", "bengaluru", "chennai", "delhi ncr", "Delhi-NCR", "Madras"]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Task 1 config must be a JSON object.")
    return cfg


def _ids(frame: pd.DataFrame) -> set[str]:
    return set(frame["employee_id"].astype(str)) if "employee_id" in frame.columns else set()


def build_rows(df: pd.DataFrame, key: pd.DataFrame) -> list[dict[str, object]]:
    """Compare planted cases to independently measurable findings (aggregates only)."""
    n = len(df)
    rows: list[dict[str, object]] = []

    by_type = {t: g for t, g in key.groupby("issue_type")}

    def key_ids(issue_type: str, detail_contains: str = "") -> set[str]:
        g = by_type.get(issue_type, key.iloc[0:0])
        if detail_contains:
            g = g[g["detail"].astype(str).str.contains(detail_contains, case=False, na=False)]
        return _ids(g)

    # --- A. Missing values -------------------------------------------------
    for label, column, substr in (
        ("missing_commute_distance", "commute_distance_km", "commute"),
        ("missing_highest_education", "highest_education", "education"),
    ):
        expected_ids = key_ids("MISSING_VALUE", substr)
        actual_ids = _ids(df.loc[df[column].isna()])
        matched = len(expected_ids & actual_ids)
        missed = len(expected_ids - actual_ids)
        detected = int(df[column].isna().sum())
        pct = detected / n * 100 if n else 0.0
        extra_note = ""
        if label == "missing_commute_distance":
            extra_note = (
                f" Measured {detected}/{n} = {pct:.2f}% on the full sheet vs the key's '2.8%': "
                f"the key's figure is consistent with a 2000-row historical-batch denominator "
                f"(55/2000 = 2.75% ~ 2.8%), while this sheet also holds snapshot rows and 5 of the "
                f"55 blanks fall outside that batch (50/2000 = 2.50% within it). "
                "The report's 2.30% is correct for all rows."
            )
        rows.append(
            {
                "issue_category": label,
                "expected_planted_count": len(expected_ids),
                "detected_count": detected,
                "matched_count": matched,
                "missed_count": missed,
                "additional_flagged_count": max(detected - matched, 0),
                "validation_status": "detected" if missed == 0 and detected == len(expected_ids) else "partially-detected",
                "explanation": (
                    f"Planted {len(expected_ids)}; audit measured {detected} ({pct:.2f}%). "
                    f"Matched {matched}, missed {missed}." + extra_note
                ),
            }
        )

    # --- B. Inconsistent work_location categories --------------------------
    loc_key_ids = key_ids("INCONSISTENT_CATEGORY")
    variant_mask = df["work_location"].isin(LOCATION_VARIANTS) if "work_location" in df.columns else pd.Series(False, index=df.index)
    detected_loc = _ids(df.loc[variant_mask])
    per_variant = (
        df.loc[variant_mask, "work_location"].value_counts().to_dict() if "work_location" in df.columns else {}
    )
    breakdown = ", ".join(f"{k}={int(v)}" for k, v in sorted(per_variant.items()))
    rows.append(
        {
            "issue_category": "inconsistent_work_location",
            "expected_planted_count": len(loc_key_ids),
            "detected_count": int(variant_mask.sum()),
            "matched_count": len(loc_key_ids & detected_loc),
            "missed_count": len(loc_key_ids - detected_loc),
            "additional_flagged_count": max(int(variant_mask.sum()) - len(loc_key_ids & detected_loc), 0),
            "validation_status": "detected"
            if len(loc_key_ids - detected_loc) == 0 and int(variant_mask.sum()) == len(loc_key_ids)
            else "partially-detected",
            "explanation": (
                f"Planted {len(loc_key_ids)} across 7 variants; audit flagged {int(variant_mask.sum())} "
                f"({breakdown}). Matched {len(loc_key_ids & detected_loc)}. Variants are formatting/case "
                "differences plus one alternate name (Madras for Chennai); canonicalization in Task 2 is "
                "justified for these, not a blanket rule for every distinct string."
            ),
        }
    )

    # --- C. Salary typos (10x, comp_ratio inconsistent) ---------------------
    sal_key_ids = key_ids("SALARY_TYPO_OUTLIER")
    check = pay_comp_consistency(df)
    flagged_count = int(check.get("flagged_count", 0))
    # Which planted rows does the specific check catch? (IDs in memory only.)
    matched_sal = 0
    if not check.get("skipped") and "annual_fixed_pay" in df.columns and "comp_ratio" in df.columns and "job_level" in df.columns:
        pay = pd.to_numeric(df["annual_fixed_pay"], errors="coerce")
        ratio = pd.to_numeric(df["comp_ratio"], errors="coerce")
        valid = pay.notna() & ratio.notna() & (ratio != 0)
        ref = (pay / ratio).where(valid)
        medians = ref.groupby(df["job_level"]).transform("median")
        rel = (ref / medians).where(valid & medians.notna() & (medians != 0))
        flagged_ids = _ids(df.loc[(rel > float(check.get("threshold", 3.0))).fillna(False)])
        matched_sal = len(sal_key_ids & flagged_ids)
    # Generic screens: were planted rows at least statistical outliers?
    generic = outlier_comparison(df["annual_fixed_pay"], "annual_fixed_pay") if "annual_fixed_pay" in df.columns else {}
    generic_matched = 0
    if "annual_fixed_pay" in df.columns and sal_key_ids:
        v = pd.to_numeric(df["annual_fixed_pay"], errors="coerce").astype(float)
        q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        generic_matched = int(df.loc[((v < lo) | (v > hi)) & df["employee_id"].astype(str).isin(sal_key_ids)].shape[0])
    iqr_n = int(generic.get("iqr_flag_count", 0)) if isinstance(generic, dict) else 0
    z_n = int(generic.get("zscore_flag_count", 0)) if isinstance(generic, dict) else 0
    rows.append(
        {
            "issue_category": "salary_typo_10x",
            "expected_planted_count": len(sal_key_ids),
            "detected_count": flagged_count,
            "matched_count": matched_sal,
            "missed_count": len(sal_key_ids) - matched_sal,
            "additional_flagged_count": max(iqr_n - generic_matched, 0),
            "validation_status": "detected-after-correction" if matched_sal == len(sal_key_ids) and sal_key_ids else "partially-detected",
            "explanation": (
                f"Planted {len(sal_key_ids)}. Initial audit caught {generic_matched}/{len(sal_key_ids)} only as generic statistical "
                f"outliers (IQR {iqr_n}, z {z_n}) without "
                "identifying the pay-vs-ratio inconsistency. Post-validation correction adds a specific check: "
                "pay-implied reference (pay/comp_ratio) vs the job-level median; "
                f"{flagged_count} rows exceed 3.0x with no other rows above 2x. Never auto-correct values."
            ),
        }
    )

    # --- D. Impossible attendance ------------------------------------------
    att_key_ids = key_ids("IMPOSSIBLE_VALUE")
    att_actual = _ids(df.loc[pd.to_numeric(df["attendance_rate_3m"], errors="coerce") > 1]) if "attendance_rate_3m" in df.columns else set()
    att_count = len(att_actual)
    rows.append(
        {
            "issue_category": "impossible_attendance_gt_1",
            "expected_planted_count": len(att_key_ids),
            "detected_count": att_count,
            "matched_count": len(att_key_ids & att_actual),
            "missed_count": len(att_key_ids - att_actual),
            "additional_flagged_count": max(att_count - len(att_key_ids & att_actual), 0),
            "validation_status": "detected" if len(att_key_ids - att_actual) == 0 else "partially-detected",
            "explanation": (
                f"Planted {len(att_key_ids)}; audit invalid-value check found {att_count}. "
                "Decision register D09 quarantines/flags out-of-range shares for Task 2."
            ),
        }
    )

    # --- E. Negative tenure -------------------------------------------------
    ten_key_ids = key_ids("NEGATIVE_TENURE")
    ten_actual = _ids(df.loc[pd.to_numeric(df["tenure_months"], errors="coerce") < 0]) if "tenure_months" in df.columns else set()
    join_cols = [c for c in df.columns if any(k in str(c).lower() for k in ("join", "hire", "start_date"))]
    rows.append(
        {
            "issue_category": "negative_tenure",
            "expected_planted_count": len(ten_key_ids),
            "detected_count": len(ten_actual),
            "matched_count": len(ten_key_ids & ten_actual),
            "missed_count": len(ten_key_ids - ten_actual),
            "additional_flagged_count": max(len(ten_actual) - len(ten_key_ids & ten_actual), 0),
            "validation_status": "detected" if len(ten_key_ids - ten_actual) == 0 else "partially-detected",
            "explanation": (
                f"Planted {len(ten_key_ids)}; audit found {len(ten_actual)}. "
                + (
                    "No join-date column exists in the sheet, so the observation-date-vs-join-date "
                    "relationship cannot be confirmed from available information."
                    if not join_cols
                    else f"Join-date-like column(s) present ({', '.join(join_cols)}); relationship check recorded in invalid_values."
                )
            ),
        }
    )

    # --- F. Duplicate records ----------------------------------------------
    dup_key_ids = key_ids("DUPLICATE_ROW")
    full_dups = int(df.duplicated().sum())
    biz = business_record_duplicates(df, exclude_columns=["row_id"])
    biz_extra = int(biz["extra_duplicate_rows"])
    biz_ids = _ids(df.loc[df.duplicated(subset=[c for c in df.columns if str(c).casefold() != "row_id"], keep=False)])
    rows.append(
        {
            "issue_category": "duplicate_records",
            "expected_planted_count": len(dup_key_ids),
            "detected_count": biz_extra,
            "matched_count": len(dup_key_ids & biz_ids),
            "missed_count": len(dup_key_ids - biz_ids),
            "additional_flagged_count": max(biz_extra - len(dup_key_ids & biz_ids), 0),
            "validation_status": "detected-after-correction" if len(dup_key_ids - biz_ids) == 0 else "partially-detected",
            "explanation": (
                f"Planted {len(dup_key_ids)} appended copies. Full-row duplicates: {full_dups} (copies carry a "
                f"fresh row_id, so the initial '0 exact duplicates' statement was technically correct but "
                f"incomplete). Business-record duplicates excluding row_id: {biz_extra} extra rows in "
                f"{biz['duplicate_groups']} groups, matching all {len(dup_key_ids & biz_ids)} planted cases. "
                "Initial audit noted 6 repeated employee IDs but did not run the business-field comparison; "
                "added post-validation. Repeated IDs alone were rightly not treated as proof."
            ),
        }
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="task1_data_audit/configs/task1_config.json")
    parser.add_argument("--input", default=None)
    parser.add_argument("--sheet", default=None, help="Override attrition data worksheet")
    parser.add_argument("--answer-sheet", default=ANSWER_SHEET_DEFAULT)
    parser.add_argument("--output", default=None, help="Override validation CSV path")
    parser.add_argument("--no-rerun", action="store_true", help="Skip re-running the audit afterwards")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 2
    config = load_config(config_path)

    input_value = args.input or config.get("dataset_file")
    input_path = Path(input_value)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    data_sheet = args.sheet or config.get("sheet_name", DATA_SHEET_DEFAULT)
    output_dir = Path(config.get("output_dir", "task1_data_audit/reports/generated"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_path = Path(args.output) if args.output else output_dir / "planted_issue_validation.csv"

    if not input_path.exists():
        print(f"ERROR: workbook not found: {input_path}", file=sys.stderr)
        return 2
    # Explicit single-sheet reads only; hyperlinks are never followed.
    try:
        df = pd.read_excel(input_path, sheet_name=data_sheet)
    except ValueError as exc:
        print(f"ERROR: data worksheet {data_sheet!r} not readable: {exc}", file=sys.stderr)
        return 2
    try:
        key = pd.read_excel(input_path, sheet_name=args.answer_sheet)
    except ValueError as exc:
        print(f"ERROR: answer-key worksheet {args.answer_sheet!r} not readable: {exc}", file=sys.stderr)
        return 2
    for col in ("employee_id", "issue_type", "detail"):
        if col not in key.columns:
            print(f"ERROR: answer-key sheet lacks expected column {col!r}.", file=sys.stderr)
            return 2

    frame = summarize_planted_validation(build_rows(df, key))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    for r in frame.itertuples():
        print(
            f"{r.issue_category}: expected={int(r.expected_planted_count)} detected={int(r.detected_count)} "
            f"matched={int(r.matched_count)} missed={int(r.missed_count)} status={r.validation_status}"
        )
    print(f"Validation table: {output_path} ({len(frame)} categories, no record-level data)")

    if not args.no_rerun:
        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "task1_data_audit" / "scripts" / "run_audit.py"), "--config", str(config_path)],
            capture_output=True,
            text=True,
        )
        print(proc.stdout[-1500:] if proc.stdout else "")
        if proc.returncode != 0:
            print(proc.stderr[-1500:] if proc.stderr else "", file=sys.stderr)
            return proc.returncode
        print("Audit re-run complete; report now embeds the validation appendix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
