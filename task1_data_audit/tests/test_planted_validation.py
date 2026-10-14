"""Tests for post-audit validation helpers (synthetic fixtures only)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from audit.audit_utils import (
    business_record_duplicates,
    pay_comp_consistency,
    render_validation_section,
    summarize_planted_validation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_business_record_duplicates_excludes_technical_key():
    df = pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "employee_id": ["a", "b", "b"],
            "v": [10, 20, 20],
        }
    )
    assert int(df.duplicated().sum()) == 0  # fresh row_id hides the copy
    out = business_record_duplicates(df, exclude_columns=["row_id"])
    assert out["compared_columns"] == 2
    assert int(out["extra_duplicate_rows"]) == 1
    assert int(out["rows_in_duplicate_groups"]) == 2
    assert int(out["duplicate_groups"]) == 1


def test_business_record_duplicates_empty_and_full_exclusion():
    df = pd.DataFrame({"row_id": [1, 1], "v": [5, 5]})
    assert int(business_record_duplicates(df, exclude_columns=["row_id", "v"])["extra_duplicate_rows"]) == 0
    assert int(business_record_duplicates(df.iloc[0:0], exclude_columns=["row_id"])["extra_duplicate_rows"]) == 0


def test_pay_comp_consistency_flags_10x_typo():
    pays, ratios, levels = [], [], []
    for lvl, base in ((1, 300000), (2, 600000)):
        for i in range(7):
            pays.append(base + i * 1000)
            ratios.append(1.0)
            levels.append(lvl)
    pays.append(300000 * 10)
    ratios.append(1.0)
    levels.append(1)
    df = pd.DataFrame({"annual_fixed_pay": pays, "comp_ratio": ratios, "job_level": levels})
    out = pay_comp_consistency(df)
    assert out["skipped"] == ""
    assert int(out["flagged_count"]) == 1
    assert out["groups_compared"] == 2
    assert float(out["max_unflagged_ratio"]) < 2.0


def test_pay_comp_consistency_clean_and_missing_columns():
    df = pd.DataFrame({"annual_fixed_pay": [100, 200], "comp_ratio": [1.0, 1.0], "job_level": [1, 1]})
    out = pay_comp_consistency(df)
    assert int(out["flagged_count"]) == 0
    missing = pay_comp_consistency(pd.DataFrame({"a": [1]}))
    assert missing["skipped"] != ""
    assert int(missing["flagged_count"]) == 0


def test_summarize_planted_validation_schema_and_bad_status():
    rows = [
        {
            "issue_category": "negative_tenure",
            "expected_planted_count": 4,
            "detected_count": 4,
            "matched_count": 4,
            "missed_count": 0,
            "additional_flagged_count": 0,
            "validation_status": "detected",
            "explanation": "ok",
        }
    ]
    frame = summarize_planted_validation(rows)
    assert list(frame.columns) == [
        "issue_category",
        "expected_planted_count",
        "detected_count",
        "matched_count",
        "missed_count",
        "additional_flagged_count",
        "validation_status",
        "explanation",
    ]
    bad = [dict(rows[0], validation_status="guessed")]
    with pytest.raises(ValueError):
        summarize_planted_validation(bad)


def test_render_validation_section_branches():
    empty = render_validation_section(None)
    assert "Post-Audit Validation Against Planted Issues" in empty
    assert "has not been run" in empty
    frame = summarize_planted_validation(
        [
            {
                "issue_category": "negative_tenure",
                "expected_planted_count": 4,
                "detected_count": 4,
                "matched_count": 4,
                "missed_count": 0,
                "additional_flagged_count": 0,
                "validation_status": "detected",
                "explanation": "all matched",
            }
        ]
    )
    full = render_validation_section(frame)
    assert "negative_tenure" in full and "independent audit above was completed before" in full


def _synthetic_workbook(path: Path) -> None:
    n_base = 20
    df = pd.DataFrame(
        {
            "row_id": list(range(1, n_base + 1)),
            "employee_id": [f"TST-E{i:03d}" for i in range(1, n_base + 1)],
            "annual_fixed_pay": [300000 + i * 1000 for i in range(10)] + [600000 + i * 1000 for i in range(10)],
            "comp_ratio": [1.0] * n_base,
            "job_level": [1] * 10 + [2] * 10,
            "work_location": ["Bengaluru"] * n_base,
            "commute_distance_km": [5.0] * n_base,
            "highest_education": ["B.Tech"] * n_base,
            "attendance_rate_3m": [0.9] * n_base,
            "tenure_months": [12] * n_base,
        }
    )
    df.loc[0, "commute_distance_km"] = float("nan")
    df.loc[1, "commute_distance_km"] = float("nan")
    df.loc[2, "highest_education"] = None
    df.loc[3, "work_location"] = "BENGALURU"
    df.loc[4, "work_location"] = "Madras"
    df.loc[5, "annual_fixed_pay"] = 3000000  # 10x typo, ratio left at 1.0
    df.loc[6, "attendance_rate_3m"] = 1.2
    df.loc[7, "tenure_months"] = -3
    dup = df.iloc[[8]].copy()
    dup["row_id"] = n_base + 1
    df = pd.concat([df, dup], ignore_index=True)
    key = pd.DataFrame(
        {
            "employee_id": [
                "TST-E001", "TST-E002", "TST-E003", "TST-E004", "TST-E005",
                "TST-E006", "TST-E007", "TST-E008", "TST-E009",
            ],
            "issue_type": [
                "MISSING_VALUE", "MISSING_VALUE", "MISSING_VALUE",
                "INCONSISTENT_CATEGORY", "INCONSISTENT_CATEGORY",
                "SALARY_TYPO_OUTLIER", "IMPOSSIBLE_VALUE", "NEGATIVE_TENURE", "DUPLICATE_ROW",
            ],
            "detail": [
                "commute_distance_km blank", "commute_distance_km blank", "highest_education blank",
                "work_location variant", "work_location variant",
                "annual_fixed_pay is 10x too high; comp_ratio no longer matches",
                "attendance_rate_3m is above 1.0",
                "tenure_months is negative (observation date before join date)",
                "exact duplicate row appended at end of sheet",
            ],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="attrition_dataset", index=False)
        key.to_excel(writer, sheet_name="answer_key_planted_dirt", index=False)


def test_validate_script_end_to_end_synthetic(tmp_path: Path):
    xlsx = tmp_path / "synth.xlsx"
    _synthetic_workbook(xlsx)
    out_csv = tmp_path / "planted_issue_validation.csv"
    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "task1_data_audit" / "scripts" / "validate_planted_issues.py"),
            "--input", str(xlsx),
            "--sheet", "attrition_dataset",
            "--answer-sheet", "answer_key_planted_dirt",
            "--output", str(out_csv),
            "--no-rerun",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    frame = pd.read_csv(out_csv)
    assert len(frame) == 7
    by_cat = frame.set_index("issue_category")
    assert int(by_cat.loc["missing_commute_distance", "matched_count"]) == 2
    assert int(by_cat.loc["missing_highest_education", "matched_count"]) == 1
    assert int(by_cat.loc["inconsistent_work_location", "matched_count"]) == 2
    assert int(by_cat.loc["salary_typo_10x", "matched_count"]) == 1
    assert by_cat.loc["salary_typo_10x", "validation_status"] == "detected-after-correction"
    assert int(by_cat.loc["duplicate_records", "matched_count"]) == 1
    text = out_csv.read_text(encoding="utf-8")
    assert "TST-E001" not in text  # no employee-level disclosure
