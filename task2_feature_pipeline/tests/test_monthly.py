"""Unit tests for the monthly attendance/payroll path (synthetic fixtures only)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features import monthly as M
from features import pipeline as pipeline_mod

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def synthetic_monthly(n_emp: int = 8, months: int = 6, seed: int = 3):
    rng = np.random.default_rng(seed)
    month_starts = pd.date_range("2025-04-01", periods=months, freq="MS")
    att_rows, pay_rows = [], []
    for e in range(1, n_emp + 1):
        for m in month_starts:
            att_rows.append({
                "attendance_feature_id": f"A{e}-{m:%Y%m}",
                "employee_id": f"ME-{e:03d}",
                "month": m,
                "department_id": "D01",
                "job_level": 2,
                "scheduled_days": 22,
                "present_days": 20,
                "wfh_days": 2,
                "half_days_unpaid": 0,
                "paid_leave_days": 1,
                "unpaid_absence_days": 1,
                "longest_unpaid_streak_days": 1,
                "late_arrivals": int(rng.integers(0, 5)),
                "avg_hours_worked": 8.5,
                "overtime_hours": float(rng.integers(0, 8)),
                "attendance_rate": 0.9,
            })
            pay_rows.append({
                "payroll_item_id": f"P{e}-{m:%Y%m}",
                "run_id": f"PR-{m:%Y-%m}",
                "employee_id": f"ME-{e:03d}",
                "month": m,
                "department_id": "D01",
                "role_id": "R001",
                "job_level": 2,
                "overtime_eligible": "N",
                "monthly_fixed_gross": 50000.0,
                "payable_days": 22.0,
                "lop_days": 0.0,
                "overtime_hours_paid": 0.0,
                "overtime_pay": 0.0,
                "variable_pay": 0.0,
                "variable_pay_type": np.nan,
                "gross_pay": 50000.0,
                "pf_employee": 1000.0,
                "esi_employee": 0.0,
                "professional_tax": 200,
                "tds": 500.0,
                "other_deductions": 0,
                "total_deductions": 1700.0,
                "net_pay": 48300.0,
            })
    return pd.DataFrame(att_rows), pd.DataFrame(pay_rows)


def test_dedup_ignores_only_item_id():
    att, pay = synthetic_monthly()
    dup = pay.iloc[[0]].copy()
    dup["payroll_item_id"] = "P-DUP"
    pay2 = pd.concat([pay, dup], ignore_index=True)
    clean, log = M.deduplicate_payroll(pay2)
    assert log["removed_duplicate_lines"] == 1
    assert len(clean) == len(pay)


def test_dedup_preserves_conflicting_records():
    att, pay = synthetic_monthly()
    dup = pay.iloc[[0]].copy()
    dup["payroll_item_id"] = "P-DUP"
    dup["gross_pay"] = 99999.0  # same key, different money -> conflict
    pay2 = pd.concat([pay, dup], ignore_index=True)
    with pytest.raises(M.MonthlyDataError, match="disagree"):
        M.deduplicate_payroll(pay2)


def test_join_one_to_one_and_unmatched_key():
    att, pay = synthetic_monthly()
    extra = pay.iloc[[0]].copy()
    extra["payroll_item_id"] = "P-X"
    extra["employee_id"] = "ME-999"
    pay2 = pd.concat([pay, extra], ignore_index=True)
    merged, stats = M.validate_and_join(att, M.deduplicate_payroll(pay2)[0])
    assert stats["matched_keys"] == len(att)
    assert stats["payroll_only_keys"] == 1
    assert stats["attendance_only_keys"] == 0
    assert stats["merged_rows"] == len(att)
    assert list(merged.columns) == M.merged_input_columns()


def test_job_level_mismatch_flag():
    att, pay = synthetic_monthly()
    pay.loc[0, "job_level"] = 5
    merged, stats = M.validate_and_join(att, pay)
    assert stats["job_level_mismatches"] == 1
    feat = M.add_monthly_features(merged)
    assert int(feat["job_level_mismatch"].sum()) == 1
    assert "job_level_att" not in M.monthly_numeric_features()
    assert "job_level_pay" not in M.monthly_numeric_features()


def test_trailing_windows_are_asof():
    att, pay = synthetic_monthly(n_emp=2, months=4)
    late = {(e, m): v for e, m, v in zip(
        att["employee_id"], att["month"].dt.strftime("%Y-%m"), att["late_arrivals"])}
    merged, _ = M.validate_and_join(att, pay)
    feat = M.add_monthly_features(merged)
    row = feat[(feat["employee_id"] == "ME-001") & (feat["month"] == pd.Timestamp("2025-06-01"))]
    expect = (late[("ME-001", "2025-04")] + late[("ME-001", "2025-05")] + late[("ME-001", "2025-06")]) / 3
    assert float(row["late_arrivals_avg_3m"].iloc[0]) == pytest.approx(expect)
    first = feat[(feat["employee_id"] == "ME-001") & (feat["month"] == pd.Timestamp("2025-04-01"))]
    assert float(first["months_of_history"].iloc[0]) == 1
    # perturbing a future month must not change earlier rows
    att2 = att.copy()
    att2.loc[att2["month"] == pd.Timestamp("2025-07-01"), "late_arrivals"] = 999
    merged2, _ = M.validate_and_join(att2, pay)
    feat2 = M.add_monthly_features(merged2)
    june1 = feat[(feat["month"] == "2025-06-01")][["employee_id", "late_arrivals_avg_3m"]].reset_index(drop=True)
    june2 = feat2[(feat2["month"] == "2025-06-01")][["employee_id", "late_arrivals_avg_3m"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(june1, june2)


def test_monthly_fit_save_load_no_nan_and_order(tmp_path: Path):
    att, pay = synthetic_monthly()
    merged, _, _ = M.prepare_monthly_frame(att, pay)
    train, test, info = M.chronological_split(merged, "2025-08")
    assert info["no_future_leak"] is True
    pipe = M.build_monthly_pipeline().fit(train)
    cols = list(pipeline_mod.transform_frame(pipe, train).columns)
    from features import persistence

    meta = M.monthly_metadata(output_columns=cols, train_info=info,
                              dedup_log={"removed_duplicate_lines": 0},
                              join_stats={"matched_keys": 1}, dept_references={},
                              train_rows=len(train), test_rows=len(test))
    paths = persistence.save_artifact(pipe, meta, tmp_path, "m.json")
    pipe2, meta2 = persistence.load_artifact(tmp_path, paths["pipeline"].name, "m.json")
    assert meta2["monthly_component_version"] == M.MONTHLY_COMPONENT_VERSION
    a = pipeline_mod.transform_frame(pipe, test, cols)
    pd.testing.assert_frame_equal(a, pipeline_mod.transform_frame(pipe2, test, cols))
    assert int(a.isna().sum().sum()) == 0


def test_monthly_schema_errors(tmp_path: Path):
    att, pay = synthetic_monthly()
    with pytest.raises(FileNotFoundError):
        M.load_monthly_sheets(tmp_path / "missing.xlsx")
    bad = tmp_path / "bad.xlsx"
    with pd.ExcelWriter(bad, engine="openpyxl") as w:
        att.to_excel(w, sheet_name="attendance_monthly_features", index=False)
        pay.drop(columns=["month"]).to_excel(w, sheet_name="payroll_monthly_features", index=False)
    with pytest.raises(M.MonthlyDataError, match="lacks column"):
        M.load_monthly_sheets(bad)
    merged, _, _ = M.prepare_monthly_frame(att, pay)
    with pytest.raises(M.MonthlyDataError, match="unusable"):
        M.chronological_split(merged, "2025-01")  # everything in holdout
    with pytest.raises(M.MonthlyDataError, match="YYYY-MM"):
        M.chronological_split(merged, "not-a-month")


def test_monthly_cli_on_synthetic_workbook(tmp_path: Path):
    att, pay = synthetic_monthly(n_emp=10, months=6)
    xlsx = tmp_path / "m.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        att.to_excel(w, sheet_name="attendance_monthly_features", index=False)
        pay.to_excel(w, sheet_name="payroll_monthly_features", index=False)
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "features.cli", "monthly-fit-transform",
         "--workbook", str(xlsx), "--output-dir", str(out),
         "--cutoff-month", "2025-08"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (out / "monthly_train_features.csv").exists()
    assert (out / "monthly_alignment.csv").exists()
