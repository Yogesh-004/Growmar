"""Unit tests for Task 1 audit helpers (synthetic fixtures only)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from audit.audit_utils import (
    build_column_profile,
    build_leakage_register,
    candidate_columns,
    class_distribution,
    duplicate_summary,
    infer_column_role,
    issue_detail_records,
    missingness_summary,
    outlier_comparison,
    outlier_details,
    resolve_target_column,
)


def test_missingness_summary_counts_and_pct():
    df = pd.DataFrame({"a": [1, None, 3, None], "b": ["x", "y", "z", "w"]})
    summary = missingness_summary(df).set_index("column")
    assert int(summary.loc["a", "missing_count"]) == 2
    assert float(summary.loc["a", "missing_pct"]) == 50.0
    assert int(summary.loc["b", "missing_count"]) == 0


def test_missingness_summary_empty_frame():
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    summary = missingness_summary(df)
    assert float(summary.loc[0, "missing_pct"]) == 0.0


def test_outlier_comparison_flags_extreme_value():
    series = pd.Series([10] * 20 + [1000])
    result = outlier_comparison(series, "measure")
    assert result["iqr_flag_count"] >= 1
    assert result["zscore_flag_count"] >= 1


def test_outlier_comparison_reports_quartiles_and_fences():
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = outlier_comparison(series, "x")
    assert result["q1"] < result["median"] < result["q3"]
    assert result["iqr"] == pytest.approx(result["q3"] - result["q1"])
    assert result["iqr_lower_fence"] == pytest.approx(result["q1"] - 1.5 * result["iqr"])
    assert result["iqr_upper_fence"] == pytest.approx(result["q3"] + 1.5 * result["iqr"])


def test_outlier_comparison_constant_column_has_no_zscore_flags():
    series = pd.Series([5.0] * 15)
    result = outlier_comparison(series, "const")
    assert result["non_missing_numeric_count"] == 15
    assert int(result["zscore_flag_count"]) == 0
    assert float(result["zscore_flag_pct"]) == 0.0


def test_outlier_comparison_handles_missing_and_empty():
    result = outlier_comparison(pd.Series([None, None], dtype=float), "empty")
    assert result["non_missing_numeric_count"] == 0
    assert int(result["iqr_flag_count"]) == 0
    assert int(result["zscore_flag_count"]) == 0
    mixed = outlier_comparison(pd.Series([8.0, 8.5, None, 9.0]), "mixed")
    assert mixed["non_missing_numeric_count"] == 3


def test_class_distribution_does_not_guess_unknown_positive_label():
    dist, rate = class_distribution(pd.Series(["Stay", "Stay", "Leave"]), positive_values=["yes", "true"])
    assert int(dist["count"].sum()) == 3
    assert rate is None


def test_class_distribution_positive_rate_when_value_matches():
    _, rate = class_distribution(pd.Series(["Stay", "Leave", "Leave", None]), positive_values=["leave"])
    assert round(rate, 2) == 66.67


def test_class_distribution_numeric_target():
    dist, rate = class_distribution(pd.Series([0, 0, 0, 1, 1]), positive_values=["1"])
    assert round(rate, 2) == 40.0
    assert int(dist["count"].sum()) == 5


def test_candidate_columns_safe_on_missing_config():
    assert candidate_columns(["a", "b"], None) == []
    assert candidate_columns(["a", "b"], []) == []
    assert candidate_columns(["a", "B"], ["b", "missing"]) == ["B"]


def test_resolve_target_column_first_match_and_none():
    df = pd.DataFrame({"left_within_12m": [0, 1], "other": [1, 2]})
    cfg = {"target_column_candidates": ["left_within_12m", "other"]}
    assert resolve_target_column(df, cfg) == "left_within_12m"
    assert resolve_target_column(df, {}) is None
    assert resolve_target_column(df, None) is None


def test_leakage_register_covers_every_column():
    cols = ["employee_id", "annual_fixed_pay", "exit_date", "left_within_12m", "custom_feature"]
    cfg = {
        "target_column_candidates": ["left_within_12m"],
        "leakage_candidate_columns": ["exit_date"],
        "id_candidate_columns": ["employee_id"],
    }
    reg = build_leakage_register(cols, cfg, target_column="left_within_12m")
    assert list(reg["column"]) == cols
    assert set(reg.columns) == {"column", "suspected_mechanism", "risk_level", "exclude_from_model", "reason", "confirmation_needed"}
    by_col = reg.set_index("column")
    assert "Target" in str(by_col.loc["left_within_12m", "risk_level"])
    assert "Exclude" in str(by_col.loc["exit_date", "exclude_from_model"])
    assert "Exclude" in str(by_col.loc["employee_id", "exclude_from_model"])


def test_leakage_register_safe_with_empty_config():
    reg = build_leakage_register(["a", "b"], None, target_column=None)
    assert len(reg) == 2
    assert (reg["risk_level"] == "No evident leakage risk").all()


def test_duplicate_summary_counts():
    df = pd.DataFrame({"employee_id": ["e1", "e1", "e2"], "v": [1, 2, 3]})
    out = duplicate_summary(df, ["employee_id"]).set_index("column")
    assert int(out.loc["<all columns>", "count"]) == 0
    assert int(out.loc["employee_id", "count"]) == 1


def test_column_profile_roles_are_provisional():
    df = pd.DataFrame(
        {
            "employee_id": ["a", "b"],
            "left_within_12m": [0, 1],
            "annual_fixed_pay": [100, 200],
            "note": ["x", "y"],
        }
    )
    cfg = {
        "target_column_candidates": ["left_within_12m"],
        "leakage_candidate_columns": ["exit_date"],
        "id_candidate_columns": ["employee_id"],
    }
    prof = build_column_profile(df, cfg).set_index("column")
    assert "identifier" in str(prof.loc["employee_id", "role_provisional"]).lower()
    assert "target" in str(prof.loc["left_within_12m", "role_provisional"]).lower()
    assert infer_column_role("mystery_col", "object", cfg).startswith("categorical") or "provisional" in infer_column_role("mystery_col", "object", cfg)


def test_artifact_generation_with_synthetic_data(tmp_path: Path):
    """End-to-end CSV/figure generation on a small synthetic workbook."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "employee_id": ["E1", "E2", "E3", "E4"],
            "annual_fixed_pay": [300000, 350000, 400000, 5000000],
            "avg_hours_worked_3m": [8.5, 8.7, 8.6, 1.8],
            "tenure_months": [12, 24, 36, 200],
            "left_within_12m": [0, 0, 1, 0],
            "exit_date": [pd.NaT, pd.NaT, pd.Timestamp("2025-01-01"), pd.NaT],
        }
    )
    xlsx = tmp_path / "synthetic.xlsx"
    df.to_excel(xlsx, sheet_name="attrition_dataset", index=False)
    loaded = pd.read_excel(xlsx, sheet_name="attrition_dataset")
    assert len(loaded) == 4

    out = tmp_path / "out"
    out.mkdir()
    (figdir := out / "figures").mkdir()

    miss = missingness_summary(loaded)
    miss.to_csv(out / "missing_values.csv", index=False)
    prof = build_column_profile(loaded, {"target_column_candidates": ["left_within_12m"], "id_candidate_columns": ["employee_id"], "leakage_candidate_columns": ["exit_date"]})
    prof.to_csv(out / "column_profile.csv", index=False)
    reg = build_leakage_register(list(loaded.columns), {"target_column_candidates": ["left_within_12m"], "id_candidate_columns": ["employee_id"], "leakage_candidate_columns": ["exit_date"]}, target_column="left_within_12m")
    reg.to_csv(out / "leakage_register.csv", index=False)
    table, rate = class_distribution(loaded["left_within_12m"], ["1"])
    table.to_csv(out / "target_distribution.csv", index=False)
    assert rate == pytest.approx(25.0)

    fig, ax = plt.subplots()
    ax.bar(table["class_value"].astype(str), table["count"])
    fig.savefig(figdir / "target_distribution.png", dpi=80)
    plt.close(fig)

    for name in ["missing_values.csv", "column_profile.csv", "leakage_register.csv", "target_distribution.csv"]:
        assert (out / name).exists()
    assert len(reg) == len(loaded.columns)
    assert (figdir / "target_distribution.png").exists()


def test_outlier_details_covers_union_of_both_methods():
    series = pd.Series([10.0] * 20 + [1000.0])
    df = pd.DataFrame({"row_id": range(21), "pay": series})
    comp = outlier_comparison(series, "pay")
    det = outlier_details(df, ["pay"])
    assert len(det) == comp["iqr_flag_count"] + comp["zscore_flag_count"] - int(
        ((det["iqr_flagged"]) & (det["zscore_flagged"])).sum()
    )
    assert set(det.columns) >= {"row_ref", "column", "value", "iqr_flagged", "zscore_flagged", "direction", "business_range_violation", "finding_status"}
    assert "employee_id" not in det.columns


def test_outlier_details_marks_business_range_violation():
    df = pd.DataFrame({"row_id": [1, 2, 3, 4, 5, 6], "tenure_months": [10, 12, 11, 13, 12, -5]})
    det = outlier_details(df, ["tenure_months"])
    flagged = det[det["value"] == -5]
    assert len(flagged) == 1
    assert bool(flagged.iloc[0]["business_range_violation"]) is True
    assert flagged.iloc[0]["finding_status"] == "suspected data error"


def test_outlier_details_skips_missing_columns_and_empty():
    df = pd.DataFrame({"row_id": [1, 2], "v": [1.0, 2.0]})
    assert outlier_details(df, ["nope"]).empty
    assert outlier_details(df.iloc[0:0], ["v"]).empty


def test_issue_detail_records_match_summary_counts():
    df = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "employee_id": ["e1", "e2", "e3", "e4"],
            "tenure_months": [5, -2, 8, 9],
            "attendance_rate_3m": [0.9, 0.8, 1.1, 0.95],
            "work_location": ["Bengaluru", "Madras", "Chennai", "Pune"],
        }
    )
    det = issue_detail_records(df)
    assert "employee_id" not in det.columns
    assert int((det["column"] == "tenure_months").sum()) == 1
    assert int((det["column"] == "attendance_rate_3m").sum()) == 1
    assert int((det["column"] == "work_location").sum()) == 1
    assert set(det["row_ref"]) <= {1, 2, 3, 4}
