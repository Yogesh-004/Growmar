"""Tests for Task 4 anomaly detection (synthetic fixtures; never the answer key)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from anomaly import evaluation as ev
from anomaly import injection as inj
from anomaly.features import anomaly_scores, attendance_feature_columns, build_preprocessor


def _monthly_frame(n_emp: int = 12, months: int = 6, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    starts = pd.date_range("2025-04-01", periods=months, freq="MS")
    rows = []
    for e in range(n_emp):
        for m in starts:
            rows.append({
                "attendance_feature_id": f"F{e:03d}-{m:%Y%m}",
                "employee_id": f"QE-{e:03d}",
                "month": m,
                "scheduled_days": 22,
                "present_days": 20,
                "wfh_days": 1,
                "half_days_unpaid": 0,
                "paid_leave_days": 1,
                "unpaid_absence_days": 1,
                "longest_unpaid_streak_days": 1,
                "late_arrivals": int(rng.integers(0, 4)),
                "avg_hours_worked": round(float(rng.normal(8.6, 0.3)), 2),
                "overtime_hours": round(float(rng.uniform(0, 5)), 1),
                "attendance_rate": 0.91,
            })
    return pd.DataFrame(rows)


def test_injection_repeatable_same_seed():
    pool = _monthly_frame()
    a, m1 = inj.inject_pool(pool, set(), 3, seed=42)
    b, m2 = inj.inject_pool(pool, set(), 3, seed=42)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(m1, m2)
    c, _ = inj.inject_pool(pool, set(), 3, seed=43)
    assert not a["avg_hours_worked"].equals(c["avg_hours_worked"])


def test_each_category_changes_its_feature():
    pool = _monthly_frame()
    out, manifest = inj.inject_pool(pool, set(), 4, seed=7)
    by_cat = manifest.groupby("category")["field"].apply(set).to_dict()
    assert "avg_hours_worked" in by_cat["impossible_hours"]
    assert {"longest_unpaid_streak_days", "attendance_rate"} <= by_cat["long_absence"]
    assert "attendance_rate" in by_cat["sudden_change"]
    hours_rows = manifest.loc[manifest["category"] == "impossible_hours", "attendance_feature_id"]
    changed = out.set_index("attendance_feature_id").loc[hours_rows, "avg_hours_worked"]
    assert bool(((changed >= 19.0) & (changed <= 22.0)).all())
    assert (manifest["category"].value_counts() > 0).all()


def test_original_input_unchanged():
    pool = _monthly_frame()
    before = pool.copy(deep=True)
    inj.inject_pool(pool, set(), 3, seed=5)
    pd.testing.assert_frame_equal(pool, before)


def test_labels_never_in_feature_columns():
    cols = attendance_feature_columns()
    for banned in ("employee_id", "is_injected", "left_within_12m", "exit_date", "category"):
        assert banned not in cols


def test_score_direction_higher_means_worse():
    from anomaly.model import build_detector

    rng = np.random.default_rng(0)
    normal = pd.DataFrame({c: rng.normal(0, 1, 300) for c in ["f1", "f2"]})
    prep = ColumnTransformer(
        [("n", Pipeline([("i", SimpleImputer(strategy="median")),
                         ("s", RobustScaler())]), ["f1", "f2"])],
        verbose_feature_names_out=False)
    pipe = build_detector(prep, n_estimators=50, random_state=0)
    pipe.fit(normal)
    clear = pd.DataFrame({"f1": [0.0, 50.0], "f2": [0.0, 50.0]})
    scores = anomaly_scores(pipe, clear)
    assert scores[1] > scores[0]


def test_metric_formulas_and_zero_denominators():
    m = ev.threshold_metrics(np.array([0.9, 0.1, 0.2]), np.array([True, False, False]), 0.5)
    assert m["n_caught"] == 1 and m["n_missed"] == 0
    assert m["catch_rate"] == pytest.approx(1.0)
    assert m["false_alarm_rate"] == pytest.approx(0.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    empty = ev.threshold_metrics(np.array([0.1]), np.array([False]), 0.5)
    assert empty["n_flagged"] == 0
    assert np.isnan(empty["precision"]) and np.isnan(empty["f1"])  # never fabricated
    # catch rate with zero injected anomalies is NaN, not 0 or 1
    assert np.isnan(ev.threshold_metrics(np.array([0.9]), np.array([False]), 0.5)["catch_rate"])
    assert np.isnan(ev.safe_divide(0, 0))


def test_selection_obeys_false_alarm_constraint():
    table = pd.DataFrame([
        {"threshold": 0.9, "catch_rate": 0.9, "precision": 0.2, "false_alarm_rate": 0.30, "flagged_pct": 25.0},
        {"threshold": 0.5, "catch_rate": 0.6, "precision": 0.5, "false_alarm_rate": 0.05, "flagged_pct": 8.0},
        {"threshold": 0.7, "catch_rate": 0.6, "precision": 0.7, "false_alarm_rate": 0.04, "flagged_pct": 6.0},
    ])
    sel = ev.select_threshold(table, max_false_alarm=0.10, max_flagged_fraction=0.20)
    assert sel["threshold"] == pytest.approx(0.7)  # tie on catch broken by precision
    assert sel["fallback"] is False
    bad = table.assign(false_alarm_rate=[0.5, 0.4, 0.35])
    sel2 = ev.select_threshold(bad, max_false_alarm=0.10, max_flagged_fraction=0.20)
    assert sel2["fallback"] is True  # fallback declared, not concealed


def test_final_labels_cannot_move_threshold():
    table = pd.DataFrame([
        {"threshold": 0.9, "catch_rate": 0.9, "precision": 0.2, "false_alarm_rate": 0.05, "flagged_pct": 8.0},
        {"threshold": 0.5, "catch_rate": 0.5, "precision": 0.6, "false_alarm_rate": 0.04, "flagged_pct": 6.0},
    ])
    frozen = ev.select_threshold(table, max_false_alarm=0.10, max_flagged_fraction=0.20)["threshold"]
    assert frozen == pytest.approx(0.9)  # decided from this table alone
    # a different final table cannot change the frozen value (no API for it)


def test_fit_save_load_score_schema(tmp_path):
    from anomaly.model import build_detector, load_detector, save_detector, score_frame

    df = _monthly_frame()
    feats = [c for c in attendance_feature_columns() if c in df.columns]
    X = df[feats].copy()
    X["late_arrivals_avg_3m"] = X["late_arrivals"].astype(float)
    X["overtime_hours_avg_3m"] = X["overtime_hours"].astype(float)
    X["attendance_rate_avg_3m"] = X["attendance_rate"].astype(float)
    X["months_of_history"] = 3
    X["attendance_invalid"] = 0
    pipe = build_detector(build_preprocessor(), n_estimators=20, random_state=0)
    pipe.fit(X)
    path = save_detector(pipe, tmp_path / "m.joblib")
    reloaded = load_detector(path)
    a = score_frame(pipe, X, threshold=0.5)
    b = score_frame(reloaded, X, threshold=0.5)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["anomaly_score", "threshold", "is_anomaly"]
    assert set(a["is_anomaly"].unique()) <= {0, 1}


def test_chronological_windows_use_no_future():
    from anomaly.features import split_periods

    df = _monthly_frame(months=8)
    parts = split_periods(df, "2025-04", "2025-08", "2025-10")
    assert parts["train"]["month"].max() < parts["calibration"]["month"].min()
    assert parts["calibration"]["month"].max() < parts["final"]["month"].min()
    assert len(parts["train"]) + len(parts["calibration"]) + len(parts["final"]) == len(df)


def test_report_and_artifact_generation(tmp_path):
    from anomaly.evaluation import pr_points, threshold_table, thresholds_from_flag_rates

    scores = np.array([0.1, 0.2, 0.8, 0.9])
    truth = np.array([False, False, True, True])
    thresholds = thresholds_from_flag_rates(scores, [0.25, 0.5])
    table = threshold_table(scores, truth, thresholds)
    assert len(table) == 2
    pts = pr_points(table)
    assert list(pts.columns) == ["threshold", "catch_rate", "precision", "flagged_pct"]
    table.to_csv(tmp_path / "threshold_comparison.csv", index=False)
    assert (tmp_path / "threshold_comparison.csv").exists()
