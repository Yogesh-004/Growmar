"""Unit tests for the Task 2 feature package (synthetic fixtures only)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features import FEATURES_VERSION
from features import engineering, pipeline as pipeline_mod
from features import rules, schema
from features.config import Task2Config
from features.split import split_manifest, stratified_split
from features.transformers import DeptMedianRatio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
N_NUM = len(rules.NUMERIC_FEATURES)
N_CAT = len(rules.CATEGORICAL_FEATURES)


def synthetic_frame(n: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    depts = np.array(["D01", "D02", "D03"])
    locs = np.array(["Bengaluru", "Chennai", "Pune", "BENGALURU"])
    df = pd.DataFrame(
        {
            "row_id": range(1, n + 1),
            "employee_id": [f"SYN-{i:04d}" for i in range(1, n + 1)],
            "job_level": rng.integers(1, 5, n),
            "tenure_months": rng.integers(0, 100, n).astype(float),
            "total_experience_years": rng.uniform(0, 20, n).round(1),
            "previous_employers_count": rng.integers(0, 5, n),
            "commute_distance_km": rng.uniform(1, 40, n).round(1),
            "annual_fixed_pay": rng.integers(200000, 2000000, n).astype(float),
            "comp_ratio": rng.uniform(0.7, 1.3, n).round(3),
            "raises_received": rng.integers(0, 4, n),
            "last_raise_pct": rng.uniform(2, 15, n).round(1),
            "months_since_last_raise": rng.integers(0, 30, n),
            "promotions_received": rng.integers(0, 3, n),
            "months_since_last_promotion": rng.uniform(3, 60, n).round(0),
            "internal_transfers": rng.integers(0, 3, n),
            "last_performance_rating": rng.integers(1, 6, n).astype(float),
            "last_engagement_score": rng.uniform(1, 5, n).round(1),
            "manager_changes_12m": rng.integers(0, 4, n),
            "training_hours_12m": rng.uniform(3, 90, n).round(1),
            "attendance_rate_3m": rng.uniform(0.7, 1.0, n).round(4),
            "late_arrivals_3m": rng.integers(0, 15, n),
            "avg_overtime_hours_per_month_3m": rng.uniform(0, 20, n).round(1),
            "avg_hours_worked_3m": rng.uniform(7.5, 9.5, n).round(2),
            "wfh_share_3m": rng.uniform(0, 1, n).round(3),
            "unpaid_leave_days_6m": rng.integers(0, 6, n).astype(float),
            "paid_leave_days_6m": rng.integers(0, 12, n),
            "dept_attrition_rate_prior_12m": rng.uniform(0, 0.3, n).round(4),
            "department_id": rng.choice(depts, n),
            "role_id": rng.choice(["R001", "R002", "R003"], n),
            "work_location": rng.choice(locs, n),
            "work_mode": rng.choice(["Hybrid", "Onsite", "Remote"], n),
            "employment_type": rng.choice(["Permanent", "Fixed-Term Contract"], n),
            "highest_education": rng.choice(["B.Tech", "MBA", "B.Com"], n),
            "left_within_12m": (rng.random(n) < 0.2).astype(int),
        }
    )
    df.loc[df["promotions_received"] == 0, "months_since_last_promotion"] = np.nan
    df.loc[df["raises_received"] == 0, "last_raise_pct"] = np.nan
    return df


def test_fit_save_reload_transform_roundtrip(tmp_path: Path):
    df = synthetic_frame()
    X, y, keys = schema.build_supervised_frame(df, "left_within_12m")
    assert list(X.columns) == schema.expected_input_columns()
    assert "employee_id" not in X.columns and "left_within_12m" not in X.columns
    parts = stratified_split(X, y, keys, test_size=0.2, random_state=42)
    assert len(parts["X_train"]) + len(parts["X_test"]) == len(X)  # no row multiplication
    pipe = pipeline_mod.build_pipeline()
    pipe.fit(parts["X_train"])
    cols = list(pipeline_mod.transform_frame(pipe, parts["X_train"]).columns)
    from features import persistence

    meta = persistence.metadata_record(
        version=FEATURES_VERSION, pipeline_filename="pipeline.joblib",
        source_sheets=["attrition_dataset"], input_columns=pipeline_mod.input_columns(),
        output_columns=cols, train_config={"seed": 42}, dept_references={},
        assumptions=["test"],
    )
    paths = persistence.save_artifact(pipe, meta, tmp_path / "art")
    assert paths["pipeline"].exists() and paths["metadata"].exists()
    pipe2, meta2 = persistence.load_artifact(tmp_path / "art")
    assert meta2["feature_package_version"] == FEATURES_VERSION
    a = pipeline_mod.transform_frame(pipe, parts["X_test"], cols)
    b = pipeline_mod.transform_frame(pipe2, parts["X_test"], cols)
    pd.testing.assert_frame_equal(a, b)  # reload reproduces exactly
    c = pipeline_mod.transform_frame(pipe, parts["X_test"], cols)
    pd.testing.assert_frame_equal(a, c)  # deterministic


def test_output_has_no_missing_and_stable_order():
    df = synthetic_frame()
    df.loc[0, "commute_distance_km"] = np.nan
    df.loc[1, "highest_education"] = None
    X, y, keys = schema.build_supervised_frame(df, "left_within_12m")
    parts = stratified_split(X, y, keys, test_size=0.25, random_state=1)
    pipe = pipeline_mod.build_pipeline().fit(parts["X_train"])
    out = pipeline_mod.transform_frame(pipe, parts["X_test"])
    assert int(out.isna().sum().sum()) == 0
    assert list(out.columns) == list(pipeline_mod.transform_frame(pipe, parts["X_train"]).columns)


def test_unseen_categories_do_not_crash():
    df = synthetic_frame()
    X, y, keys = schema.build_supervised_frame(df, "left_within_12m")
    parts = stratified_split(X, y, keys, test_size=0.2, random_state=3)
    pipe = pipeline_mod.build_pipeline().fit(parts["X_train"])
    new = parts["X_test"].copy()
    new.loc[new.index[0], "department_id"] = "D99"
    new.loc[new.index[1], "work_location"] = "Atlantis"
    out = pipeline_mod.transform_frame(pipe, new)
    assert out.shape[1] == pipeline_mod.transform_frame(pipe, parts["X_test"]).shape[1]


def test_structural_missingness_indicators():
    df = synthetic_frame()
    df.loc[0, "months_since_last_promotion"] = np.nan
    df.loc[0, "promotions_received"] = 0
    df.loc[1, "last_raise_pct"] = np.nan
    df.loc[1, "raises_received"] = 0
    eng = engineering.add_rule_based_features(schema.validate_feature_frame(df))
    assert int(eng.loc[0, "is_never_promoted"]) == 1
    assert int(eng.loc[1, "is_never_raised"]) == 1
    assert float(eng.loc[0, "months_since_last_promotion"]) == rules.STRUCTURAL_SENTINEL
    assert "BENGALURU" not in set(eng["work_location"].astype(str))  # canonicalized


def test_dept_ratio_formula_and_train_only():
    train = pd.DataFrame({
        "annual_fixed_pay": [100.0, 200.0, 300.0, 1000.0],
        "department_id": ["A", "A", "A", "B"],
    })
    t = DeptMedianRatio().fit(train)
    assert t.references() == {"A": 200.0, "B": 1000.0}
    test = pd.DataFrame({"annual_fixed_pay": [400.0, 500.0], "department_id": ["A", "ZZZ"]})
    out = t.transform(test)
    assert out["salary_vs_department_median"].tolist() == pytest.approx([2.0, 2.0])  # 500/250 global median
    # transform uses stored train medians even if test-only median differs
    assert t.transform(pd.DataFrame({"annual_fixed_pay": [200.0], "department_id": ["A"]}))[
        "salary_vs_department_median"].iloc[0] == pytest.approx(1.0)
    bad = pd.DataFrame({"annual_fixed_pay": [np.nan], "department_id": ["A"]})
    assert np.isnan(t.transform(bad)["salary_vs_department_median"].iloc[0])


def test_exclusions_and_schema_errors():
    df = synthetic_frame()
    X, _, _ = schema.build_supervised_frame(df, "left_within_12m")
    for col in ["employee_id", "row_id", "left_within_12m", "exit_date"]:
        assert col not in X.columns
    short = df.drop(columns=["tenure_months"])
    with pytest.raises(schema.SchemaError, match="tenure_months"):
        schema.validate_feature_frame(short)
    with pytest.raises(schema.SchemaError, match="absent"):
        schema.build_supervised_frame(df, "nope")
    with pytest.raises(ValueError, match="test_size"):
        stratified_split(X, df["left_within_12m"], pd.DataFrame({"row_id": [1]}), test_size=1.5, random_state=0)


def test_split_manifest_has_no_employee_ids():
    df = synthetic_frame()
    X, y, keys = schema.build_supervised_frame(df, "left_within_12m")
    parts = stratified_split(X, y, keys, test_size=0.2, random_state=42)
    man = split_manifest(version=FEATURES_VERSION, test_size=0.2, random_state=42,
                         keys_train=parts["keys_train"], keys_test=parts["keys_test"],
                         y_train=parts["y_train"], y_test=parts["y_test"])
    blob = json.dumps(man)
    assert "SYN-" not in blob
    assert man["train_rows"] + man["test_rows"] == len(X)
    assert abs(man["train_target_rate"] - man["test_target_rate"]) < 0.15


def test_config_version_and_missing_workbook(tmp_path: Path):
    cfg = Task2Config.from_json(PROJECT_ROOT / "task2_feature_pipeline" / "configs" / "task2_config.json")
    assert cfg.package_version == FEATURES_VERSION
    with pytest.raises(FileNotFoundError):
        schema.load_attrition_sheet(tmp_path / "missing.xlsx", "attrition_dataset")
    with pytest.raises(ValueError, match="version"):
        Task2Config(package_version="features-9.9.9")


def test_cli_fit_transform_on_synthetic_workbook(tmp_path: Path):
    df = synthetic_frame(120)
    xlsx = tmp_path / "synth.xlsx"
    df.to_excel(xlsx, sheet_name="attrition_dataset", index=False)
    cfg_path = tmp_path / "task2.json"
    cfg_path.write_text(json.dumps({
        "package_version": FEATURES_VERSION,
        "attrition_sheet": "attrition_dataset",
        "target_column": "left_within_12m",
        "random_state": 11, "test_size": 0.25,
        "artifacts_dir": str(tmp_path / "art"),
        "output_dir": str(tmp_path / "out"),
    }), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "features.cli", "--config", str(cfg_path),
         "fit-transform", "--workbook", str(xlsx)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (tmp_path / "art" / "pipeline.joblib").exists()
    assert (tmp_path / "out" / "train_features.csv").exists()
    assert (tmp_path / "out" / "test_labels.csv").exists()
    # transform with the saved artifact, no refit
    new_csv = tmp_path / "new.csv"
    df.drop(columns=["left_within_12m"]).head(10).to_csv(new_csv, index=False)
    proc2 = subprocess.run(
        [sys.executable, "-m", "features.cli", "--config", str(cfg_path),
         "transform", "--input", str(new_csv), "--artifacts-dir", str(tmp_path / "art"),
         "--output", str(tmp_path / "t.csv")],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    assert proc2.returncode == 0, proc2.stderr[-2000:]
    assert len(pd.read_csv(tmp_path / "t.csv")) == 10
