"""Tests for Task 3 baselines (synthetic fixtures; never the answer key)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baselines import evaluation as E
from baselines import models as M


def _toy(n: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(((X["a"] + rng.normal(scale=0.5, size=n)) > 0.5).astype(int))
    return X, y


def _passthrough():
    from sklearn.preprocessing import FunctionTransformer

    return FunctionTransformer(validate=False)


def test_target_encoding_rejects_nonbinary():
    from baselines import data as D

    bad = pd.DataFrame({"left_within_12m": [0, 2, 1], "exit_date": [pd.NaT] * 3})
    with pytest.raises(Exception):
        D.verify_target_encoding(bad, "left_within_12m")


def test_dummy_always_predicts_stays():
    X, y = _toy()
    clf = DummyClassifier(strategy="constant", constant=0)
    clf.fit(X, y)
    assert set(np.unique(clf.predict(X))) == {0}
    res = E.evaluate_holdout(clf, X, y)
    assert res["recall"] == 0.0 and res["precision"] == 0.0 and res["f1"] == 0.0
    assert res["roc_auc"] == pytest.approx(0.5)  # constant scores: uninformative


def test_candidate_construction_and_shapes():
    cands = M.build_candidates(random_state=7)
    assert set(cands) == {"dummy_stays", "logreg", "logreg_balanced", "tree", "tree_balanced"}
    X, y = _toy()
    for name, est in cands.items():
        pipe = Pipeline([("prep", _passthrough()), ("clf", est)])
        pipe.fit(X, y)
        assert pipe.predict(X).shape == (len(X),)
        assert pipe.predict_proba(X).shape == (len(X), 2)


def test_cv_uses_fresh_preprocessing_per_fold():
    seen = []

    def factory():
        from sklearn.preprocessing import StandardScaler

        class Rec(StandardScaler):
            def fit(self, X, y=None):
                seen.append(tuple(sorted(pd.DataFrame(X).iloc[:, 0].round(6).tolist())))
                return super().fit(X, y)

        return Rec()

    X, y = _toy()
    cands = {"logreg": LogisticRegression(max_iter=1000)}
    folds = E.run_cv(X, y, factory, cands, n_splits=3, random_state=1)
    assert len(folds) == 3
    assert len({s for s in seen}) == 3  # one distinct fit per fold
    assert set(folds.columns) >= {"candidate", "fold", "precision", "recall", "f1", "roc_auc"}


def test_cv_aggregation_means_and_stds():
    recs = pd.DataFrame([
        {"candidate": "m", "fold": 0, "precision": 1.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.5, "pr_auc": 0.2},
        {"candidate": "m", "fold": 1, "precision": 0.0, "recall": 1.0, "f1": 0.5, "roc_auc": 0.7, "pr_auc": 0.4},
    ])
    s = E.summarize_cv(recs)
    assert s.loc[0, "recall_mean"] == pytest.approx(0.5)
    assert s.loc[0, "recall_std"] == pytest.approx(np.std([0.0, 1.0], ddof=1))
    assert s.loc[0, "folds"] == 2


def test_selection_prefers_recall_then_f1_without_test():
    s = pd.DataFrame([
        {"candidate": "a", "recall_mean": 0.6, "f1_mean": 0.3},
        {"candidate": "b", "recall_mean": 0.6, "f1_mean": 0.5},
        {"candidate": "c", "recall_mean": 0.4, "f1_mean": 0.9},
    ])
    out = E.select_candidate(s)
    assert out["selected"] == "b"  # tie on recall broken by F1, test never consulted
    with pytest.raises(ValueError):
        E.select_candidate(s, primary="nope")


def test_feature_order_and_exclusions_preserved():
    from features.schema import build_supervised_frame

    rng = np.random.default_rng(0)
    n = 40
    raw = pd.DataFrame({
        "row_id": range(n),
        "employee_id": [f"E{i}" for i in range(n)],
        "job_level": rng.integers(1, 4, n),
        "tenure_months": rng.integers(0, 50, n).astype(float),
        "total_experience_years": rng.uniform(0, 10, n),
        "previous_employers_count": rng.integers(0, 3, n),
        "commute_distance_km": rng.uniform(1, 20, n),
        "annual_fixed_pay": rng.integers(200000, 900000, n).astype(float),
        "comp_ratio": rng.uniform(0.8, 1.2, n),
        "raises_received": rng.integers(0, 3, n),
        "last_raise_pct": rng.uniform(2, 10, n),
        "months_since_last_raise": rng.integers(0, 20, n),
        "promotions_received": rng.integers(0, 2, n),
        "months_since_last_promotion": rng.uniform(3, 40, n),
        "internal_transfers": rng.integers(0, 2, n),
        "last_performance_rating": rng.integers(1, 6, n).astype(float),
        "last_engagement_score": rng.uniform(1, 5, n),
        "manager_changes_12m": rng.integers(0, 3, n),
        "training_hours_12m": rng.uniform(3, 60, n),
        "attendance_rate_3m": rng.uniform(0.8, 1.0, n),
        "late_arrivals_3m": rng.integers(0, 10, n),
        "avg_overtime_hours_per_month_3m": rng.uniform(0, 10, n),
        "avg_hours_worked_3m": rng.uniform(8, 9, n),
        "wfh_share_3m": rng.uniform(0, 1, n),
        "unpaid_leave_days_6m": rng.integers(0, 4, n).astype(float),
        "paid_leave_days_6m": rng.integers(0, 8, n),
        "dept_attrition_rate_prior_12m": rng.uniform(0, 0.2, n),
        "department_id": rng.choice(["D01", "D02"], n),
        "role_id": rng.choice(["R1", "R2"], n),
        "work_location": rng.choice(["Bengaluru", "Chennai"], n),
        "work_mode": rng.choice(["Hybrid", "Onsite"], n),
        "employment_type": "Permanent",
        "highest_education": rng.choice(["B.Tech", "MBA"], n),
        "left_within_12m": (rng.random(n) < 0.3).astype(int),
        "exit_date": pd.NaT,
    })
    X, y, keys = build_supervised_frame(raw, "left_within_12m")
    for col in ("employee_id", "row_id", "left_within_12m", "exit_date"):
        assert col not in X.columns
    from features.schema import expected_input_columns

    assert list(X.columns) == expected_input_columns()  # stable deterministic order


def test_proba_used_for_auc_and_persistence(tmp_path):
    X, y = _toy(n=200)
    X["a"] = y * 10.0  # perfect separator
    pipe = Pipeline([("prep", _passthrough()), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)
    res = E.evaluate_holdout(pipe, X, y)
    assert res["roc_auc"] == pytest.approx(1.0)
    import joblib

    p = tmp_path / "m.joblib"
    joblib.dump(pipe, p)
    reloaded = joblib.load(p)
    pd.testing.assert_series_equal(
        pd.Series(pipe.predict(X)), pd.Series(reloaded.predict(X)), check_names=False
    )


def test_split_reconstruction_matches_manifest(tmp_path):
    from baselines import data as D

    raw = pd.DataFrame({
        "row_id": [10, 20, 30, 40, 50, 60],
        "annual_fixed_pay": [100.0] * 6,
        "comp_ratio": [1.0] * 6,
        "job_level": [1] * 6,
        "left_within_12m": [0, 0, 0, 1, 1, 0],
        "exit_date": [pd.NaT, pd.NaT, pd.NaT, pd.Timestamp("2024-01-01"),
                      pd.Timestamp("2024-01-02"), pd.NaT],
    })
    for col in ["department_id", "role_id", "work_location", "work_mode", "employment_type",
                "highest_education", "tenure_months", "total_experience_years",
                "previous_employers_count", "commute_distance_km", "raises_received",
                "last_raise_pct", "months_since_last_raise", "promotions_received",
                "months_since_last_promotion", "internal_transfers", "last_performance_rating",
                "last_engagement_score", "manager_changes_12m", "training_hours_12m",
                "attendance_rate_3m", "late_arrivals_3m", "avg_overtime_hours_per_month_3m",
                "avg_hours_worked_3m", "wfh_share_3m", "unpaid_leave_days_6m",
                "paid_leave_days_6m", "dept_attrition_rate_prior_12m"]:
        raw[col] = 0 if col not in ("department_id", "role_id", "work_location", "work_mode",
                                    "employment_type", "highest_education") else "X"
    manifest = {"train_row_ids": [10, 20, 30, 40], "test_row_ids": [50, 60],
                "random_state": 1, "test_size": 0.25,
                "train_target_rate": 0.25, "test_target_rate": 0.5}
    Xtr, ytr, Xte, yte, check = D.reconstruct_split(raw, manifest, "left_within_12m")
    assert (check["train_rows"], check["test_rows"]) == (4, 2)
    assert D.verify_target_encoding(raw, "left_within_12m")["leavers"] == 2
    bad = dict(manifest, train_row_ids=[10, 20])
    with pytest.raises(Exception):
        D.reconstruct_split(raw, bad, "left_within_12m")
