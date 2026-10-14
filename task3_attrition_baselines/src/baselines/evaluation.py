"""Fold-safe cross-validation, aggregation, selection and holdout evaluation."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

METRICS = ("precision", "recall", "f1", "roc_auc", "pr_auc")


def _auc_scores(estimator: object, X_valid: pd.DataFrame, y_valid: pd.Series) -> tuple[float, float]:
    """ROC-AUC and PR-AUC from probabilities; NaN when undefined, documented."""
    if len(np.unique(y_valid)) < 2:
        return float("nan"), float("nan")
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(X_valid)[:, 1]  # type: ignore[union-attr]
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X_valid)  # type: ignore[union-attr]
    else:
        scores = estimator.predict(X_valid)  # type: ignore[union-attr]
    try:
        return float(roc_auc_score(y_valid, scores)), float(average_precision_score(y_valid, scores))
    except ValueError:
        return float("nan"), float("nan")


def run_cv(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    prep_factory: Callable[[], Pipeline],
    candidates: dict[str, object],
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Stratified CV on training rows only; preprocessing refit inside each fold.

    Each fold builds a fresh ``Pipeline([prep_factory(), clone(estimator)])``,
    so imputers, scalers, encoders and department references learn exclusively
    from that fold's training rows. Returns one row per candidate per fold.
    """
    if n_splits < 2:
        raise ValueError("Cross-validation needs at least two folds.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None)
    records: list[dict[str, object]] = []
    y = pd.Series(y_train).reset_index(drop=True)
    X = X_train_raw.reset_index(drop=True)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        for name, estimator in candidates.items():
            pipe = Pipeline([("prep", prep_factory()), ("clf", clone(estimator))])
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_va)
            roc_auc, pr_auc = _auc_scores(pipe, X_va, y_va)
            records.append(
                {
                    "candidate": name,
                    "fold": fold,
                    "n_train": len(X_tr),
                    "n_valid": len(X_va),
                    "precision": float(precision_score(y_va, pred, zero_division=0)),
                    "recall": float(recall_score(y_va, pred, zero_division=0)),
                    "f1": float(f1_score(y_va, pred, zero_division=0)),
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                }
            )
    return pd.DataFrame(records)


def summarize_cv(fold_records: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample standard deviation per candidate and metric."""
    if fold_records.empty:
        raise ValueError("No fold records to summarize.")
    rows = []
    for name, group in fold_records.groupby("candidate"):
        row: dict[str, object] = {"candidate": name, "folds": len(group)}
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    cols = ["candidate", "folds"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    return pd.DataFrame(rows, columns=cols).sort_values("candidate").reset_index(drop=True)


def select_candidate(summary: pd.DataFrame, primary: str = "recall_mean", tiebreak: str = "f1_mean") -> dict[str, object]:
    """Predeclared rule: highest primary mean, then highest tiebreak mean.

    Operates on cross-validation results only; the test set is never consulted.
    """
    for column in (primary, tiebreak):
        if column not in summary.columns:
            raise ValueError(f"Selection metric {column!r} is not in the CV summary.")
    ordered = summary.sort_values([primary, tiebreak, "candidate"], ascending=[False, False, True])
    winner = ordered.iloc[0]
    return {
        "selected": str(winner["candidate"]),
        "rule": f"max {primary}, tie-break max {tiebreak} (CV only, test unseen)",
        primary: float(winner[primary]),
        tiebreak: float(winner[tiebreak]),
    }


def fit_finalists(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    prep_factory: Callable[[], Pipeline],
    candidates: dict[str, object],
) -> dict[str, Pipeline]:
    """Fit every candidate on the complete training partition (one fresh prep each)."""
    fitted = {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("prep", prep_factory()), ("clf", clone(estimator))])
        pipe.fit(X_train_raw, y_train)
        fitted[name] = pipe
    return fitted


def evaluate_holdout(
    fitted: Pipeline, X_test_raw: pd.DataFrame, y_test: pd.Series, *, threshold: float = 0.5
) -> dict[str, object]:
    """Final held-out metrics at a fixed threshold (default 0.5, never tuned on test)."""
    if hasattr(fitted, "predict_proba"):
        proba = fitted.predict_proba(X_test_raw)[:, 1]
    elif hasattr(fitted, "decision_function"):
        proba = fitted.decision_function(X_test_raw)
    else:  # pragma: no cover - all current candidates expose probabilities
        proba = fitted.predict(X_test_raw).astype(float)
    pred = (np.asarray(proba) >= threshold).astype(int)
    y = pd.Series(y_test).reset_index(drop=True)
    roc_auc, pr_auc = _auc_scores(fitted, X_test_raw, y)
    return {
        "threshold": float(threshold),
        "accuracy": float((pred == y.to_numpy()).mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "y_true": y.to_numpy(),
        "y_proba": np.asarray(proba, dtype=float),
    }


def roc_points(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ROC curve points for plotting (fitted scores, fixed data)."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return np.asarray(fpr), np.asarray(tpr)
