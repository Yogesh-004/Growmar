"""Isolation Forest construction, scoring, persistence and inference."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline


def build_iforest(
    *,
    n_estimators: int = 200,
    max_samples: int = 256,
    max_features: float = 1.0,
    random_state: int = 42,
) -> IsolationForest:
    """Reproducible Isolation Forest (contamination left at 'auto').

    The contamination setting only affects predict(); this workflow thresholds
    the documented anomaly_score instead, so contamination is never a hidden
    operating point.
    """
    return IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        random_state=random_state,
        contamination="auto",
    )


def build_detector(preprocessor, **iforest_kwargs) -> Pipeline:
    """Full detector: preprocessing step named 'prep', forest named 'iforest'."""
    return Pipeline([("prep", preprocessor), ("iforest", build_iforest(**iforest_kwargs))])


def save_detector(pipeline: Pipeline, path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, dest)
    return dest


def load_detector(path: str | Path) -> Pipeline:
    pipeline = joblib.load(Path(path))
    if not isinstance(pipeline, Pipeline):
        raise ValueError(f"Artifact {path} is not a scikit-learn Pipeline.")
    return pipeline


def score_frame(pipeline: Pipeline, X: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """Score rows without refitting: anomaly_score, threshold, flag.

    Alignment keys must be joined by the caller; they never enter the matrix.
    """
    from anomaly.features import anomaly_scores

    scores = anomaly_scores(pipeline, X)
    return pd.DataFrame({
        "anomaly_score": scores,
        "threshold": float(threshold),
        "is_anomaly": (scores >= threshold).astype(int),
    })
