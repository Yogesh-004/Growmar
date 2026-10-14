"""Train/holdout splitting with a reusable manifest (no employee IDs stored)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    keys: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    """Stratified split on the attrition target; returns aligned partitions."""
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")
    if y.nunique(dropna=False) < 2:
        raise ValueError("Stratified split needs at least two target classes.")
    X_train, X_test, y_train, y_test, k_train, k_test = train_test_split(
        X, y, keys, test_size=test_size, random_state=random_state, stratify=y
    )
    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "keys_train": k_train.reset_index(drop=True),
        "keys_test": k_test.reset_index(drop=True),
    }


def split_manifest(
    *,
    version: str,
    test_size: float,
    random_state: int,
    keys_train: pd.DataFrame,
    keys_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Manifest Task 3 can reuse instead of generating another test set."""
    return {
        "feature_package_version": version,
        "strategy": "stratified split on the supervised attrition target",
        "test_size": test_size,
        "random_state": random_state,
        "train_rows": int(len(keys_train)),
        "test_rows": int(len(keys_test)),
        "train_target_rate": float(y_train.mean()),
        "test_target_rate": float(y_test.mean()),
        "train_row_ids": [str(v) for v in keys_train["row_id"].tolist()],
        "test_row_ids": [str(v) for v in keys_test["row_id"].tolist()],
        "note": "Row alignment uses the technical row_id only; no employee IDs stored.",
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dest
