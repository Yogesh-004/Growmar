"""Split reuse: reconstruct Task 2's locked partition from its manifest."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class SplitError(ValueError):
    """Raised when the locked split cannot be reconstructed or verified."""


def load_manifest(path: str | Path) -> dict:
    import json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SplitError("Split manifest must be a JSON object.")
    for key in ("train_row_ids", "test_row_ids", "random_state", "test_size"):
        if key not in raw:
            raise SplitError(f"Split manifest lacks required key {key!r}.")
    return raw


def reconstruct_split(
    raw: pd.DataFrame,
    manifest: dict,
    target_column: str,
    id_column: str = "row_id",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict]:
    """Rebuild the exact locked train/test partition from manifest row IDs.

    Returns (X_train_raw, y_train, X_test_raw, y_test, check) where X frames
    hold raw (untransformed) feature columns and check records verification
    numbers. Raises SplitError when row IDs are missing or counts/rates drift
    from the manifest beyond a small tolerance.
    """
    from features.schema import build_supervised_frame

    if id_column not in raw.columns:
        raise SplitError(f"Source data lacks alignment column {id_column!r}.")
    train_ids = [str(v) for v in manifest["train_row_ids"]]
    test_ids = [str(v) for v in manifest["test_row_ids"]]
    if set(train_ids) & set(test_ids):
        raise SplitError("Manifest train/test row IDs overlap.")
    key = raw[id_column].astype(str)
    train_raw = raw.loc[key.isin(train_ids)].copy()
    test_raw = raw.loc[key.isin(test_ids)].copy()
    if len(train_raw) != len(train_ids) or len(test_raw) != len(test_ids):
        raise SplitError(
            f"Manifest alignment failed: found {len(train_raw)}/{len(train_ids)} train and "
            f"{len(test_raw)}/{len(test_ids)} test rows."
        )
    X_train, y_train, _ = build_supervised_frame(train_raw, target_column)
    X_test, y_test, _ = build_supervised_frame(test_raw, target_column)
    check = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_rate": float(y_train.mean()),
        "test_rate": float(y_test.mean()),
        "manifest_train_rate": float(manifest.get("train_target_rate", float("nan"))),
        "manifest_test_rate": float(manifest.get("test_target_rate", float("nan"))),
    }
    for observed, expected, name in (
        (check["train_rate"], check["manifest_train_rate"], "train"),
        (check["test_rate"], check["manifest_test_rate"], "test"),
    ):
        if abs(observed - expected) > 1e-9:
            raise SplitError(f"{name} target rate {observed} differs from manifest {expected}.")
    return X_train, y_train, X_test, y_test, check


def verify_label_files(manifest: dict, label_dir: str | Path, target_column: str) -> dict:
    """Cross-check Task 2 label files against the manifest (no refit involved)."""
    label_dir = Path(label_dir)
    frames = {}
    for name in ("train_labels.csv", "test_labels.csv"):
        path = label_dir / name
        if not path.exists():
            raise SplitError(f"Label file not found: {path}")
        frames[name] = pd.read_csv(path)
    result = {}
    for name, key in (("train_labels.csv", "train_row_ids"), ("test_labels.csv", "test_row_ids")):
        file_ids = {str(v) for v in frames[name]["row_id"].tolist()}
        manifest_ids = {str(v) for v in manifest[key]}
        result[name] = {"match": file_ids == manifest_ids, "rows": len(file_ids)}
        if file_ids != manifest_ids:
            raise SplitError(f"{name} row IDs do not match the manifest.")
    if target_column not in frames["train_labels.csv"].columns:
        raise SplitError(f"Label files lack target column {target_column!r}.")
    return result


def verify_target_encoding(raw: pd.DataFrame, target_column: str) -> dict:
    """Confirm 0 means stayed and 1 means left, using post-outcome presence.

    Leavers are exactly the rows carrying exit information; stayers carry none.
    Returns aggregate counts only.
    """
    labels = set(pd.to_numeric(raw[target_column], errors="coerce").dropna().unique().tolist())
    if labels != {0, 1} and labels != {0.0, 1.0}:
        raise SplitError(f"Target {target_column!r} is not binary 0/1: {sorted(labels)}.")
    has_exit = raw["exit_date"].notna() if "exit_date" in raw.columns else pd.Series(False, index=raw.index)
    is_left = pd.to_numeric(raw[target_column], errors="coerce") == 1
    info = {
        "label_values": [0, 1],
        "leavers": int(is_left.sum()),
        "stayers": int((~is_left).sum()),
        "exit_rows": int(has_exit.sum()),
    }
    if info["leavers"] != info["exit_rows"] or int((has_exit & ~is_left).sum()) != 0:
        raise SplitError(
            "Target encoding unverified: exit-information rows do not match label==1 exactly."
        )
    return info
