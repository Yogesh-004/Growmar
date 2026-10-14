"""Deterministic anomaly injection on copies (originals never modified)."""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ("impossible_hours", "long_absence", "sudden_change")

# Experimental working-hours range for monthly averages. NOT an official HR
# rule: daily evidence shows anomalies at 17+ hours while ordinary monthly
# averages cluster near 8.6 (max naturally observed monthly value is 18.94).
HOURS_INJECT_RANGE = (19.0, 22.0)
# Long-absence streaks mirror the 8-20 consecutive-day notes in the reference labels.
STREAK_RANGE = (10, 20)
# Sudden-change magnitude applied to the employee's own trailing mean.
DROP_RANGE = (0.40, 0.60)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def eligible_pool(pool: pd.DataFrame, reference_ids: set[str]) -> pd.DataFrame:
    """Rows eligible for injection: not reference-labelled, sorted for determinism."""
    mask = ~pool["attendance_feature_id"].astype(str).isin(reference_ids)
    return pool.loc[mask].sort_values(["employee_id", "month"]).reset_index(drop=True)


def inject_impossible_hours(frame: pd.DataFrame, n: int, seed: int, skip_ids: frozenset = frozenset()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Set avg_hours_worked into the experimental impossible range."""
    rng = _rng(seed)
    candidates = frame.index[~frame["attendance_feature_id"].astype(str).isin(skip_ids)].to_numpy()
    idx = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
    out = frame.copy()
    values = np.round(rng.uniform(*HOURS_INJECT_RANGE, size=len(idx)), 2)
    manifest = []
    for pos, new in zip(idx, values):
        old = float(frame.loc[pos, "avg_hours_worked"])
        out.loc[pos, "avg_hours_worked"] = new
        manifest.append(_record(frame, pos, "impossible_hours", "avg_hours_worked", old, float(new),
                                f"set into experimental range {HOURS_INJECT_RANGE} (daily evidence: anomalies at 17+ h)", seed))
    return out, pd.DataFrame(manifest)


def inject_long_absence(frame: pd.DataFrame, n: int, seed: int, skip_ids: frozenset = frozenset()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impose a long absence pattern coherent across absence fields.

    The streak value fills longest_unpaid_streak_days and unpaid_absence_days;
    present_days and attendance_rate follow arithmetically. No cause is
    recorded: the pattern is a long absence, not a claimed reason.
    """
    rng = _rng(seed)
    eligible = frame[
        (pd.to_numeric(frame["scheduled_days"], errors="coerce") >= 10)
        & (~frame["attendance_feature_id"].astype(str).isin(skip_ids))
    ].index.to_numpy()
    idx = rng.choice(eligible, size=min(n, len(eligible)), replace=False)
    out = frame.copy()
    manifest = []
    for pos in idx:
        streak = int(rng.integers(STREAK_RANGE[0], STREAK_RANGE[1] + 1))
        sched = int(frame.loc[pos, "scheduled_days"])
        paid = int(frame.loc[pos, "paid_leave_days"])
        present = int(max(0, min(sched, sched - paid - streak)))
        rate = round(present / sched, 4) if sched else 0.0
        changes = {
            "longest_unpaid_streak_days": (int(frame.loc[pos, "longest_unpaid_streak_days"]), streak),
            "unpaid_absence_days": (int(frame.loc[pos, "unpaid_absence_days"]), streak),
            "present_days": (int(frame.loc[pos, "present_days"]), present),
            "attendance_rate": (float(frame.loc[pos, "attendance_rate"]), rate),
        }
        for field, (old, new) in changes.items():
            out.loc[pos, field] = new
            manifest.append(_record(frame, pos, "long_absence", field, old, new,
                                    f"streak {streak} days maps daily-streak evidence to monthly fields", seed))
    return out, pd.DataFrame(manifest)


def inject_sudden_change(frame: pd.DataFrame, n: int, seed: int, skip_ids: frozenset = frozenset()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop attendance far below the employee's own trailing mean.

    Eligibility needs at least three prior months in the same pool and a
    trailing mean of 0.5 or more, so the change is large and meaningful.
    """
    rng = _rng(seed)
    ordered = frame.sort_values(["employee_id", "month"])
    trail = (
        pd.to_numeric(ordered["attendance_rate"], errors="coerce")
        .groupby(ordered["employee_id"].to_numpy(), sort=False)
        .transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
    )
    eligible = ordered.index[
        (trail >= 0.5) & trail.notna()
        & (~ordered["attendance_feature_id"].astype(str).isin(skip_ids))
    ].to_numpy()
    idx = rng.choice(eligible, size=min(n, len(eligible)), replace=False)
    out = frame.copy()
    manifest = []
    for pos in idx:
        base = float(trail.loc[pos])
        drop = float(rng.uniform(*DROP_RANGE))
        new_rate = round(float(np.clip(base - drop, 0.0, 1.0)), 4)
        sched = int(frame.loc[pos, "scheduled_days"])
        new_present = int(round(new_rate * sched)) if sched else 0
        for field, old, new in (
            ("attendance_rate", float(frame.loc[pos, "attendance_rate"]), new_rate),
            ("present_days", int(frame.loc[pos, "present_days"]), new_present),
        ):
            out.loc[pos, field] = new
            manifest.append(_record(frame, pos, "sudden_change", field, old, new,
                                    f"drop {drop:.2f} from own trailing mean {base:.2f}", seed))
    return out, pd.DataFrame(manifest)


def _record(frame: pd.DataFrame, pos, category: str, field: str, old, new, rule: str, seed: int) -> dict:
    row = frame.loc[pos]
    return {
        "attendance_feature_id": str(row["attendance_feature_id"]),
        "employee_id": str(row["employee_id"]),
        "month": str(row["month"]),
        "category": category,
        "field": field,
        "original_value": old,
        "injected_value": new,
        "rule": rule,
        "seed": seed,
        "is_injected": True,
    }


def inject_pool(
    pool: pd.DataFrame, reference_ids: set[str], per_category: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply all three categories in fixed order with disjoint row sets.

    Categories run in CATEGORIES order; rows already injected stay eligible for
    no later category. Each category uses its own seeded stream (seed, seed+1,
    seed+2), so reruns with the same inputs are identical. Returns the injected
    frame plus a manifest with one row per modified field.
    """
    eligible = eligible_pool(pool, reference_ids)
    out, manifests, used = eligible.copy(), [], set()
    for offset, category in enumerate(CATEGORIES):
        func = {"impossible_hours": inject_impossible_hours,
                "long_absence": inject_long_absence,
                "sudden_change": inject_sudden_change}[category]
        out, part = func(out, per_category, seed + offset, frozenset(used))
        used |= set(part["attendance_feature_id"].astype(str).tolist())
        manifests.append(part)
    manifest = pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame()
    return out.reset_index(drop=True), manifest


def injection_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Aggregate counts per category (no identifiers)."""
    if manifest.empty:
        return pd.DataFrame(columns=["category", "fields_modified", "rows_affected"])
    return pd.DataFrame({
        "category": sorted(manifest["category"].unique()),
        "fields_modified": [int((manifest["category"] == c).sum()) for c in sorted(manifest["category"].unique())],
        "rows_affected": [int(manifest.loc[manifest["category"] == c, "attendance_feature_id"].nunique())
                          for c in sorted(manifest["category"].unique())],
    })
