"""Threshold tables, selection, PR data and metric formulas (aggregates only)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(numerator: float, denominator: float) -> float:
    """Division with NaN (not fabricated zeros) on zero denominators."""
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def threshold_metrics(
    scores: np.ndarray, is_injected: np.ndarray, threshold: float
) -> dict[str, float]:
    """Catch/false-alarm/precision/F1 at one threshold with correct denominators."""
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(is_injected, dtype=bool)
    flagged = scores >= threshold
    n_injected = int(truth.sum())
    n_normal = int((~truth).sum())
    caught = int((flagged & truth).sum())
    false_alarms = int((flagged & ~truth).sum())
    n_flagged = int(flagged.sum())
    catch = safe_divide(caught, n_injected)
    false_alarm = safe_divide(false_alarms, n_normal)
    precision = safe_divide(caught, n_flagged)
    f1 = safe_divide(2 * precision * catch, (precision + catch)) if (
        not np.isnan(precision) and not np.isnan(catch) and (precision + catch) > 0
    ) else float("nan")
    return {
        "threshold": float(threshold),
        "n_records": len(scores),
        "n_flagged": n_flagged,
        "flagged_pct": safe_divide(100.0 * n_flagged, len(scores)),
        "n_injected": n_injected,
        "n_caught": caught,
        "n_missed": n_injected - caught,
        "catch_rate": catch,
        "n_normal": n_normal,
        "n_false_alarms": false_alarms,
        "false_alarm_rate": false_alarm,
        "precision": precision,
        "f1": f1,
    }


def threshold_table(scores: np.ndarray, is_injected: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    """One metrics row per candidate threshold, in the given order."""
    return pd.DataFrame([threshold_metrics(scores, is_injected, t) for t in thresholds])


def thresholds_from_flag_rates(scores: np.ndarray, flag_rates: list[float]) -> list[float]:
    """Score cutoffs flagging approximately the requested fractions (higher = worse)."""
    scores = np.asarray(scores, dtype=float)
    return [float(np.quantile(scores, 1.0 - r)) for r in flag_rates]


def select_threshold(
    table: pd.DataFrame, *, max_false_alarm: float, max_flagged_fraction: float
) -> dict[str, object]:
    """Highest catch rate within the false-alarm constraint (precision tie-break).

    Fallback, recorded explicitly: if nothing satisfies the constraint, take the
    highest catch rate among thresholds flagging at most max_flagged_fraction
    of records; if still nothing, take the lowest flag-rate threshold.
    """
    work = table.copy()
    feasible = work[work["false_alarm_rate"] <= max_false_alarm]
    if not feasible.empty:
        pick = feasible.sort_values(["catch_rate", "precision", "threshold"],
                                    ascending=[False, False, True]).iloc[0]
        return {"threshold": float(pick["threshold"]), "rule": (
            f"max catch rate with false-alarm rate <= {max_false_alarm:.0%} "
            f"(precision tie-break); constraint satisfied"),
            "fallback": False}
    limited = work[work["flagged_pct"] / 100.0 <= max_flagged_fraction]
    if not limited.empty:
        pick = limited.sort_values(["catch_rate", "precision", "threshold"],
                                   ascending=[False, False, True]).iloc[0]
        return {"threshold": float(pick["threshold"]), "rule": (
            f"FALLBACK: no threshold met false-alarm <= {max_false_alarm:.0%}; "
            f"chose max catch rate within flagged fraction <= {max_flagged_fraction:.0%}"),
            "fallback": True}
    pick = work.sort_values("flagged_pct", ascending=True).iloc[0]
    return {"threshold": float(pick["threshold"]), "rule": (
        "FALLBACK: no threshold met either constraint; chose the lowest flag-rate threshold"),
        "fallback": True}


def pr_points(table: pd.DataFrame) -> pd.DataFrame:
    """Precision-recall points across thresholds for plotting."""
    return table[["threshold", "catch_rate", "precision", "flagged_pct"]].copy()
