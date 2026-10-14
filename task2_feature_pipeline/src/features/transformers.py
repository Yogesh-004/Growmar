"""Fitted transformers: department salary reference (training data only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class DeptMedianRatio(BaseEstimator, TransformerMixin):
    """Append salary_vs_department_median = pay / department median pay.

    Department medians (plus a global fallback median) are learned from the
    training frame at fit time and frozen afterwards, so validation, test and
    future rows always use the same references. Departments unseen during fit
    fall back to the global median; missing or non-positive pay/reference
    values yield NaN for the downstream imputer. Never uses the target.
    """

    def __init__(
        self,
        pay_column: str = "annual_fixed_pay",
        dept_column: str = "department_id",
        output_column: str = "salary_vs_department_median",
    ) -> None:
        self.pay_column = pay_column
        self.dept_column = dept_column
        self.output_column = output_column

    def fit(self, X: pd.DataFrame, y=None) -> "DeptMedianRatio":
        if self.pay_column not in X.columns or self.dept_column not in X.columns:
            raise ValueError(
                f"DeptMedianRatio needs columns {self.pay_column!r} and {self.dept_column!r}."
            )
        pay = pd.to_numeric(X[self.pay_column], errors="coerce")
        positive = pay.where(pay > 0)
        medians = positive.groupby(X[self.dept_column]).median()
        self.dept_medians_ = medians[medians.notna() & (medians > 0)].to_dict()
        global_med = float(positive.median()) if positive.notna().any() else float("nan")
        self.global_median_ = global_med if global_med > 0 else float("nan")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["dept_medians_", "global_median_"])
        if self.pay_column not in X.columns or self.dept_column not in X.columns:
            raise ValueError(
                f"DeptMedianRatio needs columns {self.pay_column!r} and {self.dept_column!r}."
            )
        out = X.copy()
        pay = pd.to_numeric(out[self.pay_column], errors="coerce")
        ref = out[self.dept_column].map(self.dept_medians_).astype(float)
        ref = ref.where(ref.notna() & (ref > 0), self.global_median_)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = pay / ref
        ratio = ratio.where(pay.notna() & (pay > 0) & ref.notna() & (ref > 0), np.nan)
        out[self.output_column] = ratio.astype(float)
        return out

    def references(self) -> dict[str, float]:
        """Fitted per-department medians (for metadata persistence)."""
        check_is_fitted(self, ["dept_medians_"])
        return {str(k): float(v) for k, v in self.dept_medians_.items()}
