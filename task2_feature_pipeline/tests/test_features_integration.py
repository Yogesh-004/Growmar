"""Integration checks against the real workbook (three authorized sheets only).

These tests read attrition_dataset, attendance_monthly_features and
payroll_monthly_features. They never touch the answer-key sheet.
"""
from __future__ import annotations

import pandas as pd

from features import schema

WORKBOOK = "data/raw/CortexHR_AI_ML_Dataset.xlsx"


def _load(sheet: str) -> pd.DataFrame:
    return schema.load_attrition_sheet(WORKBOOK, sheet)


def test_authorized_sheet_schemas():
    attr = _load("attrition_dataset")
    att = _load("attendance_monthly_features")
    pay = _load("payroll_monthly_features")
    assert attr.shape == (2389, 42)
    assert att.shape[1] == 16 and pay.shape[1] == 23
    assert "month" in att.columns and "month" in pay.columns
    assert int(att.duplicated(subset=["employee_id", "month"]).sum()) == 0
    overlap = set(attr["employee_id"].astype(str)) & set(att["employee_id"].astype(str))
    assert len(overlap) == 383  # documents the limited join coverage
    assert (attr["observation_date"] >= pd.Timestamp("2025-04-01")).sum() == 389


def test_supervised_frame_from_real_workbook():
    raw = _load("attrition_dataset")
    X, y, keys = schema.build_supervised_frame(raw, "left_within_12m")
    assert len(X) == len(raw)  # no missing labels in the supplied data
    assert abs(float(y.mean()) - 0.1461) < 0.001
    assert "row_id" in keys.columns and "employee_id" not in keys.columns
