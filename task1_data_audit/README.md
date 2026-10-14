# Task 1: Data Audit & Exploration

Independent audit of `attrition_dataset` (2,389 rows × 42 columns): profiling, missingness,
IQR/z-score outlier comparison, attrition prevalence, full-column leakage register, duplicate
and invalid-value review, decisions D01–D10, and a separate post-audit validation appendix.

## Layout

- `src/audit/` — shared audit helpers (imported as `audit.*`).
- `scripts/` — `run_audit.py`, `validate_planted_issues.py` (canonical; root `scripts/` holds thin wrappers).
- `notebooks/01_data_audit.ipynb` — review workspace using the same helpers.
- `configs/task1_config.json` — workbook path, sheet, target/outlier/leakage/ID candidates.
- `reports/TASK1_AUDIT_REPORT.md` — plain-language report (charts: `reports/generated/figures/`).
- `reports/generated/` — CSV evidence + `planted_issue_validation.csv` + figures.
- `tests/` — `test_audit_utils.py`, `test_planted_validation.py` (synthetic fixtures).

## Commands (from the project root)

```powershell
python task1_data_audit/scripts/run_audit.py
python task1_data_audit/scripts/validate_planted_issues.py
python scripts/run_audit.py              # compatibility wrapper, same behavior
pytest -q task1_data_audit/tests
jupyter lab                              # then open task1_data_audit/notebooks/01_data_audit.ipynb
```

The audit never reads the answer-key sheet; only the validation script does, after the fact.
