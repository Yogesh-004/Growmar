# Task 2: Reusable Feature Pipeline (`features-1.0.0` + `monthly-1.0.0`)

Versioned scikit-learn preprocessing implementing Task 1 decisions D01–D10. Two paths,
one task — monthly history is never merged into older attrition observations.

## Layout

- `src/features/` — `rules`, `schema`, `engineering`, `transformers`, `pipeline`,
  `persistence`, `split`, `monthly`, `cli`, `config` (imported as `features.*`).
- `configs/task2_config.json` — sheets, split seed, artifact/output dirs, exclusions.
- `reports/` — `TASK2_SCHEMA_MAPPING.md`, `TASK2_FEATURE_DICTIONARY.md`.
- `tests/` — `test_features.py`, `test_monthly.py`, `test_features_integration.py`.

## Pipelines

- **Attrition** (131 features): structural-missingness flags, train-only median/mode
  imputation + RobustScaler/one-hot, provisional location map, fitted department salary
  ratio; stratified split (seed 42) with `row_id` manifest, no employee IDs.
- **Monthly** (107 features): payroll dedup ignoring `payroll_item_id`, 1:1 join on
  `(employee_id, month)`, `job_level_mismatch` flag, as-of trailing windows,
  chronological holdout (train < 2026-04).

## Commands (from the project root)

```powershell
pip install -e .
python -m features.cli fit-transform --workbook data/raw/CortexHR_AI_ML_Dataset.xlsx
python -m features.cli transform --input new_rows.csv --output feats.csv
python -m features.cli monthly-fit-transform --workbook data/raw/CortexHR_AI_ML_Dataset.xlsx
python -m features.cli monthly-transform --input data/raw/CortexHR_AI_ML_Dataset.xlsx --output data/processed/monthly_features.csv
pytest -q task2_feature_pipeline/tests
```

Artifacts: `artifacts/task2/features/` (`pipeline.joblib`, `monthly_pipeline.joblib`, metadata,
manifests). Processed matrices: `data/processed/task2_features/` (features + separate labels;
local `monthly_alignment.csv` stays out of Git).
