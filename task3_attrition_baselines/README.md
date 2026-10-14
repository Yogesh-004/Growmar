# Task 3: Attrition Baseline Models & Evaluation (`baselines-1.0.0`)

Honest, reproducible baselines for `left_within_12m` (confirmed by the project owner as the Task 3 target):
a stays-always dummy, logistic regression (plain + balanced), and a deliberately shallow
depth-4 decision tree (plain + balanced).

## Layout

- `src/baselines/` — `config`, `data` (locked-split reconstruction), `models`
  (prespecified candidates), `evaluation` (fold-safe CV, selection, holdout).
- `configs/task3_config.json` — target, seeds, folds, model settings, paths.
- `scripts/run_baselines.py` — full workflow: CV on train → select → fit → locked test → report.
- `reports/BASELINE_REPORT.md` + `reports/generated/` — metrics CSVs, confusion matrix, ROC curve.
- `tests/test_baselines.py` — synthetic fixtures only.

## Method guarantees

- Task 2's locked stratified split (1911/478, seed 42) is reconstructed from the manifest
  and cross-checked against Task 2 label files — never regenerated.
- Task 2's **unfitted** `build_pipeline()` is cloned inside every CV fold, so imputers,
  scalers, encoders and department medians learn from fold-train rows only.
- Selection rule (recall first, F1 tie-break) runs on CV results; the test set is
  evaluated once, afterwards, at the fixed 0.5 threshold.

## Commands (from the project root)

```powershell
python task3_attrition_baselines/scripts/run_baselines.py
pytest -q task3_attrition_baselines/tests
```

Selected winner: **logreg_balanced** (CV recall 0.613, test recall 0.60 at precision 0.25).
Finalists saved under `artifacts/task3/` with `selected_config.json` (local, git-ignored).
