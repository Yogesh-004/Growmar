# Growmar Aura Workforce — Phase 1

HR attrition and workforce analytics. Phase 1 currently contains three organized tasks sharing
one dataset, one environment, and one Docker setup.

## Task index

- **Task 1: Data Audit & Exploration** (`task1_data_audit/`) — independent audit of
  `attrition_dataset`: profiling, missingness, outliers, imbalance, leakage, duplicates,
  decisions D01–D10, plain-language report, and a separate planted-issue validation appendix.
- **Task 2: Reusable Feature Pipeline** (`task2_feature_pipeline/`) — versioned sklearn
  preprocessing (`features-1.0.0` + `monthly-1.0.0`): an attrition pipeline and a monthly
  attendance/payroll pipeline. Monthly history is never merged into older attrition rows.
- **Task 3: Attrition Baselines** (`task3_attrition_baselines/`) — dummy, logistic
  regression and shallow-tree baselines with 5-fold CV on the locked split; winner
  **logreg_balanced** by CV leaver recall. Report: `task3_attrition_baselines/reports/BASELINE_REPORT.md`.
- **Task 4: Attendance Anomaly Detection** (`task4_attendance_anomaly/`) — Isolation Forest
  over employee-month attendance with seeded injections; frozen threshold catches 58% at
  7.0% false alarms. Report: `task4_attendance_anomaly/reports/ATTENDANCE_ANOMALY_REPORT.md`.

## How the tasks depend on each other

Task 1 discovers and documents data-quality rules (D01–D10); Task 2 implements the
applicable rules as reusable preprocessing code (exclusions, missingness flags, train-only
fitting, location mapping, dedup). Task 2 reads Task 1's decisions, never the reverse.
Task 3 evaluates baselines on Task 2's locked split and frozen pipeline discipline.
Task 4 detects attendance anomalies with a calibrated Isolation Forest.
No Phase 2 code exists yet.

## Layout

```text
data/raw/                        shared workbook (never committed, never modified)
data/processed/task2_features/   feature matrices + labels (local, git-ignored)
task1_data_audit/                audit code, configs, notebook, reports, tests
task2_feature_pipeline/          feature package, configs, docs, tests
task3_attrition_baselines/       baseline models, configs, reports, tests
task4_attendance_anomaly/       anomaly detection code, configs, reports, tests
artifacts/task2/features/        saved pipelines, metadata, manifests (local, git-ignored)
artifacts/task3/                 fitted baselines + selection record (local, git-ignored)
artifacts/task4/                 fitted detector + threshold metadata (local, git-ignored)
tests/shared/                    shared-test-helper convention (none needed yet)
requirements.txt / pyproject.toml / Dockerfile / docker-compose.yml   shared env
```

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Docker and Jupyter

```powershell
docker build -t aura-task1-audit .
docker run --rm -p 127.0.0.1:8888:8888 -v "${PWD}:/workspace" aura-task1-audit
docker compose build
docker compose up          # Jupyter Lab on 127.0.0.1:8888
```

## Run the tasks

```powershell
python task1_data_audit/scripts/run_audit.py
python task1_data_audit/scripts/validate_planted_issues.py
python scripts/run_audit.py              # compatibility wrapper
python -m features.cli fit-transform --workbook data/raw/CortexHR_AI_ML_Dataset.xlsx
python -m features.cli monthly-fit-transform --workbook data/raw/CortexHR_AI_ML_Dataset.xlsx
python task3_attrition_baselines/scripts/run_baselines.py
python task4_attendance_anomaly/scripts/run_anomaly_detection.py
```

## Tests

```powershell
pytest -q                                # full suite (tasks 1–4)
pytest -q task1_data_audit/tests
pytest -q task2_feature_pipeline/tests
pytest -q task3_attrition_baselines/tests
pytest -q task4_attendance_anomaly/tests
```

## Data protection (local-only outputs)

The workbook (`data/raw/`), processed matrices (`data/processed/`), saved artifacts
(`artifacts/`), and row-alignment files stay out of version control (see `.gitignore`).
Record-level detail files (`outlier_details.csv`, `data_quality_issue_details.csv`,
`scored_final.csv`, injection manifests) are generated locally and likewise excluded;
regenerate them with the documented commands. Reports and logs never contain employee IDs.
Do not commit, push, or deploy without review.
