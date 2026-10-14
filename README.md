# Growmar Aura Workforce — Phase 1

HR attrition and workforce analytics. Phase 1 currently contains two organized tasks sharing
one dataset, one environment, and one Docker setup.

## Task index

- **Task 1: Data Audit & Exploration** (`task1_data_audit/`) — independent audit of
  `attrition_dataset`: profiling, missingness, outliers, imbalance, leakage, duplicates,
  decisions D01–D10, plain-language report, and a separate planted-issue validation appendix.
- **Task 2: Reusable Feature Pipeline** (`task2_feature_pipeline/`) — versioned sklearn
  preprocessing (`features-1.0.0` + `monthly-1.0.0`): an attrition pipeline and a monthly
  attendance/payroll pipeline. Monthly history is never merged into older attrition rows.

## How the tasks depend on each other

Task 1 discovers and documents data-quality rules (D01–D10); Task 2 implements the
applicable rules as reusable preprocessing code (exclusions, missingness flags, train-only
fitting, location mapping, dedup). Task 2 reads Task 1's decisions, never the reverse.
No Task 3/4 code exists yet; Task 3 will reuse the split manifests and saved artifacts.

## Layout

```text
data/raw/                        shared workbook (never committed, never modified)
data/processed/task2_features/   feature matrices + labels (local, git-ignored)
task1_data_audit/                audit code, configs, notebook, reports, tests
task2_feature_pipeline/          feature package, configs, docs, tests
artifacts/task2/features/        saved pipelines, metadata, manifests (local, git-ignored)
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
```

## Tests

```powershell
pytest -q                                # full suite (task1 + task2)
pytest -q task1_data_audit/tests
pytest -q task2_feature_pipeline/tests
```

## Data protection (local-only outputs)

The workbook (`data/raw/`), processed matrices (`data/processed/`), saved artifacts
(`artifacts/`), and row-alignment files stay out of version control (see `.gitignore`).
Reports and logs never contain employee IDs. Do not commit, push, or deploy without review.
