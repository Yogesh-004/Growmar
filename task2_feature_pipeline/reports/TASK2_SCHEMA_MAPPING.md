# Task 2 Schema Mapping — Source Sheets and Join Decision

Workbook: `data/raw/CortexHR_AI_ML_Dataset.xlsx` (unmodified). Each sheet below was read with an explicit `sheet_name`; no hyperlinks were followed and no other sheets were inspected.

## 1. `attrition_dataset` — supervised frame (2,389 rows × 42 columns)

One row per employee observation: `employee_id` + `observation_date` (2021-01-01 → 2025-09-30).
Technical key `row_id` (unique); target `left_within_12m` (349 leavers, 14.61%);
alternate label `voluntary_left_within_12m`; five post-outcome fields (D06 exclusions);
`source` batch flag (`historical_extension` 2,000 / `master_employee_snapshot` 389).
All six engineered features come from this sheet (see `TASK2_FEATURE_DICTIONARY.md`).

## 2. `attendance_monthly_features` — monthly grain (7,339 rows × 16 columns)

One row per employee × month (`month` is month-start datetime, 2025-04-01 → 2026-09-01).
528 employees, ~13.9 rows each, no `(employee_id, month)` duplicates.
Key columns: `scheduled_days` (never 0), `present_days`, `late_arrivals`, `overtime_hours`,
`avg_hours_worked`, `attendance_rate` (all within [0, 1]; note: `present_days/scheduled_days`
does not always reproduce `attendance_rate`, so the rate must not be recomputed blindly).

## 3. `payroll_monthly_features` — monthly grain (7,355 rows × 23 columns)

One row per payroll line per employee × month (2025-04-01 → 2026-09-01), 528 employees,
18 monthly runs (`run_id` PR-2025-04 … PR-2026-09). 15 `(employee_id, month)` duplicates are
byte-identical re-processed lines under the same `run_id`. Genuine overtime fields exist
(`overtime_hours_paid`, `overtime_pay`; mostly zero). There is **no salary-revision date
column**, so months-since-revision cannot be derived from payroll history.

## 4. Relationships and join decision

- Monthly sheets cover 528 employees; only **383 overlap** the 2,383 attrition employees.
- Monthly history starts 2025-04, but only **389/2,389** attrition rows are observed on/after
  2025-04-01 — an as-of join (`month <= observation_date`) would be empty for ~84% of rows.
- Monthly data runs to 2026-09, past the latest observation (2025-09-30): merging without an
  as-of guard would leak future information (D10).
- **Decision (documented, not silent): the supervised training matrix is built from
  `attrition_dataset` alone**, whose audited 3-month aggregates already provide attendance,
  lateness and overtime. Monthly tables are inspected evidence for this decision, not merged
  inputs. If Task 4 needs monthly grain later, the same as-of rule (`month <= observation
  date`, deduped `(employee_id, month)`) must apply.

## 5. Monthly attendance/payroll processing (separate path, `monthly-1.0.0`)

Implemented in `src/features/monthly.py` (CLI `monthly-fit-transform` / `monthly-transform`).

- **Dedup:** 15 duplicate `(employee_id, month)` payroll groups are byte-identical except
  `payroll_item_id` → dropped (count logged); conflicting groups raise an error.
- **Join:** inner 1:1 on `(employee_id, month)` — 7,339 matched, 0 attendance-only, 1
  payroll-only key (excluded from features, counted in metadata).
- **Field agreement:** `department_id` agrees 7,339/7,339; `job_level` differs on 168 rows →
  `job_level_mismatch` flag, both values kept in the local alignment file, neither used as
  an input until source-owner confirmation.
- **Split:** chronological holdout at 2026-04 (train 2025-04…2026-03, 4,792 rows; holdout
  2026-04…2026-09, 2,547 rows); trailing 3-month means use current + prior months only.
- **Outputs:** `artifacts/task2/features/monthly_pipeline.joblib` + `monthly_features_metadata.json`;
  `data/processed/monthly_{train,test}_features.csv` and git-ignored `monthly_alignment.csv`.
