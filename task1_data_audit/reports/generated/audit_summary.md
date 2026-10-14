# Task 1: Data Audit & Exploration (Initial Automated Profile)

## Scope

- Input workbook: `CortexHR_AI_ML_Dataset.xlsx`
- Worksheet read: `attrition_dataset` only
- Rows: **2,389**
- Columns: **42**
- Exact duplicate rows: **0**
- This report is a profiling draft, not the final human-reviewed audit.
- The source workbook was not modified. No rows or columns were dropped, filled, capped, or rewritten.
- Workbook links are not opened or followed by this script. The configured planted-answer-key sheet is not read.

## Missingness

- `resignation_date`: 2,068 missing (86.56%). Decision to be documented.
- `exit_date`: 2,040 missing (85.39%). Decision to be documented.
- `exit_interview_done`: 2,040 missing (85.39%). Decision to be documented.
- `exit_reason`: 2,040 missing (85.39%). Decision to be documented.
- `fnf_settlement_amount`: 2,040 missing (85.39%). Decision to be documented.
- `months_since_last_promotion`: 1,459 missing (61.07%). Decision to be documented.
- `last_engagement_score`: 528 missing (22.10%). Decision to be documented.
- `last_raise_pct`: 202 missing (8.46%). Decision to be documented.
- `last_performance_rating`: 182 missing (7.62%). Decision to be documented.
- `commute_distance_km`: 55 missing (2.30%). Decision to be documented.
- `highest_education`: 15 missing (0.63%). Decision to be documented.

Complete inventory: `data_dictionary.csv`; decision fields: `missingness.csv`.

## Outlier screening

The IQR method flags values outside Q1 - 1.5xIQR or Q3 + 1.5xIQR. The z-score screen flags absolute z-scores above 3.0. Both are screening methods and can flag legitimate extreme values.

- `annual_fixed_pay`: IQR flagged 94; |z-score| > 3 flagged 26. These are review flags, not confirmed errors.
- `avg_hours_worked_3m`: IQR flagged 48; |z-score| > 3 flagged 15. These are review flags, not confirmed errors.
- `tenure_months`: IQR flagged 74; |z-score| > 3 flagged 42. These are review flags, not confirmed errors.

## Target balance

- Target column detected: `left_within_12m`.
- Rate matching configured positive label values: 14.61%. Confirm the configured positive values match the actual label semantics.

The class counts and percentages are in `target_distribution.csv`.

## Potential leakage fields

These are candidates requiring manual review. Exclude a field from model inputs when it would not be known at the prediction time or directly/indirectly reveals the outcome.

- `exit_date`: verify whether available at the prediction time.
- `resignation_date`: verify whether available at the prediction time.
- `exit_reason`: verify whether available at the prediction time.
- `fnf_settlement_amount`: verify whether available at the prediction time.
- `exit_interview_done`: verify whether available at the prediction time.

Full per-column review: `leakage_register.csv`.

## Duplicate review

See `duplicate_review.csv`. Duplicate identifiers require investigation in context; they are not automatically deleted.

## Human decisions still required

1. Complete `definition_in_own_words` in `data_dictionary.csv`.
2. Document a choice and rationale for each missing column in `missingness.csv`.
3. Inspect each flagged outlier before deciding to retain, flag, correct, or cap it.
4. Confirm target column and positive label semantics from the source specification.
5. Finalize the leakage list based on when predictions are intended to be made.
6. Review suspicious ranges and identifier duplication, and add any findings to the final report.

## Plain-language conclusion (complete after review)

Write 1-2 paragraphs describing the most important data-quality risks, what can safely be used, what needs validation, and the decisions that will become rules in Task 2.
