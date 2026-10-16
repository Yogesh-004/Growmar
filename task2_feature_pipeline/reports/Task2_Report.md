# Aura Workforce: Task 2 Feature Pipeline Report

**Task:** Reusable Feature Pipeline  
**Purpose:** Turn approved source data into consistent model-ready features for later tasks.

## What we built

Task 2 creates a reusable preprocessing package rather than making each model clean data in its own way. The package is versioned as `features-1.0.0`. It can fit a pipeline, save it, reload it, and transform new compatible records using the same rules.

The package has two separate processing paths: one for employee attrition observations and one for monthly attendance/payroll records. Keeping them separate avoids mixing records with different dates and coverage.

## Why two paths are needed

The attrition worksheet contains 2,389 employee observations covering January 2021 to September 2025. The monthly attendance and payroll worksheets cover April 2025 to September 2026 and contain records by employee and month.

Only 383 employees overlap between the attrition dataset and the monthly data. Most older attrition observations have no matching monthly history, and some monthly records are dated after an attrition observation. Joining all of them would create many empty values and could accidentally let future information influence an earlier prediction.

For that reason, the attrition pipeline uses the audited features already available in `attrition_dataset`, while the monthly pipeline processes attendance and payroll at employee-month level. This is an intentional data-protection and timing decision, not an incomplete join.

## What the pipeline prepares

The requested feature definitions are handled as follows:

| Feature | What the pipeline uses |
|---|---|
| Attendance rate | Uses the audited `attendance_rate_3m` value and flags invalid values instead of recomputing the rate with an unverified formula. |
| Late-arrival count | Uses `late_arrivals_3m`, which already represents the relevant period. |
| Average overtime | Uses the actual `avg_overtime_hours_per_month_3m` overtime field, not total working hours. |
| Tenure | Uses `tenure_months` and flags invalid values. |
| Salary compared with department | Calculates salary relative to a department median learned from training data. |
| Time since last revision or raise | Uses `months_since_last_raise` and a flag for no raise. A separate salary-revision date is not available, so the distinction between a raise and a revision still needs confirmation. |

The monthly path also validates its schemas, removes confirmed duplicate payroll lines, combines matching employee-month records, creates trailing three-month summaries and tracks mismatched job levels.

## How missing data and unusual values are handled

The pipeline follows the Task 1 decisions. It preserves meaningful information when a blank means an event has not happened, such as no promotion or raise. Where ordinary missing values need filling, the imputation values are learned from training data only.

The pipeline does not silently correct unusual salary, tenure, attendance or work-from-home values. It preserves the documented flags and applies the rules recorded in Task 1. The salary-to-`comp_ratio` interpretation and some HR field definitions remain provisional.

## How we prevent information leakage

Before fitting, the attrition data is split into training and test sets using the existing stratified split and seed 42. The reported split is 1,911 training rows and 478 test rows.

The pipeline learns its filling values, scaling values, category encodings and department salary references from the training portion only. It then applies those saved settings to test data or future compatible records without fitting again.

The monthly path uses time order: April 2025 to March 2026 for training, and April to September 2026 for holdout. Its trailing three-month features use the current month and earlier months only. This prevents a later month from changing features for an earlier month.

## Duplicate payroll lines and source differences

Fifteen payroll lines were duplicates apart from their technical `payroll_item_id`. The pipeline removes a duplicate only when all the other relevant fields match. If records disagree on those fields, it raises an error for review rather than silently choosing one.

After deduplication, all 7,339 attendance employee-month keys match payroll keys, and one payroll-only key remains outside the feature output. The department agrees on all matched rows. The job level differs on 168 matched rows, so both source values are retained in a local alignment file and a mismatch flag is added. Neither disputed job-level value is used as a feature until the source owner confirms which is correct.

## Saved outputs and how they are reused

The attrition pipeline is saved locally at `artifacts/task2/features/pipeline.joblib` with `features_metadata.json`. The monthly pipeline is saved at `artifacts/task2/features/monthly_pipeline.joblib` with `monthly_features_metadata.json`.

The metadata records the version, input schema, feature order, preprocessing settings and assumptions. The generated feature matrices keep labels and sensitive alignment information separate.

The command-line interface supports fitting and transforming data, including the separate monthly workflow. A saved pipeline can transform new records without being fitted again, and the output columns stay in the same order.

## Verification reported

The implementation report records 46 tests passing locally and in Docker at the time Task 2 was completed. It also records successful fitting, saving, loading, transforming and running the monthly workflow inside Docker. Later Phase 1 work expanded the full project test suite, so use the latest root-level test run for the current overall count.

## Questions still open

The following definitions need confirmation before the affected fields should be treated as authoritative:
- Which job-level source is correct when attendance and payroll disagree.
- The exact business meaning of `comp_ratio`.
- Whether `months_since_last_raise` is an acceptable proxy for time since salary revision.
- The official time window for the department prior attrition rate.
- Whether engagement-score missingness follows a systematic pattern.

These questions are documented rather than silently resolved.

## Handoff to later tasks

Task 3 uses the attrition feature pipeline for baseline model comparisons. Task 4 uses the separate monthly path for attendance anomaly detection. Both should reuse the saved preprocessing logic and its version metadata instead of independently preparing the same data again.
