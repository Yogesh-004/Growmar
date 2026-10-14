# Task 2 Feature Dictionary — `features-1.0.0`

All six requested features are implemented from audited `attrition_dataset` fields.
Formulas, units and assumptions are also stored in `artifacts/task2/features/features_metadata.json`.

| # | Feature | Source column(s) | Formula / rule | Unit | Assumption / status |
|---|---|---|---|---|---|
| A | `attendance_rate` | `attendance_rate_3m` | Validated as-is; `attendance_invalid` = 1 when outside [0, 1] | proportion | Existing audited 3-month field used (allowed alternative); >1 invalid until definition confirmed (D09) |
| B | `late_count` | `late_arrivals_3m` | Used as-is (late arrivals in 3 months) | count | Grain already employee-level; no re-aggregation, no double counting |
| C | `avg_overtime` | `avg_overtime_hours_per_month_3m` | Used as-is (mean overtime/month over 3 months) | hours/month | Genuine overtime field; total hours worked never treated as overtime |
| D | `tenure_months` | `tenure_months` | Used as-is; `tenure_invalid` = 1 when negative | months | Negatives quarantined via flag, never silently corrected (D09) |
| E | `salary_vs_department_median` | `annual_fixed_pay`, `department_id` | `pay / department_median(pay)`; medians fitted on training data only, unseen departments use the global median, missing/non-positive values yield NaN for the imputer | ratio | Reference statistics frozen in the artifact; no validation/test participation |
| F | `months_since_revision` | `months_since_last_raise` (+ `is_never_raised` = 1 when `last_raise_pct` is missing) | Used as-is | months | A pay rise is the revision event; `months_since_last_promotion` never substituted. Values present when `raises_received` == 0 need HR confirmation |

## Supporting preprocessing (D01–D04, D09)

- Structural missingness: `is_never_promoted` / `is_never_raised` flags + 9999.0 sentinel fill (D01).
- `engagement_missing` flag + median/mode imputation fitted on training data only (D02).
- Numeric: median imputation + RobustScaler (robust to skew, D04). Categorical: most-frequent + one-hot with unknown categories ignored.
- Location mapping (7 observed variants → canonical names) is **provisional pending HR approval** (D09).
- Excluded from inputs (D06/D07 + batch flag): `row_id`, `employee_id`, both targets, all five exit/settlement fields, `source`.

## Monthly employee-month features (component `monthly-1.0.0`, separate path)

| Feature | Source column(s) | Formula / rule | Unit |
|---|---|---|---|
| `attendance_rate` | monthly `attendance_rate` | Validated as-is + `attendance_invalid` flag | proportion |
| `late_arrivals`, `late_arrivals_avg_3m` | monthly `late_arrivals` | Monthly count; trailing mean of current + 2 prior months | count/month |
| `overtime_hours`, `overtime_hours_avg_3m` | monthly `overtime_hours` | Monthly value; trailing mean (total hours never substituted) | hours/month |
| `attendance_rate_avg_3m` | monthly `attendance_rate` | Trailing mean, strictly as-of | proportion |
| Monthly pay fields | `monthly_fixed_gross`, `payable_days`, `lop_days`, `overtime_hours_paid`, `overtime_pay`, `variable_pay`, `gross_pay`, deductions, `net_pay` | Monthly values as-is | as labelled |
| `salary_vs_dept_median_monthly` | `monthly_fixed_gross` / `department_id` | Ratio vs dept median fitted on training months only | ratio |
| `job_level_mismatch` | `job_level_att` vs `job_level_pay` | 1 when both present and unequal; disputed levels excluded from inputs | flag |

Months-since-revision is **not** derived here: payroll history has no verified revision-date
column, so the attrition pipeline's `months_since_last_raise` definition stands with its
documented caveat. Full per-feature records: `artifacts/task2/features/monthly_features_metadata.json`.
