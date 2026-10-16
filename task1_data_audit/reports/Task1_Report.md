# Aura Workforce: Task 1 Audit Report

**Task:** Data Audit and Exploration  
**Source:** `CortexHR_AI_ML_Dataset.xlsx` · **Worksheet:** `attrition_dataset`  
**Size:** 2,389 rows and 42 columns

## What we checked

Reviewed the employee information to find missing values, unusual measurements, repeated records, and fields that could give a prediction model information about the answer in advance. The original workbook was not changed. The detailed column list and supporting calculations are in the accompanying CSV files.

## Main findings

The data can move to Task 2 if the decisions below are followed. Some blanks have a meaningful explanation, others need careful filling, and some values need to be checked with the source team. Information recorded after an employee leaves must not be used to predict whether they will leave.

![Missing values by field](generated/figures/missingness_top20.png)

## Missing information and decisions

| Field or group | What we found | Decision and reason |
|---|---|---|
| Exit and settlement fields | About 85%–87% blank; generally recorded only when someone leaves. | Exclude from prediction inputs. These fields can reveal the answer. |
| `months_since_last_promotion` | 1,459 blank (61.07%); blanks line up with no promotion. | Preserve the meaning and add a flag indicating that no promotion is recorded. |
| `last_raise_pct` | 202 blank (8.46%); blanks may mean no raise occurred. | Preserve the meaning and add a flag, subject to confirming the definition. |
| `last_engagement_score` | 528 blank (22.10%); reason not fully known. | Investigate. If used, fill using training data and add a “was missing” flag. |
| `last_performance_rating` | 182 blank (7.62%). | Fill during Task 2 using a method learned from training data only. |
| `commute_distance_km` | 55 blank (2.30%). | Fill during Task 2 using training data only. |
| `highest_education` | 15 blank (0.63%). | Fill during Task 2 using an appropriate training-only method. |

We will not drop a row just because an input field is blank. We will not fill exit-related fields because they must not be model inputs.

### Think for yourself: what if a column has 40% missing values?

We would **not drop a column just because 40% is missing**. First, we would check what it means, why values are missing, whether it would be available when a real prediction is made, and whether it is useful.

If it is a valid, useful input, our starting choice is to keep it, add a `was_missing` flag when that provides useful information, and fill blanks using a method fitted on training data only. If a blank has a real meaning, such as “no promotion yet,” we preserve that meaning instead of blindly replacing it. We drop a column only for a documented reason, such as revealing the future outcome, not being available at prediction time, or providing too little reliable information.

This is a hypothetical example. No current column has exactly 40% missingness; the reported rates include 22.10% and 61.07%.

## Unusual values: two checks compared

An **outlier** is a value that looks far from many others. It deserves a closer look, but is not automatically an error.

We used two checks. The **IQR method** looks for values far outside the middle half of the data. The **z-score method** looks for values far from the average compared with the overall spread.

| Field | IQR flags | Z-score flags | Decision |
|---|---:|---:|---|
| Annual fixed pay | 94 (3.93%) | 26 (1.09%) | Keep and investigate extremes against the source; pay values are unevenly distributed. |
| Average hours worked | 48 (2.01%) | 15 (0.63%) | Keep and investigate extremes; one value around 1.8 hours looks implausible under the current definition. |
| Tenure in months | 74 (3.10%) | 42 (1.76%) | Keep and investigate; four negative values cannot represent ordinary tenure. |

![Outlier method comparison](generated/figures/outlier_method_comparison.png)

The two methods flag different numbers partly because salary has a long high-value tail. The audit does not automatically delete or change flagged records. Exact flagged values and the reasons each one was flagged are in `generated/outlier_details.csv`.

## Attrition: how common was it?

The confirmed target is `left_within_12m`: `1` means left within 12 months, and `0` means stayed. There are **349 leavers out of 2,389 observations (14.61%)** and 2,040 stayers. That is about one leaver for every seven observations.

A model that predicts “stayed” for everyone would be about 85.4% accurate, but it would find no leavers. Later comparisons must therefore measure how many leavers are found, not accuracy alone.

![Attrition label distribution](generated/figures/target_distribution.png)

## Information that would give the model the answer

**Data leakage** happens when a model is given information that reveals the outcome or would not be available when the prediction is made.

These fields describe events after an employee leaves and must be excluded from attrition model inputs: `exit_date`, `resignation_date`, `exit_reason`, `exit_interview_done`, and `fnf_settlement_amount`. We also exclude `employee_id`, `row_id`, and the alternate outcome label from predictive inputs. The complete register reviews all 42 columns: `generated/leakage_register.csv`.

## Other data-quality issues

| Finding | Decision |
|---|---|
| Six business-record duplicate rows appear when `row_id` is ignored, despite zero exact full-row duplicates. | Keep audit data unchanged; apply a documented business-record rule in Task 2 and keep an audit trail. |
| Four negative tenure values and ten negative experience values. | Flag and verify with the source team; do not silently replace with zero. |
| `wfh_share_3m` is below zero in 190 rows and above one in 23 rows. | Confirm the field's definition and valid range before use. |
| `attendance_rate_3m` is above one in two rows. | Check the rate definition and flag the records. |
| Three pay values conflict with the provisional pay-to-`comp_ratio` check. | Verify with payroll; never automatically divide the values by ten. |
| Thirteen records use non-standard location spellings. | Normalize only with an approved mapping. |
| 41 dated resignations occur before the observation date. | Confirm date meanings before building time-based features. |

Detailed supporting records are in `generated/data_quality_issue_details.csv`, `generated/duplicate_review.csv`, and `generated/invalid_values.csv`.

## Decisions carried into Task 2

| Decision | What Task 2 must do |
|---|---|
| D01–D03: Missing values | Preserve meaningful missingness, add flags where justified, and fit fill values on training data only. Do not drop columns solely because of missingness. |
| D04: Outliers | Flag and investigate unusual values; do not automatically delete or cap them. |
| D05: Target | Use `left_within_12m`, confirmed by the project owner. |
| D06–D07: Leakage and identifiers | Exclude post-outcome fields, identifiers and the alternate outcome label from model inputs. |
| D08: Repeated records | Use a documented business-record rule; a repeated ID by itself is not proof of duplication. |
| D09: Invalid values and categories | Check ranges and use an approved location mapping; keep unresolved values flagged. |
| D10: Timing | Define what information is available at prediction time and verify time windows before creating features. |

## Handoff to Task 2

Task 1 records **what is wrong or uncertain and what should happen next**. Task 2 implements those decisions in a reusable, versioned pipeline. Learned filling values, scaling, encodings and reference values must be fitted on training data only, then reused unchanged for validation, test and future records.

## Questions still open

The location mapping, causes of negative or out-of-range values, department prior-rate time window, engagement-score missingness, and the exact meaning of `comp_ratio` still need source-owner confirmation.

## Post-audit check against planted issues

After the independent audit was complete, the authorized answer-key comparison checked seven categories. All seven matched: commute distance missingness (55), education missingness (15), location variants (13), salary typos (3), attendance above one (2), negative tenure (4), and business-record duplicates (6). The original audit and later comparison remain separate.

The full-sheet commute count is 55 of 2,389 (2.30%). The supplied key's approximately 2.8% does not reconcile with the currently documented subgroup count: 50 missing values among 2,000 historical rows equals 2.50%, while five additional missing values are outside that batch. The report therefore uses 2.30% for the full sheet and leaves the key's denominator discrepancy documented rather than claiming it is resolved.

**Supporting files:** `generated/decision_register.csv`, `generated/missing_values.csv`, `generated/outlier_summary.csv`, `generated/leakage_register.csv`, `generated/duplicate_review.csv`, `generated/invalid_values.csv`, and `generated/planted_issue_validation.csv`. Detailed record-level files are generated locally and intentionally excluded from version control.
