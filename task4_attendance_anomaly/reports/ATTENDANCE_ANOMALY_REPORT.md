# Attendance Anomaly Detection — Aura Workforce (Task 4)

> Plain-language report for HR and technical reviewers. Anomalies below are injected test cases with known labels; the detector itself never sees labels during training. No employee identifiers are disclosed.

## 1. Executive summary

We trained an Isolation Forest on 3559 normal monthly attendance records (2025-04-01 00:00:00…2025-12-01 00:00:00) and tested it on deliberately injected attendance problems. Operating threshold **0.0004** [max catch rate with false-alarm rate <= 10% (precision tie-break); constraint satisfied]. Final independent check: caught **58%** of injected anomalies while wrongly flagging **7%** of normal records (10.1% of all final rows flagged). This is a detection baseline, not a production monitor.

## 2. Data sources and monthly observation grain

Workbook `data/raw/CortexHR_AI_ML_Dataset.xlsx`: `attendance_monthly_features` plus `payroll_monthly_features` for join context (payroll dedup removed 15 re-processed lines; join matched 7339 of 7339 attendance keys 1:1, 1 payroll-only key excluded with a count logged). `attendance_anomaly_labels` (150 reference labels) was used only to keep evaluation pools clean and for schema context — never as a model input. `attendance_daily_examples` informed the injection ranges. One row is one employee-month; hyperlinks never opened; workbook unmodified.

## 3. Features used and limitations

Attendance-focused monthly features (rate, late arrivals, hours, overtime, presence/absence counts, trailing 3-month means, history depth, invalid flag; pay fields excluded by scope, disputed job levels excluded with a mismatch flag). Trailing windows use each row's month plus prior months only and restart at pool boundaries. Median imputation + RobustScaler fitted on training months only.

## 4. Three injection categories and their definitions

- **Impossible hours**: `avg_hours_worked` set into [19, 22] — an experimental range (daily evidence shows anomalies from 17 h; ordinary months cluster near 8.6), not an HR rule.
- **Long absence**: streak of 10–20 days mapped coherently onto `longest_unpaid_streak_days`, `unpaid_absence_days`, `present_days`, `attendance_rate` (mirrors reference-note magnitudes; no cause claimed).
- **Sudden change**: attendance dropped 0.40–0.60 below the employee's own trailing mean (needs 3+ prior months and a base mean ≥ 0.5).

| Category | Injected rows (calibration + final) |
|---|---:|
| impossible_hours | 75 |
| long_absence | 75 |
| sudden_change | 75 |

Fixed seeds make reruns byte-identical; originals never modified (copies only).

## 5. How the Isolation Forest works, explained simply

The forest learns what ordinary months look like from training data, then scores each new month by how hard it is to separate from the crowd: isolated, unusual months score higher. Our `anomaly_score` is the negated model output so **higher always means more unusual**. Labels only grade the answers afterwards.

## 6. Training, calibration, and final evaluation separation

Train (2025-04-01 00:00:00…2025-12-01 00:00:00) fitted preprocessing + forest on clean rows. Calibration (2026-01-01 00:00:00…2026-05-01 00:00:00) compared thresholds on fresh injections. Final (2026-06-01 00:00:00…2026-09-01 00:00:00) used independently injected rows at the frozen threshold — final labels never influenced the choice.

## 7. Candidate threshold comparison

| Threshold | Flagged | Catch rate | False-alarm rate | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| 0.1773 | 21 (1.0%) | 15% | 0% | 0.905 | 0.264 |
| 0.1635 | 41 (2.0%) | 22% | 1% | 0.659 | 0.329 |
| 0.0432 | 102 (5.0%) | 39% | 3% | 0.471 | 0.427 |
| 0.0004 | 203 (10.0%) | 53% | 7% | 0.320 | 0.399 |
| -0.0292 | 304 (15.0%) | 67% | 12% | 0.270 | 0.384 |
| -0.0502 | 405 (20.0%) | 77% | 16% | 0.235 | 0.360 |

Catch rate = caught ÷ injected; false-alarm rate = wrongly flagged ÷ normal; precision = caught ÷ flagged. Zero denominators yield n/a, never fabricated.

## 8. Precision-recall trade-off and selected threshold

![Precision-recall](generated/precision_recall.png)

Threshold **0.0004** gives calibration catch 53% at false alarms 7%.

## 9. Catch rate and false-alarm rate

Calibration: catch 53%, false alarms 7%. Final: catch 58%, false alarms 7%.

## 10. Percentage of records flagged

Final flagged 10.1% of rows. Flagging a fifth or more of records would erode HR trust; the operating point stays well clear of that.

## 11. Final independent evaluation results

| Metric | Value |
|---|---:|
| Injected anomalies | 102 |
| Caught / missed | 59 / 43 |
| Catch rate | 58% |
| False alarms / normal | 111 / 1587 |
| False-alarm rate | 7% |
| Precision / F1 | 0.347 / 0.434 |

![Score distribution](generated/score_distribution.png)

Reference context: 122 of 150 reference-labelled rows score above the frozen threshold (context only; these labels never trained the model or chose the threshold).

## 12. The saved model and metadata

`artifacts/task4/attendance_iforest.joblib` (preprocessing + forest) and `attendance_anomaly_metadata.json` (periods, features, parameters, score convention, frozen threshold + rule, both metric sets, injection seed). Reload scoring: `python task4_attendance_anomaly/scripts/run_anomaly_detection.py score --input <monthly rows>`.

## 13. Reproducibility and test instructions

`pytest -q task4_attendance_anomaly/tests`; `python task4_attendance_anomaly/scripts/inject_anomalies.py --pool both` reproduces injected copies + manifest byte-identically (same seed); full workflow reruns deterministically.

## 14. Known limitations and assumptions

Experimental hours range; streak magnitudes from reference notes, not HR policy; truncated trailing history at pool starts; reference labels assumed non-exhaustive; single chronological split; default 10% false-alarm policy is experimental, not an HR requirement; monthly_alignment-style manifests stay local and Git-ignored.

## 15. Connection to Phase 2 payroll anomaly detection and the future service

The saved detector, frozen threshold, and monthly feature discipline transfer directly: Phase 2 payroll work reuses the dedup/join/period machinery with pay-focused features and its own calibrated threshold. No Task 4 code changes are needed to score new months.

*Artifacts: `injection_summary.csv`, `threshold_comparison.csv`, `calibration_metrics.csv`, `final_evaluation.csv`, `precision_recall.png`, `score_distribution.png`, model + metadata in `artifacts/task4/`.*
*Row-level outputs (`scored_final.csv`, injection manifests, alignment files) are generated locally and intentionally excluded from version control.*
