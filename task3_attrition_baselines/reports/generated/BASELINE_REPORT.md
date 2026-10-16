# Attrition Baseline Models & Evaluation — Aura Workforce (Task 3)

> Baselines for HR and technical reviewers. Model choice came from training-only cross-validation; the locked test set was evaluated once at the end and never used for selection. No employee-level records are disclosed.

## 1. Executive summary

Compared five prespecified baselines for predicting who leaves: a dummy that always predicts "stays", logistic regression (plain and class-weighted), and a shallow depth-4 decision tree (plain and class-weighted). Selection rule (fixed before seeing test data): highest cross-validated leaver recall, tie-break F1. Winner: **logreg_balanced**. On the locked test set it found 60% of leavers at 25% precision (AUC 0.671). These are starting baselines for comparison, not production-ready models.

## 2. Dataset and target definition

Source `data/raw/CortexHR_AI_ML_Dataset.xlsx`, sheet `attrition_dataset` (2,389 rows). Target `left_within_12m` (1 = left within 12 months; verified: exactly the 349 rows with exit information left, 2,040 stayed). Overall leaver rate 14.61%. The alternate label `voluntary_left_within_12m` was never substituted; the project owner has confirmed `left_within_12m` as the Task 3 target (it was the working target at audit time).

## 3. Split and test-set protection

Reused Task 2's locked stratified 80/20 split (seed 42): train 1,911 (rate 14.60%), test 478 (rate 14.64%), verified against the manifest and Task 2 label files. All model comparison used training folds only; the test set was evaluated once, after selection, at the fixed 0.5 threshold.

## 4. Dummy baseline

`dummy_stays` always predicts stays: CV recall 0 by construction (precision/F1 0 with documented zero-division handling; ROC-AUC 0.5 is uninformative for a constant predictor). Its test accuracy is 0.854 while finding zero leavers — the reason accuracy alone misleads on imbalanced data.

## 5. Logistic regression results

Plain vs `class_weight="balanced"` (solver lbfgs, max_iter 5000, C 1). Weighting trades precision for recall: it flags more stayers wrongly but catches substantially more leavers. Preprocessing (imputers, scaler, encoder, department medians) was refit inside every fold.

## 6. Decision-tree results

Deliberately shallow (max_depth=4) plain vs balanced trees. Same trade-off pattern as logistic regression, with wider fold-to-fold variation typical of single trees.

## 7. Class-weight comparison

| Candidate | Recall (mean ± sd) | Precision (mean ± sd) | F1 (mean ± sd) | ROC-AUC (mean ± sd) |
|---|---:|---:|---:|---:|
| logreg_balanced | 0.613 ± 0.060 | 0.264 ± 0.024 | 0.368 ± 0.032 | 0.714 ± 0.047 |
| tree_balanced | 0.491 ± 0.077 | 0.203 ± 0.045 | 0.286 ± 0.051 | 0.607 ± 0.035 |
| logreg | 0.122 ± 0.056 | 0.472 ± 0.183 | 0.190 ± 0.080 | 0.712 ± 0.047 |
| tree | 0.082 ± 0.033 | 0.335 ± 0.088 | 0.130 ± 0.044 | 0.656 ± 0.036 |
| dummy_stays | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |

Weighting lifts leaver recall for both model families at a precision cost; ROC-AUC moves little, since ranking changes less than the 0.5 threshold decisions.

## 8. Five-fold cross-validation mean and spread

Stratified 5-fold CV on training rows only (shuffle, seed 42); spread is the sample standard deviation across folds. Full fold table: `cv_fold_metrics.csv`; summary: `cv_summary.csv`. Undefined fold metrics would be recorded as NaN with a reason — none occurred (every fold held both classes).

## 9. Final held-out evaluation

| Candidate | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| dummy_stays | 0.854 | 0.000 | 0.000 | 0.000 | 0.500 |
| logreg | 0.847 | 0.412 | 0.100 | 0.161 | 0.676 |
| logreg_balanced | 0.678 | 0.250 | 0.600 | 0.353 | 0.671 |
| tree | 0.841 | 0.385 | 0.143 | 0.208 | 0.670 |
| tree_balanced | 0.722 | 0.260 | 0.486 | 0.338 | 0.618 |

Test confusion matrix for logreg_balanced (rows = actual stayed/left, columns = predicted stayed/left): TN 282, FP 126, FN 28, TP 42. Shown for all prespecified candidates for transparency; selection used CV only.

## 10. Confusion matrix and ROC curve

![Confusion matrix](generated/confusion_matrix.png)

![ROC curve](generated/roc_curve.png)

![Model comparison](generated/model_comparison.png)

## 11. Recommended baseline and why

**logreg_balanced**, chosen by [max recall_mean, tie-break max f1_mean (CV only, test unseen)]. HR preferred catching more leavers even at the cost of some false alarms, so leaver recall led the decision while precision, F1 and ROC-AUC were monitored. Rerun: `python task3_attrition_baselines/scripts/run_baselines.py`.

## 12. Limitations and assumptions

Working (unapproved) target; provisional location map and `comp_ratio` meaning inherited from Task 2; department prior-rate window unverified; single 80/20 split (CV mitigates but does not replace outer validation); default 0.5 threshold untuned; shallow models only — stronger learners are Task 3+ work, not baselines.

## 13. Connection to subsequent tasks

Later models must beat these CV/test numbers on the same locked split and pipeline discipline (fold-fitted preprocessing, train-only references). Saved finalists in `artifacts/task3/` plus `selected_config.json` give Task 3+ a starting point; the monthly path stays separate for attendance work.

## Appendix — reference benchmarks vs measured

Reference values from `baseline_benchmarks` (labels, not training data) beside this run's CV means:

| Reference model | ROC-AUC | Precision | Recall | F1 | Note |
|---|---:|---:|---:|---|---|
| Dummy (always stays) | 0.500 ± 0.000 | 0.000 | 0.000 | 0.000 | Looks 85% accurate yet catches nobody: why accuracy alone misleads |
| Logistic Regression (class_weight=balanced) | 0.713 ± 0.010 | 0.260 | 0.630 | 0.368 | Clean features only (reference target) |
| Decision Tree (depth 4, balanced) | 0.650 ± 0.019 | 0.234 | 0.524 | 0.322 | Clean features only (reference target) |
| Random Forest (300 trees, balanced) | 0.715 ± 0.018 | 0.523 | 0.089 | 0.151 | Clean features only (reference target) |
| LEAKY Logistic Regression (adds exit information) | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | Too good to be true: this is data leakage |

The leaky row (perfect 1.000s) demonstrates data leakage, not a real target; the random-forest row is out of baseline scope. Agreement is approximate by design — references are guideposts, not pass/fail gates.

*Artifacts: `cv_fold_metrics.csv`, `cv_summary.csv`, `model_configs.csv`, `holdout_metrics.csv`, `selected_config.json`, `confusion_matrix.png`, `roc_curve.png`, `model_comparison.png`, fitted pipelines in `artifacts/task3/`.*
