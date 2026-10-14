"""Run Task 3 attrition baselines: CV on train, locked test evaluation at the end."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _entry in (PROJECT_ROOT / "task3_attrition_baselines" / "src", PROJECT_ROOT / "task2_feature_pipeline" / "src", PROJECT_ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from baselines import BASELINES_VERSION  # noqa: E402
from baselines.config import Task3Config  # noqa: E402
from baselines import data as data_mod  # noqa: E402
from baselines import evaluation as eval_mod  # noqa: E402
from baselines import models as models_mod  # noqa: E402
from features.pipeline import build_pipeline  # noqa: E402
from features.schema import load_attrition_sheet  # noqa: E402


def _project_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return root / p if not p.is_absolute() else p


def _save_confusion_plot(confusion: list[list[int]], candidate: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    mat = np.asarray(confusion)
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["predicted stays", "predicted leaves"])
    ax.set_yticklabels(["actually stayed", "actually left"])
    ax.set_title(f"Confusion matrix (test): {candidate}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, label="employees")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_roc_plot(fpr: np.ndarray, tpr: np.ndarray, auc: float, candidate: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(fpr, tpr, label=f"{candidate} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="chance")
    ax.set_xlabel("False positives (stayers flagged as leavers)")
    ax.set_ylabel("True positives (leavers found)")
    ax.set_title("ROC curve (test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_comparison_plot(summary: pd.DataFrame, path: Path) -> None:
    order = summary.sort_values("recall_mean", ascending=False)["candidate"].tolist()
    x = np.arange(len(order))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for k, metric in enumerate(("recall_mean", "precision_mean", "f1_mean")):
        vals = summary.set_index("candidate").loc[order, metric].to_numpy()
        ax.bar(x + (k - 1) * width, vals, width, label=metric.replace("_mean", ""))
    ax.set_xticks(x, order, rotation=18, ha="right")
    ax.set_ylabel("CV mean (leaver class)")
    ax.set_title("Baseline comparison (5-fold CV on training rows)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="task3_attrition_baselines/configs/task3_config.json")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 2
    cfg = Task3Config.from_json(config_path)
    out_dir = _project_path(PROJECT_ROOT, cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = _project_path(PROJECT_ROOT, cfg.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    workbook = _project_path(PROJECT_ROOT, cfg.dataset_file)
    raw = load_attrition_sheet(workbook, cfg.attrition_sheet)
    manifest = data_mod.load_manifest(_project_path(PROJECT_ROOT, cfg.split_manifest))
    encoding = data_mod.verify_target_encoding(raw, cfg.target_column)
    data_mod.verify_label_files(
        manifest, _project_path(PROJECT_ROOT, "data/processed/task2_features"), cfg.target_column
    )
    X_train, y_train, X_test, y_test, check = data_mod.reconstruct_split(raw, manifest, cfg.target_column)
    print(f"Locked split: train {check['train_rows']} (rate {check['train_rate']:.4f}), "
          f"test {check['test_rows']} (rate {check['test_rate']:.4f}). "
          f"Target {cfg.target_column}: {encoding['leavers']} left / {encoding['stayers']} stayed.")

    try:
        benchmarks = pd.read_excel(workbook, sheet_name=cfg.benchmark_sheet)
    except ValueError as exc:
        print(f"ERROR: benchmark sheet {cfg.benchmark_sheet!r} not readable: {exc}", file=sys.stderr)
        return 2

    candidates = models_mod.build_candidates(
        random_state=cfg.random_state,
        logreg_solver=cfg.logreg_solver,
        logreg_max_iter=cfg.logreg_max_iter,
        logreg_c=cfg.logreg_c,
        tree_max_depth=cfg.tree_max_depth,
    )
    pd.DataFrame(models_mod.describe_candidates()).to_csv(out_dir / "model_configs.csv", index=False)

    # --- Stage 1: training-only cross-validation (test set untouched) ---
    folds = eval_mod.run_cv(
        X_train, y_train, build_pipeline, candidates,
        n_splits=cfg.cv_folds, shuffle=cfg.cv_shuffle, random_state=cfg.random_state,
    )
    folds.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    summary = eval_mod.summarize_cv(folds)
    summary.to_csv(out_dir / "cv_summary.csv", index=False)
    selection = eval_mod.select_candidate(summary, cfg.selection_primary, cfg.selection_tiebreak)
    selected = str(selection["selected"])
    (art_dir / cfg.selected_filename).write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"CV complete ({cfg.cv_folds}-fold, train only). Selected by [{selection['rule']}]: {selected}.")

    # --- Stage 2: finalists fitted on full train, locked test evaluated once ---
    fitted = eval_mod.fit_finalists(X_train, y_train, build_pipeline, candidates)
    holdout_rows = []
    for name, pipe in fitted.items():
        joblib.dump(pipe, art_dir / f"{name}.joblib")
        res = eval_mod.evaluate_holdout(pipe, X_test, y_test, threshold=cfg.decision_threshold)
        holdout_rows.append({k: res[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc") } | {"candidate": name})
        if name == selected:
            selected_res = res
    holdout = pd.DataFrame(holdout_rows, columns=["candidate", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"])
    holdout.to_csv(out_dir / "holdout_metrics.csv", index=False)

    _save_confusion_plot(selected_res["confusion"], selected, out_dir / "confusion_matrix.png")
    fpr, tpr = eval_mod.roc_points(selected_res["y_true"], selected_res["y_proba"])
    _save_roc_plot(fpr, tpr, float(selected_res["roc_auc"]), selected, out_dir / "roc_curve.png")
    _save_comparison_plot(summary, out_dir / "model_comparison.png")

    report = render_report(
        cfg=cfg, check=check, encoding=encoding, summary=summary, selection=selection,
        holdout=holdout, selected=selected, selected_res=selected_res,
        benchmarks=benchmarks, threshold=cfg.decision_threshold,
    )
    report_path = _project_path(PROJECT_ROOT, cfg.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Test ({selected}): acc {selected_res['accuracy']:.3f} | P {selected_res['precision']:.3f} | "
          f"R {selected_res['recall']:.3f} | F1 {selected_res['f1']:.3f} | AUC {selected_res['roc_auc']:.3f}")
    print(f"Report: {report_path}")
    print("Note: model choice came from CV only; test scores did not select the model.")
    return 0


def render_report(*, cfg, check, encoding, summary, selection, holdout, selected, selected_res, benchmarks, threshold) -> str:
    cv_lines = []
    for r in summary.sort_values("recall_mean", ascending=False).itertuples():
        cv_lines.append(
            f"| {r.candidate} | {r.recall_mean:.3f} ± {r.recall_std:.3f} | {r.precision_mean:.3f} ± {r.precision_std:.3f} | "
            f"{r.f1_mean:.3f} ± {r.f1_std:.3f} | {r.roc_auc_mean:.3f} ± {r.roc_auc_std:.3f} |"
        )
    hold_lines = []
    for r in holdout.itertuples():
        hold_lines.append(
            f"| {r.candidate} | {r.accuracy:.3f} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | {r.roc_auc:.3f} |"
        )
    bench_lines = []
    for r in benchmarks.itertuples():
        bench_lines.append(
            f"| {r.model} | {float(r.roc_auc_mean):.3f} ± {float(r.roc_auc_sd):.3f} | {float(r.precision_mean):.3f} | "
            f"{float(r.recall_mean):.3f} | {float(r.f1_mean):.3f} | {r.comment} |"
        )
    tn, fp, fn, tp = (int(v) for v in np.asarray(selected_res["confusion"]).ravel())
    return f"""# Attrition Baseline Models & Evaluation — Aura Workforce (Task 3)

> Baselines for HR and technical reviewers. Model choice came from training-only cross-validation; the locked test set was evaluated once at the end and never used for selection. No employee-level records are disclosed.

## 1. Executive summary

We compared five prespecified baselines for predicting who leaves: a dummy that always predicts "stays", logistic regression (plain and class-weighted), and a shallow depth-{cfg.tree_max_depth} decision tree (plain and class-weighted). Selection rule (fixed before seeing test data): highest cross-validated leaver recall, tie-break F1. Winner: **{selected}**. On the locked test set it found {selected_res['recall']:.0%} of leavers at {selected_res['precision']:.0%} precision (AUC {selected_res['roc_auc']:.3f}). These are starting baselines for comparison, not production-ready models.

## 2. Dataset and target definition

Source `{cfg.dataset_file}`, sheet `{cfg.attrition_sheet}` ({check['train_rows'] + check['test_rows']:,} rows). Target `{cfg.target_column}` (1 = left within 12 months; verified: exactly the {encoding['leavers']:,} rows with exit information left, {encoding['stayers']:,} stayed). Overall leaver rate {encoding['leavers'] / (encoding['leavers'] + encoding['stayers']):.2%}. The alternate label `{cfg.alternate_target_column}` was never substituted; the project owner has confirmed `{cfg.target_column}` as the Task 3 target (it was the working target at audit time).

## 3. Split and test-set protection

Reused Task 2's locked stratified 80/20 split (seed {cfg.random_state}): train {check['train_rows']:,} (rate {check['train_rate']:.2%}), test {check['test_rows']:,} (rate {check['test_rate']:.2%}), verified against the manifest and Task 2 label files. All model comparison used training folds only; the test set was evaluated once, after selection, at the fixed {threshold:g} threshold.

## 4. Dummy baseline

`dummy_stays` always predicts stays: CV recall 0 by construction (precision/F1 0 with documented zero-division handling; ROC-AUC 0.5 is uninformative for a constant predictor). Its test accuracy is 0.854 while finding zero leavers — the reason accuracy alone misleads on imbalanced data.

## 5. Logistic regression results

Plain vs `class_weight="balanced"` (solver {cfg.logreg_solver}, max_iter {cfg.logreg_max_iter}, C {cfg.logreg_c:g}). Weighting trades precision for recall: it flags more stayers wrongly but catches substantially more leavers. Preprocessing (imputers, scaler, encoder, department medians) was refit inside every fold.

## 6. Decision-tree results

Deliberately shallow (max_depth={cfg.tree_max_depth}) plain vs balanced trees. Same trade-off pattern as logistic regression, with wider fold-to-fold variation typical of single trees.

## 7. Class-weight comparison

| Candidate | Recall (mean ± sd) | Precision (mean ± sd) | F1 (mean ± sd) | ROC-AUC (mean ± sd) |
|---|---:|---:|---:|---:|
{chr(10).join(cv_lines)}

Weighting lifts leaver recall for both model families at a precision cost; ROC-AUC moves little, since ranking changes less than the 0.5 threshold decisions.

## 8. Five-fold cross-validation mean and spread

Stratified 5-fold CV on training rows only (shuffle, seed {cfg.random_state}); spread is the sample standard deviation across folds. Full fold table: `cv_fold_metrics.csv`; summary: `cv_summary.csv`. Undefined fold metrics would be recorded as NaN with a reason — none occurred (every fold held both classes).

## 9. Final held-out evaluation

| Candidate | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
{chr(10).join(hold_lines)}

Test confusion matrix for {selected} (rows = actual stayed/left, columns = predicted stayed/left): TN {tn}, FP {fp}, FN {fn}, TP {tp}. Shown for all prespecified candidates for transparency; selection used CV only.

## 10. Confusion matrix and ROC curve

![Confusion matrix](generated/confusion_matrix.png)

![ROC curve](generated/roc_curve.png)

![Model comparison](generated/model_comparison.png)

## 11. Recommended baseline and why

**{selected}**, chosen by [{selection['rule']}]. HR preferred catching more leavers even at the cost of some false alarms, so leaver recall led the decision while precision, F1 and ROC-AUC were monitored. Rerun: `python task3_attrition_baselines/scripts/run_baselines.py`.

## 12. Limitations and assumptions

Working (unapproved) target; provisional location map and `comp_ratio` meaning inherited from Task 2; department prior-rate window unverified; single 80/20 split (CV mitigates but does not replace outer validation); default 0.5 threshold untuned; shallow models only — stronger learners are Task 3+ work, not baselines.

## 13. Connection to subsequent tasks

Later models must beat these CV/test numbers on the same locked split and pipeline discipline (fold-fitted preprocessing, train-only references). Saved finalists in `artifacts/task3/` plus `selected_config.json` give Task 3+ a starting point; the monthly path stays separate for attendance work.

## Appendix — reference benchmarks vs measured

Reference values from `{cfg.benchmark_sheet}` (labels, not training data) beside this run's CV means:

| Reference model | ROC-AUC | Precision | Recall | F1 | Note |
|---|---:|---:|---:|---|---|
{chr(10).join(bench_lines)}

The leaky row (perfect 1.000s) demonstrates data leakage, not a real target; the random-forest row is out of baseline scope. Agreement is approximate by design — references are guideposts, not pass/fail gates.

*Artifacts: `cv_fold_metrics.csv`, `cv_summary.csv`, `model_configs.csv`, `holdout_metrics.csv`, `selected_config.json`, `confusion_matrix.png`, `roc_curve.png`, `model_comparison.png`, fitted pipelines in `artifacts/task3/`.*
"""


if __name__ == "__main__":
    raise SystemExit(main())
