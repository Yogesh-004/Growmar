"""Full Task 4 workflow: train, calibrate thresholds, freeze, final evaluation, report.

Also supports `--mode score` to score new monthly rows with the saved artifact
(without refitting). Run from the project root.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _entry in (PROJECT_ROOT / "task4_attendance_anomaly" / "src",
               PROJECT_ROOT / "task2_feature_pipeline" / "src",
               PROJECT_ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from anomaly import evaluation as ev  # noqa: E402
from anomaly import injection as inj  # noqa: E402
from anomaly.config import Task4Config  # noqa: E402
from anomaly.features import (  # noqa: E402
    anomaly_scores,
    attendance_feature_columns,
    build_preprocessor,
    period_bounds,
    split_periods,
)
from anomaly.model import build_detector, load_detector, save_detector  # noqa: E402
from features.monthly import add_monthly_features, prepare_monthly_frame  # noqa: E402
from features.monthly import load_monthly_sheets  # noqa: E402


def _project_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return root / p if not p.is_absolute() else p


def _load_reference_ids(workbook: Path, sheet: str) -> set[str]:
    try:
        labels = pd.read_excel(workbook, sheet_name=sheet)
    except ValueError:
        return set()
    if "attendance_feature_id" not in labels.columns:
        return set()
    return set(labels["attendance_feature_id"].astype(str))


def _feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    cols = attendance_feature_columns()
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"Feature frame lacks column(s): {', '.join(missing)}")
    return frame[cols].copy()


def _save_pr_plot(table: pd.DataFrame, threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(table["catch_rate"], table["precision"], marker="o")
    sel = table.loc[(table["threshold"] - threshold).abs().idxmin()]
    ax.scatter([sel["catch_rate"]], [sel["precision"]], s=90, label=f"selected (t={threshold:.3f})")
    ax.set_xlabel("Catch rate (injected anomalies found)")
    ax.set_ylabel("Precision (flagged rows that were injected)")
    ax.set_title("Precision-recall trade-off (calibration)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_score_plot(scores: np.ndarray, is_injected: np.ndarray, threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(scores[~is_injected], bins=40, alpha=0.6, label="normal")
    ax.hist(scores[is_injected], bins=20, alpha=0.8, label="injected")
    ax.axvline(threshold, linestyle="--", label=f"threshold {threshold:.3f}")
    ax.set_xlabel("Anomaly score (higher = more unusual)")
    ax.set_ylabel("Monthly records")
    ax.set_title("Score distribution (final evaluation)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cmd_run(args: argparse.Namespace, cfg: Task4Config, root: Path) -> int:
    out_dir = _project_path(root, cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = _project_path(root, cfg.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    inj_dir = _project_path(root, "data/processed/task4_injection")
    inj_dir.mkdir(parents=True, exist_ok=True)

    workbook = _project_path(root, cfg.dataset_file)
    att, pay = load_monthly_sheets(workbook)
    merged, dedup_log, join_stats = prepare_monthly_frame(att, pay)
    ref_ids = _load_reference_ids(workbook, cfg.labels_sheet)
    parts = split_periods(merged, cfg.train_start_month, cfg.train_end_month, cfg.calibration_end_month)
    bounds = period_bounds(parts)
    print(f"Periods: train {bounds['train_rows']} rows ({bounds['train_min']}..{bounds['train_max']}), "
          f"calibration {bounds['calibration_rows']}, final {bounds['final_rows']}. "
          f"Reference-labelled rows set aside: {len(ref_ids)}.")

    # --- Train on unmodified training-period rows (labels never inputs) ---
    train_feat = add_monthly_features(parts["train"])
    X_train = _feature_matrix(train_feat)
    detector = build_detector(
        build_preprocessor(),
        n_estimators=cfg.iforest_n_estimators,
        max_samples=cfg.iforest_max_samples,
        max_features=cfg.iforest_max_features,
        random_state=cfg.random_state,
    )
    detector.fit(X_train)
    save_detector(detector, art_dir / cfg.pipeline_filename)

    # --- Calibration: inject, score, compare thresholds, freeze one ---
    cal_pool = parts["calibration"]
    per_cat = max(1, int(round(len(cal_pool) * cfg.injection_rate_per_category)))
    cal_inj, cal_manifest = inj.inject_pool(cal_pool, ref_ids, per_cat, cfg.injection_seed)
    cal_manifest.to_csv(inj_dir / "injection_manifest_calibration.csv", index=False)
    cal_feat = add_monthly_features(cal_inj)
    cal_ids = set(cal_manifest["attendance_feature_id"].astype(str))
    cal_truth = cal_feat["attendance_feature_id"].astype(str).isin(cal_ids).to_numpy()
    cal_scores = anomaly_scores(detector, _feature_matrix(cal_feat))
    thresholds = ev.thresholds_from_flag_rates(cal_scores, list(cfg.threshold_flag_rates))
    cal_table = ev.threshold_table(cal_scores, cal_truth, thresholds)
    cal_table.to_csv(out_dir / "threshold_comparison.csv", index=False)
    selection = ev.select_threshold(cal_table, max_false_alarm=cfg.max_false_alarm,
                                    max_flagged_fraction=cfg.max_flagged_fraction)
    frozen = float(selection["threshold"])
    print(f"Calibration: {cal_truth.sum()} injected in {len(cal_truth)} rows. "
          f"Frozen threshold {frozen:.4f} [{selection['rule']}]")
    _save_pr_plot(cal_table, frozen, out_dir / "precision_recall.png")

    # --- Final: independently injected pool, frozen threshold, no reselection ---
    fin_pool = parts["final"]
    fin_per_cat = max(1, int(round(len(fin_pool) * cfg.injection_rate_per_category)))
    fin_inj, fin_manifest = inj.inject_pool(fin_pool, ref_ids, fin_per_cat, cfg.injection_seed + 1000)
    fin_manifest.to_csv(inj_dir / "injection_manifest_final.csv", index=False)
    fin_feat = add_monthly_features(fin_inj)
    fin_ids = set(fin_manifest["attendance_feature_id"].astype(str))
    fin_truth = fin_feat["attendance_feature_id"].astype(str).isin(fin_ids).to_numpy()
    fin_scores = anomaly_scores(detector, _feature_matrix(fin_feat))
    fin_metrics = ev.threshold_metrics(fin_scores, fin_truth, frozen)
    pd.DataFrame([{**fin_metrics, "pool": "final", "threshold_source": "frozen from calibration"}]).to_csv(
        out_dir / "final_evaluation.csv", index=False)
    cal_row = ev.threshold_metrics(cal_scores, cal_truth, frozen)
    pd.DataFrame([{**cal_row, "pool": "calibration", "threshold_source": "selected here"}]).to_csv(
        out_dir / "calibration_metrics.csv", index=False)
    _save_score_plot(fin_scores, fin_truth, frozen, out_dir / "score_distribution.png")
    scored = pd.DataFrame({
        "attendance_feature_id": fin_feat["attendance_feature_id"].astype(str),
        "month": fin_feat["month"].astype(str),
        "anomaly_score": fin_scores,
        "threshold": frozen,
        "is_anomaly": (fin_scores >= frozen).astype(int),
        "is_injected": fin_truth.astype(int),
    })
    scored.to_csv(out_dir / "scored_final.csv", index=False)

    # --- Informational agreement with reference labels (non-decisive context) ---
    ref_rows = merged[merged["attendance_feature_id"].astype(str).isin(ref_ids)]
    ref_agreement = None
    if not ref_rows.empty:
        ref_feat = add_monthly_features(ref_rows)
        ref_scores = anomaly_scores(detector, _feature_matrix(ref_feat))
        ref_agreement = {
            "n_reference_rows": len(ref_rows),
            "flagged_at_frozen_threshold": int((ref_scores >= frozen).sum()),
        }

    summary = inj.injection_summary(pd.concat([cal_manifest, fin_manifest], ignore_index=True))
    summary.to_csv(out_dir / "injection_summary.csv", index=False)

    metadata = {
        "anomaly_package_version": "anomaly-1.0.0",
        "pipeline_artifact": cfg.pipeline_filename,
        "source_sheets": [cfg.attendance_sheet, cfg.payroll_sheet],
        "reference_labels_sheet": cfg.labels_sheet,
        "daily_examples_sheet": cfg.daily_sheet,
        "periods": bounds,
        "train_boundaries": {"train": [cfg.train_start_month, cfg.train_end_month],
                             "calibration_end": cfg.calibration_end_month},
        "feature_columns": attendance_feature_columns(),
        "preprocessing": "median imputation + RobustScaler, fitted on training months only",
        "iforest": {"n_estimators": cfg.iforest_n_estimators, "max_samples": cfg.iforest_max_samples,
                    "max_features": cfg.iforest_max_features, "random_state": cfg.random_state,
                    "contamination": "auto (predict() unused; explicit threshold governs flags)"},
        "score_convention": "anomaly_score = -decision_function; higher means more anomalous",
        "threshold": frozen,
        "selection_rule": selection["rule"],
        "selection_fallback": bool(selection["fallback"]),
        "calibration_metrics": cal_row,
        "final_metrics": fin_metrics,
        "reference_label_agreement": ref_agreement,
        "dedup_log": dedup_log,
        "join_stats": join_stats,
        "injection": {"seed_calibration": cfg.injection_seed, "seed_final": cfg.injection_seed + 1000,
                      "rate_per_category": cfg.injection_rate_per_category,
                      "per_category_counts": {"calibration": per_cat, "final": fin_per_cat},
                      "reference_rows_excluded": len(ref_ids)},
        "assumptions": [
            "Impossible-hours range [19, 22] is experimental, not an HR rule.",
            "Long-absence streaks mirror reference-note magnitudes; no cause is claimed.",
            "Trailing windows restart at pool starts (months_of_history records depth).",
            "Reference labels are context only, never inputs or selection criteria.",
        ],
    }
    (art_dir / cfg.metadata_filename).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report_path = _project_path(root, cfg.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(cfg, bounds, dedup_log, join_stats, cal_table,
                                         selection, cal_row, fin_metrics, ref_agreement,
                                         summary, frozen, len(ref_ids)), encoding="utf-8")
    flagged_pct = fin_metrics["flagged_pct"]
    flagged_txt = f"{flagged_pct:.1f}%" if not np.isnan(flagged_pct) else "n/a"
    print(f"Final: catch {fin_metrics['catch_rate']:.0%} with false alarms {fin_metrics['false_alarm_rate']:.1%}; "
          f"flagged {flagged_txt} of final rows.")
    print(f"Report: {report_path} | Model: {art_dir / cfg.pipeline_filename}")
    return 0


def cmd_score(args: argparse.Namespace, cfg: Task4Config, root: Path) -> int:
    art_dir = _project_path(root, args.artifacts_dir or cfg.artifacts_dir)
    meta_path = art_dir / cfg.metadata_filename
    if not meta_path.exists():
        print(f"ERROR: metadata not found: {meta_path}. Run the full workflow first.", file=sys.stderr)
        return 2
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    detector = load_detector(art_dir / cfg.pipeline_filename)
    threshold = float(metadata["threshold"])
    src = Path(args.input)
    if src.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        att, pay = load_monthly_sheets(src if src.is_absolute() else root / src)
        merged, _, _ = prepare_monthly_frame(att, pay)
        frame = add_monthly_features(merged)
    else:
        frame = pd.read_csv(src if src.is_absolute() else root / src)
    X = _feature_matrix(frame)
    scores = anomaly_scores(detector, X)
    keys = pd.DataFrame({
        "attendance_feature_id": frame["attendance_feature_id"].astype(str)
        if "attendance_feature_id" in frame.columns else frame.index.astype(str),
        "month": frame["month"].astype(str) if "month" in frame.columns else "",
    })
    out = keys.assign(anomaly_score=scores, threshold=threshold,
                      is_anomaly=(scores >= threshold).astype(int))
    dest = Path(args.output) if args.output else _project_path(root, cfg.output_dir) / "scored_new.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"Scored {len(out)} rows x threshold {threshold:.4f} -> {dest} (no refitting).")
    return 0


def render_report(cfg, bounds, dedup_log, join_stats, cal_table, selection, cal_row, fin_metrics,
                  ref_agreement, summary, frozen, n_ref) -> str:
    cal_lines = []
    for r in cal_table.itertuples():
        cal_lines.append(
            f"| {r.threshold:.4f} | {int(r.n_flagged)} ({_pctval(r.flagged_pct)}) | "
            f"{_f(r.catch_rate)} | {_f(r.false_alarm_rate)} | {_f3(r.precision)} | {_f1(r.f1)} |"
        )
    sum_lines = [
        f"| {r.category} | {int(r.rows_affected)} |" for r in summary.itertuples()
    ]

    ref_line = (
        f"{ref_agreement['flagged_at_frozen_threshold']} of {ref_agreement['n_reference_rows']} "
        "reference-labelled rows score above the frozen threshold (context only; these labels "
        "never trained the model or chose the threshold)."
        if ref_agreement else "No reference-labelled rows available for context."
    )
    return f"""# Attendance Anomaly Detection — Aura Workforce (Task 4)

> Plain-language report for HR and technical reviewers. Anomalies below are injected test cases with known labels; the detector itself never sees labels during training. No employee identifiers are disclosed.

## 1. Executive summary

We trained an Isolation Forest on {bounds['train_rows']} normal monthly attendance records ({bounds['train_min']}…{bounds['train_max']}) and tested it on deliberately injected attendance problems. Operating threshold **{frozen:.4f}** [{selection['rule']}]. Final independent check: caught **{_f(fin_metrics['catch_rate'])}** of injected anomalies while wrongly flagging **{_f(fin_metrics['false_alarm_rate'])}** of normal records ({_pctval(fin_metrics['flagged_pct'])} of all final rows flagged). This is a detection baseline, not a production monitor.

## 2. Data sources and monthly observation grain

Workbook `{cfg.dataset_file}`: `{cfg.attendance_sheet}` plus `{cfg.payroll_sheet}` for join context (payroll dedup removed {dedup_log['removed_duplicate_lines']} re-processed lines; join matched {join_stats['matched_keys']} of {join_stats['attendance_keys']} attendance keys 1:1, {join_stats['payroll_only_keys']} payroll-only key excluded with a count logged). `{cfg.labels_sheet}` ({n_ref} reference labels) was used only to keep evaluation pools clean and for schema context — never as a model input. `{cfg.daily_sheet}` informed the injection ranges. One row is one employee-month; hyperlinks never opened; workbook unmodified.

## 3. Features used and limitations

Attendance-focused monthly features (rate, late arrivals, hours, overtime, presence/absence counts, trailing 3-month means, history depth, invalid flag; pay fields excluded by scope, disputed job levels excluded with a mismatch flag). Trailing windows use each row's month plus prior months only and restart at pool boundaries. Median imputation + RobustScaler fitted on training months only.

## 4. Three injection categories and their definitions

- **Impossible hours**: `avg_hours_worked` set into [19, 22] — an experimental range (daily evidence shows anomalies from 17 h; ordinary months cluster near 8.6), not an HR rule.
- **Long absence**: streak of 10–20 days mapped coherently onto `longest_unpaid_streak_days`, `unpaid_absence_days`, `present_days`, `attendance_rate` (mirrors reference-note magnitudes; no cause claimed).
- **Sudden change**: attendance dropped 0.40–0.60 below the employee's own trailing mean (needs 3+ prior months and a base mean ≥ 0.5).

| Category | Injected rows (calibration + final) |
|---|---:|
{chr(10).join(sum_lines)}

Fixed seeds make reruns byte-identical; originals never modified (copies only).

## 5. How the Isolation Forest works, explained simply

The forest learns what ordinary months look like from training data, then scores each new month by how hard it is to separate from the crowd: isolated, unusual months score higher. Our `anomaly_score` is the negated model output so **higher always means more unusual**. Labels only grade the answers afterwards.

## 6. Training, calibration, and final evaluation separation

Train ({bounds['train_min']}…{bounds['train_max']}) fitted preprocessing + forest on clean rows. Calibration ({bounds['calibration_min']}…{bounds['calibration_max']}) compared thresholds on fresh injections. Final ({bounds['final_min']}…{bounds['final_max']}) used independently injected rows at the frozen threshold — final labels never influenced the choice.

## 7. Candidate threshold comparison

| Threshold | Flagged | Catch rate | False-alarm rate | Precision | F1 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(cal_lines)}

Catch rate = caught ÷ injected; false-alarm rate = wrongly flagged ÷ normal; precision = caught ÷ flagged. Zero denominators yield n/a, never fabricated.

## 8. Precision-recall trade-off and selected threshold

![Precision-recall](generated/precision_recall.png)

Threshold **{frozen:.4f}** gives calibration catch {_f(cal_row['catch_rate'])} at false alarms {_f(cal_row['false_alarm_rate'])}.

## 9. Catch rate and false-alarm rate

Calibration: catch {_f(cal_row['catch_rate'])}, false alarms {_f(cal_row['false_alarm_rate'])}. Final: catch {_f(fin_metrics['catch_rate'])}, false alarms {_f(fin_metrics['false_alarm_rate'])}.

## 10. Percentage of records flagged

Final flagged {_pctval(fin_metrics['flagged_pct'])} of rows. Flagging a fifth or more of records would erode HR trust; the operating point stays well clear of that.

## 11. Final independent evaluation results

| Metric | Value |
|---|---:|
| Injected anomalies | {int(fin_metrics['n_injected'])} |
| Caught / missed | {int(fin_metrics['n_caught'])} / {int(fin_metrics['n_missed'])} |
| Catch rate | {_f(fin_metrics['catch_rate'])} |
| False alarms / normal | {int(fin_metrics['n_false_alarms'])} / {int(fin_metrics['n_normal'])} |
| False-alarm rate | {_f(fin_metrics['false_alarm_rate'])} |
| Precision / F1 | {_f3(fin_metrics['precision'])} / {_f1(fin_metrics['f1'])} |

![Score distribution](generated/score_distribution.png)

Reference context: {ref_line}

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
"""


def _f(x: float) -> str:
    return f"{x:.0%}" if not np.isnan(x) else "n/a"


def _f3(x: float) -> str:
    return f"{x:.3f}" if not np.isnan(x) else "n/a"


def _pctval(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}%" if not np.isnan(x) else "n/a"


def _f1(x: float) -> str:
    return f"{x:.3f}" if not np.isnan(x) else "n/a"


def build_parser(argv: list[str] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="task4_attendance_anomaly/configs/task4_config.json")
    parser.add_argument("--mode", choices=["run", "score"], default="run")
    parser.add_argument("--input", default=None, help="score mode: workbook or monthly CSV")
    parser.add_argument("--output", default=None)
    parser.add_argument("--artifacts-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = Task4Config.from_json(config_path)
    root = PROJECT_ROOT
    try:
        if args.mode == "score":
            return cmd_score(args, cfg, root)
        return cmd_run(args, cfg, root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
