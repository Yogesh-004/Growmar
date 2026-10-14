"""Repeatable anomaly injection on copies (originals never modified).

Reads the two monthly sheets plus reference-label IDs (excluded from every
pool, never used as model inputs). Writes injected feature copies and the
identifier-bearing manifest to a Git-ignored directory, and an aggregate
summary beside the reports. Fixed seeds make reruns identical.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _entry in (PROJECT_ROOT / "task4_attendance_anomaly" / "src",
               PROJECT_ROOT / "task2_feature_pipeline" / "src",
               PROJECT_ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from anomaly import injection as inj  # noqa: E402
from anomaly.config import Task4Config  # noqa: E402
from anomaly.features import split_periods  # noqa: E402
from features.monthly import (  # noqa: E402
    ATTENDANCE_SHEET,
    PAYROLL_SHEET,
    load_monthly_sheets,
    prepare_monthly_frame,
)


def _project_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return root / p if not p.is_absolute() else p


def reference_ids(workbook: Path, sheet: str) -> set[str]:
    """IDs of reference-labelled rows, used only to keep pools clean."""
    try:
        labels = pd.read_excel(workbook, sheet_name=sheet)
    except ValueError:
        return set()
    if "attendance_feature_id" not in labels.columns:
        return set()
    return set(labels["attendance_feature_id"].astype(str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="task4_attendance_anomaly/configs/task4_config.json")
    parser.add_argument("--workbook", default=None)
    parser.add_argument("--pool", choices=["calibration", "final", "both"], default="both")
    parser.add_argument("--per-category", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="data/processed/task4_injection",
                        help="Git-ignored directory for injected copies and the manifest")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = Task4Config.from_json(config_path)
    workbook = _project_path(PROJECT_ROOT, args.workbook or cfg.dataset_file)
    seed = args.seed if args.seed is not None else cfg.injection_seed

    att, pay = load_monthly_sheets(workbook)
    _ = (ATTENDANCE_SHEET, PAYROLL_SHEET)
    merged, _, _ = prepare_monthly_frame(att, pay)
    ref_ids = reference_ids(workbook, cfg.labels_sheet)
    parts = split_periods(merged, cfg.train_start_month, cfg.train_end_month, cfg.calibration_end_month)
    pools = {"calibration": parts["calibration"], "final": parts["final"]}
    if args.pool != "both":
        pools = {args.pool: pools[args.pool]}

    out_dir = _project_path(PROJECT_ROOT, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, pool in pools.items():
        pool_seed = seed if name == "calibration" else seed + 1000
        per_category = args.per_category or max(1, int(round(len(pool) * cfg.injection_rate_per_category)))
        injected, manifest = inj.inject_pool(pool, ref_ids, per_category, pool_seed)
        injected.to_csv(out_dir / f"injected_{name}.csv", index=False)
        manifest.to_csv(out_dir / f"injection_manifest_{name}.csv", index=False)
        summary = inj.injection_summary(manifest)
        summary["pool"] = name
        summaries.append(summary)
        print(f"{name}: pool {len(pool)} rows -> {manifest['attendance_feature_id'].nunique()} injected rows "
              f"({summary['rows_affected'].sum()} across {len(summary)} categories), seed {pool_seed}")
    combined = pd.concat(summaries, ignore_index=True)
    summary_dir = PROJECT_ROOT / cfg.output_dir
    summary_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(summary_dir / "injection_summary.csv", index=False)
    print(f"Aggregate summary: {summary_dir / 'injection_summary.csv'} "
          f"(manifests with alignment keys stay in Git-ignored {out_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
