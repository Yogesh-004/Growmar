"""Command-line interface for the Task 2 feature package.

Usage (from the project root):
    python -m features.cli fit-transform --workbook data/raw/CortexHR_AI_ML_Dataset.xlsx
    python -m features.cli fit --workbook data/raw/CortexHR_AI_ML_Dataset.xlsx
    python -m features.cli transform --input data/processed/new_rows.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for _entry in (PROJECT_ROOT / "task2_feature_pipeline" / "src", PROJECT_ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from features import FEATURES_VERSION  # noqa: E402
from features import pipeline as pipeline_mod  # noqa: E402
from features import persistence, schema, split  # noqa: E402
from features.config import Task2Config  # noqa: E402

ASSUMPTIONS = [
    "Supervised frame is attrition_dataset rows; monthly attendance/payroll sheets were inspected but not merged (only 383/2383 employees overlap and their months start 2025-04, after most observation dates; merging would add mostly-missing columns and future-leakage risk).",
    "months_since_last_raise is the salary-revision feature; months_since_last_promotion is never substituted (different business meaning).",
    "comp_ratio interpretation and the location mapping stay provisional until HR confirms (D09).",
    "dept_attrition_rate_prior_12m is used as-is on the assumption its window ends at observation time (D10).",
    "source column excluded from inputs as a batch indicator (leakage-register verdict).",
]


def load_config(path: str | Path) -> Task2Config:
    cfg = Task2Config.from_json(path)
    if cfg.package_version != FEATURES_VERSION:
        raise ValueError(
            f"Config version {cfg.package_version!r} does not match package {FEATURES_VERSION!r}."
        )
    return cfg


def _project_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return root / p if not p.is_absolute() else p


def _read_transform_input(path: Path, sheet: str) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return schema.load_attrition_sheet(path, sheet)
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read transform input {path}: {type(exc).__name__}: {exc}") from exc


def cmd_fit(args: argparse.Namespace, cfg: Task2Config, root: Path) -> int:
    from features.schema import build_supervised_frame

    workbook = _project_path(root, args.workbook or cfg.dataset_file)
    raw = schema.load_attrition_sheet(workbook, cfg.attrition_sheet)
    X, y, keys = build_supervised_frame(raw, cfg.target_column)
    dropped = len(raw) - len(X)
    parts = split.stratified_split(X, y, keys, test_size=cfg.test_size, random_state=cfg.random_state)
    pipe = pipeline_mod.build_pipeline()
    pipe.fit(parts["X_train"])
    train_matrix = pipeline_mod.transform_frame(pipe, parts["X_train"])
    output_columns = list(train_matrix.columns)
    dept_refs = pipe.named_steps["dept_salary_reference"].references()
    metadata = persistence.metadata_record(
        version=cfg.package_version,
        pipeline_filename=cfg.pipeline_filename,
        source_sheets=[cfg.attrition_sheet],
        input_columns=pipeline_mod.input_columns(),
        output_columns=output_columns,
        train_config={
            "rows_total": len(X),
            "rows_dropped_missing_label": int(dropped),
            "rows_train": len(parts["X_train"]),
            "rows_test": len(parts["X_test"]),
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "stratified_on": cfg.target_column,
            "numeric": {"imputation": cfg.numeric_imputation, "scaler": cfg.scaler},
            "categorical": {"imputation": cfg.categorical_imputation, "encoder": cfg.encoder},
            "structural_sentinel": cfg.structural_sentinel,
        },
        dept_references=dept_refs,
        assumptions=ASSUMPTIONS,
    )
    artifacts = persistence.save_artifact(pipe, metadata, _project_path(root, cfg.artifacts_dir))
    manifest = split.split_manifest(
        version=cfg.package_version,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        keys_train=parts["keys_train"],
        keys_test=parts["keys_test"],
        y_train=parts["y_train"],
        y_test=parts["y_test"],
    )
    split.write_manifest(manifest, _project_path(root, cfg.artifacts_dir) / cfg.split_manifest_filename)
    print(f"Fit complete on {len(parts['X_train'])} training rows ({len(parts['X_test'])} held out).")
    print(f"Output columns: {len(output_columns)} | Artifact: {artifacts['pipeline']}")
    print(f"Manifest: train rate={manifest['train_target_rate']:.4f} test rate={manifest['test_target_rate']:.4f}")
    return 0


def cmd_transform(args: argparse.Namespace, cfg: Task2Config, root: Path) -> int:
    pipe, metadata = persistence.load_artifact(
        _project_path(root, args.artifacts_dir or cfg.artifacts_dir), cfg.pipeline_filename
    )
    if metadata.get("feature_package_version") != FEATURES_VERSION:
        print(
            f"WARNING: artifact version {metadata.get('feature_package_version')!r} "
            f"differs from package {FEATURES_VERSION!r}.",
            file=sys.stderr,
        )
    frame = _read_transform_input(Path(args.input), args.sheet or cfg.attrition_sheet)
    X = schema.validate_feature_frame(frame)
    out = pipeline_mod.transform_frame(pipe, X, metadata.get("output_columns"))
    dest = Path(args.output) if args.output else _project_path(root, cfg.output_dir) / "transformed_features.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"Transformed {len(out)} rows x {len(out.columns)} columns -> {dest} (no refitting).")
    return 0


def cmd_fit_transform(args: argparse.Namespace, cfg: Task2Config, root: Path) -> int:
    from features.schema import build_supervised_frame

    rc = cmd_fit(args, cfg, root)
    if rc != 0:
        return rc
    pipe, metadata = persistence.load_artifact(
        _project_path(root, cfg.artifacts_dir), cfg.pipeline_filename
    )
    workbook = _project_path(root, args.workbook or cfg.dataset_file)
    raw = schema.load_attrition_sheet(workbook, cfg.attrition_sheet)
    X, y, keys = build_supervised_frame(raw, cfg.target_column)
    parts = split.stratified_split(X, y, keys, test_size=cfg.test_size, random_state=cfg.random_state)
    out_dir = _project_path(root, args.output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = metadata.get("output_columns")
    train_matrix = pipeline_mod.transform_frame(pipe, parts["X_train"], cols)
    test_matrix = pipeline_mod.transform_frame(pipe, parts["X_test"], cols)
    train_matrix.to_csv(out_dir / "train_features.csv", index=False)
    test_matrix.to_csv(out_dir / "test_features.csv", index=False)
    for name, kk, yy in (("train", parts["keys_train"], parts["y_train"]), ("test", parts["keys_test"], parts["y_test"])):
        labels = kk.copy()
        labels[cfg.target_column] = yy.to_numpy()
        labels.to_csv(out_dir / f"{name}_labels.csv", index=False)
    print(f"Processed outputs in {out_dir}: train {train_matrix.shape}, test {test_matrix.shape}.")
    print("Labels/alignment kept separate from feature columns.")
    return 0


def cmd_monthly_fit_transform(args: argparse.Namespace, cfg: Task2Config, root: Path) -> int:
    from features import monthly as M

    workbook = _project_path(root, args.workbook or cfg.dataset_file)
    cutoff = args.cutoff_month or cfg.monthly_cutoff_month
    att, pay = M.load_monthly_sheets(workbook)
    merged, dedup_log, join_stats = M.prepare_monthly_frame(att, pay)
    train, test, train_info = M.chronological_split(merged, cutoff)
    pipe = M.build_monthly_pipeline()
    pipe.fit(train)
    cols = list(pipeline_mod.transform_frame(pipe, train).columns)
    metadata = M.monthly_metadata(
        output_columns=cols,
        train_info=train_info,
        dedup_log=dedup_log,
        join_stats=join_stats,
        dept_references=pipe.named_steps["dept_salary_reference"].references(),
        train_rows=len(train),
        test_rows=len(test),
    )
    artifacts = persistence.save_artifact(
        pipe, metadata, _project_path(root, cfg.artifacts_dir), cfg.monthly_metadata_filename
    )
    split.write_manifest(
        {
            "feature_package_version": FEATURES_VERSION,
            "monthly_component_version": M.MONTHLY_COMPONENT_VERSION,
            "strategy": "chronological holdout by month",
            "cutoff_month": cutoff,
            **train_info,
            "note": "Earlier months train; later months evaluate. Row matching via local alignment file only.",
        },
        _project_path(root, cfg.artifacts_dir) / cfg.monthly_manifest_filename,
    )
    out_dir = _project_path(root, args.output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_mod.transform_frame(pipe, train, cols).to_csv(out_dir / "monthly_train_features.csv", index=False)
    pipeline_mod.transform_frame(pipe, test, cols).to_csv(out_dir / "monthly_test_features.csv", index=False)
    align = pd.concat([M.alignment_frame(train, "train"), M.alignment_frame(test, "test")], ignore_index=True)
    align.to_csv(out_dir / "monthly_alignment.csv", index=False)
    print(f"Monthly fit complete: train {len(train)} rows ({train_info['train_month_min']}..{train_info['train_month_max']}), "
          f"holdout {len(test)} rows ({train_info['test_month_min']}..{train_info['test_month_max']}).")
    print(f"Dedup removed {dedup_log['removed_duplicate_lines']} payroll lines; "
          f"join matched {join_stats['matched_keys']}, payroll-only {join_stats['payroll_only_keys']}, "
          f"job mismatches {join_stats['job_level_mismatches']}.")
    print(f"Output columns: {len(cols)} | Artifact: {artifacts['pipeline']}")
    print("Alignment (with employee/month keys) kept separate in monthly_alignment.csv.")
    return 0


def cmd_monthly_transform(args: argparse.Namespace, cfg: Task2Config, root: Path) -> int:
    from features import monthly as M

    pipe, metadata = persistence.load_artifact(
        _project_path(root, args.artifacts_dir or cfg.artifacts_dir),
        cfg.monthly_pipeline_filename,
        cfg.monthly_metadata_filename,
    )
    if metadata.get("feature_package_version") != FEATURES_VERSION:
        print(
            f"WARNING: artifact version {metadata.get('feature_package_version')!r} "
            f"differs from package {FEATURES_VERSION!r}.",
            file=sys.stderr,
        )
    src = Path(args.input)
    if src.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        att, pay = M.load_monthly_sheets(src)
        merged, _, _ = M.prepare_monthly_frame(att, pay)
    else:
        try:
            merged = pd.read_csv(src)
        except Exception as exc:
            raise ValueError(f"Could not read monthly input {src}: {type(exc).__name__}: {exc}") from exc
        missing = [c for c in M.merged_input_columns() if c not in merged.columns]
        if missing:
            raise ValueError(f"Monthly input lacks column(s): {', '.join(missing)}")
    out = pipeline_mod.transform_frame(pipe, merged, metadata.get("output_columns"))
    dest = Path(args.output) if args.output else _project_path(root, cfg.output_dir) / "monthly_features.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"Transformed {len(out)} monthly rows x {len(out.columns)} columns -> {dest} (no refitting).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 2 reusable feature pipeline CLI.")
    parser.add_argument("--config", default="task2_feature_pipeline/configs/task2_config.json")
    sub = parser.add_subparsers(dest="command", required=True)
    fit_p = sub.add_parser("fit", help="Fit on training split only; save artifact + metadata.")
    fit_p.add_argument("--workbook", default=None)
    fit_p.set_defaults(func=cmd_fit)
    tr_p = sub.add_parser("transform", help="Transform new rows with a saved artifact (no refit).")
    tr_p.add_argument("--input", required=True)
    tr_p.add_argument("--sheet", default=None)
    tr_p.add_argument("--output", default=None)
    tr_p.add_argument("--artifacts-dir", default=None)
    tr_p.set_defaults(func=cmd_transform)
    ft_p = sub.add_parser("fit-transform", help="Full reproducible workflow incl. processed outputs.")
    ft_p.add_argument("--workbook", default=None)
    ft_p.add_argument("--output-dir", default=None)
    ft_p.set_defaults(func=cmd_fit_transform)
    mft_p = sub.add_parser(
        "monthly-fit-transform",
        help="Monthly path: dedup, join, chronological fit, processed outputs.",
    )
    mft_p.add_argument("--workbook", default=None)
    mft_p.add_argument("--output-dir", default=None)
    mft_p.add_argument("--cutoff-month", default=None, help="YYYY-MM; train holds earlier months.")
    mft_p.set_defaults(func=cmd_monthly_fit_transform)
    mt_p = sub.add_parser(
        "monthly-transform", help="Transform new monthly rows with a saved artifact (no refit)."
    )
    mt_p.add_argument("--input", required=True, help="Workbook (.xlsx) or pre-joined monthly CSV.")
    mt_p.add_argument("--output", default=None)
    mt_p.add_argument("--artifacts-dir", default=None)
    mt_p.set_defaults(func=cmd_monthly_transform)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = PROJECT_ROOT
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    try:
        cfg = load_config(config_path)
        return args.func(args, cfg, root)
    except (FileNotFoundError, ValueError, schema.SchemaError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
