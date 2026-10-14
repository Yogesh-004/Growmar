# Task 4: Attendance Anomaly Detection (`anomaly-1.0.0`)

Unsupervised Isolation Forest over employee-month attendance records, evaluated with
seeded injected anomalies. Labels never train the model — they only grade it.

## Layout

- `src/anomaly/` — `config`, `features` (periods, preprocessor, score convention),
  `injection` (deterministic per-category injection), `model` (forest build/save/score),
  `evaluation` (threshold tables, selection, metric formulas).
- `configs/task4_config.json` — sheets, seeds, periods, injection rates, forest settings,
  flag-rate candidates, false-alarm policy.
- `scripts/inject_anomalies.py` — standalone repeatable injection (byte-identical reruns).
- `scripts/run_anomaly_detection.py` — full workflow (`--mode run`) and reload scoring
  (`--mode score --input …`).
- `reports/ATTENDANCE_ANOMALY_REPORT.md` + `reports/generated/` — tables, PR chart,
  score plot; `TASK4_SCHEMA_MAPPING.md` documents the three source sheets.
- `tests/test_anomaly.py` — synthetic fixtures only.

## Periods and rule

Train 2025-04…2025-12 (fit only) → calibration 2026-01…2026-05 (threshold choice) →
final 2026-06…2026-09 (frozen-threshold check). Reference-labelled rows are excluded
from all pools. Operating rule: highest calibration catch rate with false alarms ≤ 10%
(experimental policy, not an HR requirement).

## Commands (from the project root)

```powershell
python task4_attendance_anomaly/scripts/inject_anomalies.py --pool both
python task4_attendance_anomaly/scripts/run_anomaly_detection.py
python task4_attendance_anomaly/scripts/run_anomaly_detection.py --mode score --input data/raw/CortexHR_AI_ML_Dataset.xlsx --output data/processed/scored_new.csv
pytest -q task4_attendance_anomaly/tests
```

Measured result: frozen threshold catches 58% of injected anomalies at 7.0% false alarms
(10.1% of final rows flagged). Model + metadata in `artifacts/task4/` (local, git-ignored);
identifier-bearing manifests live only in git-ignored `data/processed/task4_injection/`.
