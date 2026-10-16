# Task 4 Schema Mapping — Attendance Sources

Workbook `data/raw/CortexHR_AI_ML_Dataset.xlsx` (unmodified; explicit sheet reads only).

## 1. `attendance_monthly_features` (7,339 rows × 16)

One row per employee-month (`attendance_feature_id` technical key; `month` is month-start).
Fields: `scheduled_days` (1–23, never 0 in practice), `present_days`, `wfh_days`,
`half_days_unpaid`, `paid_leave_days`, `unpaid_absence_days` (0–22),
`longest_unpaid_streak_days` (0–22), `late_arrivals` (0–20), `avg_hours_worked` (0–18.94,
clustered near 8.6), `overtime_hours` (0–29), `attendance_rate` (0–1).
Months 2025-04 → 2026-09; 528 employees; no `(employee_id, month)` duplicates.

## 2. `attendance_anomaly_labels` (150 rows × 5)

Reference annotations on existing monthly rows: `attendance_feature_id`,
`employee_id`, `month`, `anomaly_type`
(ABSENCE_STREAK 40, IMPOSSIBLE_HOURS 35, LATE_SPIKE 30, SUDDEN_DROP 30, ZERO_ATTENDANCE 15),
`injection_note` (free text, e.g. streak lengths 8–20 days, late counts 15–20/month).
All 150 IDs match monthly rows (132 employees). Purpose: example/reference labels for
context and schema understanding — **not complete ground truth, never model inputs**.
Task 4 excludes these rows from training and evaluation pools and evaluates only its own
seeded injections; informational agreement is reported separately as context.

## 3. `attendance_daily_examples` (2,000 rows × 13)

Daily attendance events for 471 employees (2025-04-01 → 2026-09-30): `status`
(PRESENT/WFH), check-in/out times, `hours_worked` (−9.17…22.73), `overtime_hours`,
`late_flag`, `is_anomaly` (120 rows), `anomaly_type`
(ODD_HOUR_LOGIN, PRESENT_WITH_ZERO_HOURS, IMPOSSIBLE_HOURS, CHECKOUT_BEFORE_CHECKIN,
30 each). Used only to ground injection ranges (daily anomalies from 17 h upward;
streak evidence) — daily rows are not merged into the monthly model.

## 4. Keys and limitations

- Join key `(employee_id, month)` is unique in the monthly table; payroll duplicates are
  handled by Task 2's verified rule (see Task 2 monthly pipeline).
- No absence-cause field exists: long absences are patterns, never "unexplained" claims.
- No official working-hours rule exists: the [19, 22] injection range is experimental.
- No salary-revision dates exist: months-since-revision cannot be derived here.
