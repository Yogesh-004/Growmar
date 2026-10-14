"""Task 1 decisions D01-D10 translated into preprocessing rules (no audit changes)."""
from __future__ import annotations

# D06/D07 (+ source per the leakage register's provisional verdict).
EXCLUDED_COLUMNS = (
    "row_id",
    "employee_id",
    "left_within_12m",
    "voluntary_left_within_12m",
    "exit_date",
    "resignation_date",
    "exit_reason",
    "exit_interview_done",
    "fnf_settlement_amount",
    "source",
)

# D01: missingness is meaningful exactly when no promotion/raise ever occurred.
STRUCTURAL_INDICATORS = {
    "months_since_last_promotion": "is_never_promoted",
    "last_raise_pct": "is_never_raised",
}
STRUCTURAL_SENTINEL = 9999.0

# D02: ordinary gaps are imputed; engagement additionally keeps a flag.
ENGAGEMENT_COLUMN = "last_engagement_score"
ENGAGEMENT_MISSING_FLAG = "engagement_missing"

# D09: provisional location mapping (Task 1 evidence; HR approval still pending).
# Only the observed case/punctuation variants plus one alternate name are mapped.
LOCATION_COLUMN = "work_location"
LOCATION_CANONICAL_MAP = {
    "BENGALURU": "Bengaluru",
    "Bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "chennai": "Chennai",
    "delhi ncr": "Delhi NCR",
    "Delhi-NCR": "Delhi NCR",
    "Madras": "Chennai",
}
LOCATION_MAP_STATUS = "provisional - HR confirmation needed (D09)"

# D09: rule-based validity flags (stateless, no fitting involved).
TENURE_INVALID_RULE = "tenure_months < 0"
WFH_INVALID_RULE = "wfh_share_3m < 0 or wfh_share_3m > 1"
ATTENDANCE_INVALID_RULE = "attendance_rate_3m < 0 or attendance_rate_3m > 1"

# Base feature columns present in attrition_dataset (order is deterministic).
NUMERIC_FEATURES = (
    "job_level",
    "tenure_months",
    "total_experience_years",
    "previous_employers_count",
    "commute_distance_km",
    "annual_fixed_pay",
    "comp_ratio",
    "raises_received",
    "last_raise_pct",
    "months_since_last_raise",
    "promotions_received",
    "months_since_last_promotion",
    "internal_transfers",
    "last_performance_rating",
    "last_engagement_score",
    "manager_changes_12m",
    "training_hours_12m",
    "attendance_rate_3m",
    "late_arrivals_3m",
    "avg_overtime_hours_per_month_3m",
    "avg_hours_worked_3m",
    "wfh_share_3m",
    "unpaid_leave_days_6m",
    "paid_leave_days_6m",
    "dept_attrition_rate_prior_12m",
)
CATEGORICAL_FEATURES = (
    "department_id",
    "role_id",
    "work_location",
    "work_mode",
    "employment_type",
    "highest_education",
)

# Engineered numeric features appended by the rules step (D01/D02/D09).
ENGINEERED_NUMERIC_FEATURES = (
    "is_never_promoted",
    "is_never_raised",
    "engagement_missing",
    "tenure_invalid",
    "wfh_invalid",
    "attendance_invalid",
    "salary_vs_department_median",
)
