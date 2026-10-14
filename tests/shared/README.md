# tests/shared — genuinely shared test helpers

No shared fixtures currently exist. Task-specific tests live with their tasks:

- `task1_data_audit/tests/` — audit utility and planted-issue validation tests.
- `task2_feature_pipeline/tests/` — feature pipeline, monthly pipeline, and
  integration tests.

Add a helper here only when at least two task suites genuinely need it.
Path bootstrapping for all suites lives in the root `conftest.py`.
