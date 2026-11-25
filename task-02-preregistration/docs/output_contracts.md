# Output Contracts

These contracts are what the pipeline promises to produce. They were fixed before the real
data was analysed and are enforced by `tests/test_pipeline.py` on synthetic fixtures.

## Input: `data/raw/daily-demand-series.csv`

| Column | Type | Rule |
|---|---|---|
| `date` | `YYYY-MM-DD` | Required, unique, inside the observation window |
| `demand` | integer | Required, ≥ 0 |
| `marketing_event` | 0/1 | Required |
| `holiday` | 0/1 | Required |

A missing column raises `SchemaError`. Rows that break a rule are removed according to
preregistration §4, and each removal is counted.

## Stage outputs

| Path | Columns (exact order) |
|---|---|
| `data/interim/demand_clean.csv` | `date, demand, marketing_event, holiday` (rows that survived exclusions, sorted by date) |
| `data/processed/demand_features.csv` | `date, demand, marketing_event, holiday, weekend, trend` |
| `reports/results.json` | See below |

## `results.json`

```text
prereg_version     "prereg-v1"
input_sha256       hex string. Must equal the hash in preregistration.md for the real run
window             [start, end]
exclusions         n_raw, excluded_bad_date, excluded_outside_window,
                   excluded_duplicate_date, excluded_missing_value,
                   excluded_invalid_value, n_kept, excluded_share
status             "ok" | "insufficient_data"
descriptive        n_event_days, n_control_days, mean_event, mean_control
-- only when status == "ok" --
primary            formula, dropped_covariates, coef, ci95[2], p_one_sided ∈ [0,1],
                   significant (bool), n
decision           "reject_H0" | "fail_to_reject_H0"
secondary          welch_t, welch_p_one_sided, mean_difference, pct_lift,
                   hedges_g, practically_meaningful
robustness         R1_mann_whitney, R2_log_demand, R3_exclude_holidays, robust (bool)
diagnostics        shapiro_resid_p, durbin_watson
```

## Behavioural guarantees (checked by the tests on synthetic data)

| Guarantee | Test |
|---|---|
| Each exclusion rule fires, in the preregistered order | `test_exclusion_order_duplicates_caught_first`, `test_missing_and_invalid_values_excluded` |
| Derived features are correct (weekend, trend) | `test_features_contract` |
| The 95% CI covers a known true effect of +20 | `test_recovers_known_positive_effect` |
| False-positive rate under H₀ is ≤ 10% over 200 simulations (nominal 5%) | `test_false_positive_rate_is_calibrated` |
| Power ≥ 90% for a +20 lift over 200 simulations | `test_power_for_realistic_effect` |
| A negative effect never supports H₁ | `test_negative_effect_never_supports_h1` |
| A rare covariate is dropped instead of breaking the model | `test_rare_covariate_dropped_not_singular` |
| The stopping rule triggers when a group has fewer than 3 days | `test_stopping_rule_small_group` |
| Output files and JSON keys match this contract | `test_output_files_and_results_contract` |
| Two runs on the same input give byte-identical results | `test_pipeline_is_deterministic` |
| The real data cannot be analysed without `--confirm-prereg-locked` | `test_real_data_guard_blocks_without_flag` |
