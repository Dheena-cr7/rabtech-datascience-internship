# Analysis Preregistration: Marketing Events and Daily Demand

| Field | Value |
|---|---|
| Author | Dheena Dhayalan P |
| Programme | RabTech Academy, Data Science with Python & R Internship, Task 2 |
| Date registered | 2026-11-25 |
| Status | **LOCKED**. Frozen at git tag `prereg-v1` before any outcome values were analysed |
| Dataset | `data/raw/daily-demand-series.csv` (RabTech Academy starter file) |
| Raw file SHA-256 | `171eb7a471f10bfdb4674d34bfa78428c6118823de728245e24239409a053f89` |

> **Data-blind declaration.** When this plan was written, only the file's column names
> (`date, demand, marketing_event, holiday`) and its row count had been inspected. No
> summaries, plots or group comparisons of `demand` had been produced. The full pipeline
> was tested only on synthetic data (`tests/`). The real file is analysed only after the
> `prereg-v1` tag exists, and any later change to this plan is logged in §9.

---

## 1. Research question (PECO)

**Is daily demand higher on days with a marketing event than on days without one, after
adjusting for holidays, weekends and the underlying time trend?**

| Element | Definition |
|---|---|
| **Population** | Calendar days (the unit of analysis) in the daily demand series |
| **Exposure** | `marketing_event == 1`: a marketing event ran on that day |
| **Comparator** | `marketing_event == 0`: no marketing event that day |
| **Outcome** | `demand`: total units demanded that day (non-negative count) |
| **Observation window** | 2026-06-01 to 2026-06-30 inclusive (30 days), the full span of the file. Days outside this window are excluded |

## 2. Hypotheses

The test is one-sided because the hypothesis is directional and stated in advance.

- **H₀ (null):** β_event ≤ 0. Holding holiday, weekend and trend fixed, marketing-event days
  do not have higher expected demand.
- **H₁ (alternative):** β_event > 0. Marketing-event days have higher expected demand.

**Falsification rule:** H₁ is rejected for this dataset if the one-sided p-value for β_event
is ≥ 0.05. If the estimated β_event is ≤ 0, H₁ is rejected outright, whatever the p-value.

**Smallest effect of interest:** a 5% lift relative to the mean demand on non-event days.
A significant effect smaller than this is reported as "statistically detectable but
practically negligible".

## 3. Variables and transformations

| Variable | Source | Transformation |
|---|---|---|
| `demand` | raw | Used as-is (primary). `log(demand)` used only in robustness check R2 |
| `marketing_event` | raw | Binary 0/1, validated |
| `holiday` | raw | Binary 0/1, validated, used as a covariate |
| `weekend` | derived | 1 if `date` is a Saturday or Sunday, else 0 |
| `trend` | derived | Whole days since the first date in the window (0, 1, 2, …) |

No outliers are trimmed or winsorised in the primary analysis. Outliers are handled only
through robustness checks R1 and R2.

## 4. Exclusion rules (applied in this order, each logged with a count)

1. Rows with an unparseable or missing `date`.
2. Rows whose `date` falls outside the observation window.
3. Duplicate dates: keep the first occurrence and drop the rest.
4. Rows with missing `demand`, `marketing_event` or `holiday`.
5. Rows with `demand < 0`, or with `marketing_event`/`holiday` not in {0, 1}.

**Stopping rule:** if more than 20% of rows are excluded, or fewer than 3 days remain in
either the exposure or the comparator group, no inferential test is run. The results are
reported descriptively only, with status `insufficient_data`.

## 5. Statistical analysis plan

All tests use α = 0.05. Every estimate is reported with a 95% confidence interval.

### Primary analysis (confirmatory)

Ordinary least squares with heteroskedasticity-robust (HC3) standard errors:

```
demand ~ marketing_event + holiday + weekend + trend
```

The decision statistic is the one-sided p-value for β_event, taken from the two-sided HC3
p-value: p/2 if β > 0, otherwise 1 − p/2.

**Covariate rule:** a binary covariate (`holiday`, `weekend`) is dropped from the model if
either of its values occurs on fewer than 2 days. Otherwise the model cannot be estimated
(no variation) or the HC3 errors are undefined (leverage = 1). Dropped covariates are
recorded in `results.json` → `primary.dropped_covariates`. `marketing_event` is never
dropped; the §4 stopping rule covers it instead.

### Secondary analysis

- **Welch's two-sample t-test** comparing event and non-event days (one-sided,
  `alternative="greater"`), without covariates.
- **Effect size:** Hedges' g (Cohen's d with small-sample correction), plus the raw mean
  difference and the percentage lift over the non-event mean.

### Robustness checks (exploratory, will not override the primary decision)

| ID | Check | Purpose |
|---|---|---|
| R1 | Mann-Whitney U, one-sided | Distribution-free test that does not assume normality |
| R2 | Primary model on `log(demand)` | Multiplicative (percentage) effect; reduces outlier influence |
| R3 | Primary model with holidays excluded | Rules out holidays driving the result |

The result is called **robust** if R1, R2 and R3 all agree with the primary result on the
sign of the effect and on significance at α = 0.05. Any disagreement is reported in full.

### Model diagnostics (reported, not used to change the analysis)

Shapiro-Wilk test on the residuals of the primary model, and residual-vs-fitted plots.

## 6. Multiple comparisons

There is one confirmatory test (the primary analysis), so no correction is needed. The
secondary and robustness results are labelled exploratory and their p-values are not
adjusted.

## 7. Known limitations (stated in advance)

- **Small sample:** 30 days, with an unknown (not yet inspected) split between event and
  non-event days. The study may be underpowered to detect small lifts.
- **Observational design:** marketing events were not randomised, so the estimate is an
  adjusted association, not a proven causal effect.
- **Autocorrelation:** consecutive days are not independent. HC3 handles heteroskedasticity
  but not serial correlation. Durbin-Watson is reported for transparency.

## 8. Reproducibility

- Environment pinned in `uv.lock` (exact versions and hashes of every package).
- Raw data is read-only. Its SHA-256 hash (above) is checked before every run.
- Pipeline stages: `raw → interim (validated + exclusions) → processed (features) → results.json`.
- Output contracts are in `docs/output_contracts.md` and are enforced by the test suite on
  synthetic fixtures.
- The real-data run requires the explicit flag `--confirm-prereg-locked`.

## 9. Deviations log

| Date | Change | Reason |
|---|---|---|
| none so far | | |
