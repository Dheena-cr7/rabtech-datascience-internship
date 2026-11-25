# Task 6: Data Science Research Capstone

## Predicting Customer Churn for Profit
**RabTech Academy · Data Science with Python & R Internship** · Dheena Dhayalan P

**Research report (PDF, 11 pages):** [`reports/capstone_report.pdf`](reports/capstone_report.pdf)
**Walkthrough notebook:** [`notebooks/capstone_walkthrough.ipynb`](notebooks/capstone_walkthrough.ipynb)
**Analysis plan (committed before modelling):** [`docs/analysis_plan.md`](docs/analysis_plan.md)

### Research questions
1. **Prediction:** how accurately can telecom churn be predicted from account, service and billing data?
2. **Explanation:** which attributes drive churn?
3. **Decision:** is a model-targeted retention campaign more profitable than no campaign, contacting everyone, or random targeting?

### Headline results (IBM Telco Customer Churn, 7,043 customers, 20% held-out test set)

| | Result |
|---|---|
| Best model (by CV) | Random Forest: **test ROC-AUC 0.845** [95% CI 0.822–0.866], PR-AUC 0.660, top-decile lift 2.8× |
| vs logistic regression | +0.004 AUC, **not significant** (bootstrap p = 0.13): H1 not supported |
| Top churn drivers | Contract type, tenure, internet service (fiber), no online security or tech support: H2 supported |
| Campaign profit (test set) | **Model targeting $50,940** vs contact-everyone $19,678 vs random $8,042 vs none $0: H3 supported |
| Robustness | Profitable in **20/20** sensitivity scenarios. Contacting everyone loses money in 9/20 |

### Repository layout
```
task-06-capstone/
├── data/raw/telco-customer-churn.csv   IBM sample dataset (SHA-256 in docs/analysis_plan.md)
├── docs/analysis_plan.md               research questions, hypotheses, methods (written first)
├── src/churn/
│   ├── data.py          load, validate, clean, engineer features
│   ├── stats_tests.py   chi-square / Mann-Whitney driver tests + Holm correction
│   ├── modeling.py      pipelines, randomised-search tuning, bootstrap CIs
│   ├── business.py      retention-campaign profit model
│   ├── pipeline.py      end-to-end run -> reports/metrics.json + figures
│   └── report.py        builds the PDF from metrics.json (no hand-typed numbers)
├── tests/test_capstone.py              13 unit tests
├── notebooks/capstone_walkthrough.ipynb
└── reports/  capstone_report.pdf · metrics.json · figures/
```

### Reproduce
```bash
uv sync --locked
uv run pytest                      # 13 tests
uv run python -m churn.pipeline    # about 10 min: tests, models, figures, metrics.json
uv run python -m churn.report      # rebuilds reports/capstone_report.pdf
```
