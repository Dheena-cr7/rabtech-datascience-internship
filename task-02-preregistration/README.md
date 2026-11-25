# Task 2: Reproducible Research Question & Analysis Preregistration

**RabTech Academy · Data Science with Python & R Internship**

> **Research question:** Is daily demand higher on marketing-event days than on non-event
> days, after adjusting for holidays, weekends and time trend? (June 2026, daily data)

The analysis plan was written and locked **before** any outcome values were examined.
The pipeline was tested only on synthetic data with a known answer.

## Deliverables

| Requirement | Where |
|---|---|
| Population, exposure, comparator, outcome, window | [`docs/preregistration.md` §1](docs/preregistration.md) |
| Hypotheses, exclusions, transformations, tests, effect sizes, robustness checks | [`docs/preregistration.md` §2–6](docs/preregistration.md) |
| Environment lockfile | [`uv.lock`](uv.lock) (+ [`pyproject.toml`](pyproject.toml)) |
| raw / interim / processed / notebooks / reports structure | see layout below |
| Data-blind pipeline check on synthetic fixtures | [`tests/`](tests/) |
| Expected output contracts | [`docs/output_contracts.md`](docs/output_contracts.md) |

## Layout

```
task-02-preregistration/
├── data/
│   ├── raw/            daily-demand-series.csv (read-only, SHA-256 verified)
│   ├── interim/        validated rows after exclusions        (generated)
│   └── processed/      model-ready features                   (generated)
├── docs/
│   ├── preregistration.md
│   └── output_contracts.md
├── notebooks/          analysis notebooks (after the prereg is locked)
├── reports/            results.json + figures                 (generated)
├── src/demand_prereg/pipeline.py
├── tests/
│   ├── fixtures/synthetic.py   known data-generating process
│   └── test_pipeline.py        13 data-blind checks
├── pyproject.toml
└── uv.lock
```

## Reproduce

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked          # exact environment from uv.lock
uv run pytest             # data-blind checks on synthetic fixtures (13 tests)
```

Real-data run (only after the `prereg-v1` tag exists):

```bash
uv run python -m demand_prereg.pipeline \
    --input data/raw/daily-demand-series.csv --out . --confirm-prereg-locked
```

Without `--confirm-prereg-locked`, the pipeline refuses to read the real file.

## Data-blind check result

```
13 passed
```

The simulations in the test suite (200 synthetic datasets each) show the method rejects
H₀ about **8%** of the time when there is no true effect (nominal 5%, with 30 days of
data), and detects a +20-unit lift **99.5%** of the time.
