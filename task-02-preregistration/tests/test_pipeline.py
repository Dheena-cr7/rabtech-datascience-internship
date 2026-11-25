"""Data-blind pipeline checks: every test runs on synthetic fixtures only."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from demand_prereg.pipeline import (
    PROCESSED_COLUMNS,
    SchemaError,
    Window,
    apply_exclusions,
    build_features,
    load_raw,
    main,
    run_analysis,
    run_pipeline,
)
from fixtures.synthetic import make_series, with_dirty_rows

WIN = Window.of("2026-06-01", "2026-06-30")


@pytest.fixture
def csv(tmp_path):
    def _write(df, name="raw.csv"):
        p = tmp_path / name
        df.to_csv(p, index=False)
        return p

    return _write


# ------------------------------------------------------------ input contract
def test_missing_column_raises(csv):
    with pytest.raises(SchemaError):
        load_raw(csv(make_series().drop(columns="holiday")))


def test_exclusion_order_duplicates_caught_first(csv):
    # Full 30-day series: the 06-01, 06-29 and 06-30 dirty rows are all duplicates,
    # so rule 3 removes them before rules 4/5 ever see them (prereg §4 order).
    raw = load_raw(csv(with_dirty_rows(make_series())))
    kept, log = apply_exclusions(raw, WIN)
    assert log["excluded_bad_date"] == 1
    assert log["excluded_outside_window"] == 1
    assert log["excluded_duplicate_date"] == 3
    assert log["excluded_missing_value"] == 0
    assert log["excluded_invalid_value"] == 0
    assert log["n_kept"] == 30
    assert kept["date"].is_unique


def test_missing_and_invalid_values_excluded(csv):
    df = make_series(n_days=28)  # leaves 06-29 and 06-30 free
    raw = load_raw(csv(with_dirty_rows(df)))
    _, log = apply_exclusions(raw, WIN)
    assert log["excluded_missing_value"] == 1
    assert log["excluded_invalid_value"] == 1
    assert log["n_kept"] == 28


# ---------------------------------------------------------------- features
def test_features_contract(csv):
    kept, _ = apply_exclusions(load_raw(csv(make_series())), WIN)
    feat = build_features(kept, WIN)
    assert list(feat.columns) == PROCESSED_COLUMNS
    assert feat["trend"].tolist() == list(range(30))
    # 2026-06-06 is a Saturday, 2026-06-08 a Monday
    by_date = feat.set_index(feat["date"].dt.strftime("%Y-%m-%d"))
    assert by_date.loc["2026-06-06", "weekend"] == 1
    assert by_date.loc["2026-06-08", "weekend"] == 0


# ------------------------------------------------- analysis recovers truth
def test_recovers_known_positive_effect(csv, tmp_path):
    res = run_pipeline(csv(make_series(event_effect=20)), tmp_path / "o", WIN)
    assert res["status"] == "ok"
    lo, hi = res["primary"]["ci95"]
    assert lo <= 20 <= hi, "95% CI must cover the true effect"
    assert res["decision"] == "reject_H0"


def _rejection_rate(event_effect: float, n_sims: int = 200) -> float:
    rejections = runs = 0
    for seed in range(n_sims):
        raw = make_series(event_effect=event_effect, seed=1000 + seed)
        kept, log = apply_exclusions(raw, WIN)
        res = run_analysis(build_features(kept, WIN), log)
        if res["status"] == "ok":
            runs += 1
            rejections += res["decision"] == "reject_H0"
    return rejections / runs


def test_false_positive_rate_is_calibrated():
    # Under H0 (true effect = 0) the test should reject about 5% of the time.
    assert _rejection_rate(0.0) <= 0.10


def test_power_for_realistic_effect():
    # A 20-unit lift (~18%) should be detected in most samples of 30 days.
    assert _rejection_rate(20.0) >= 0.90


def test_negative_effect_never_supports_h1(csv, tmp_path):
    res = run_pipeline(csv(make_series(event_effect=-20)), tmp_path / "o", WIN)
    assert res["primary"]["coef"] < 0
    assert res["decision"] == "fail_to_reject_H0"


def test_rare_covariate_dropped_not_singular(csv, tmp_path):
    df = make_series()
    df["holiday"] = 0
    df.loc[3, "holiday"] = 1  # a single holiday day: leverage 1, HC3 would break
    res = run_pipeline(csv(df), tmp_path / "o", WIN)
    assert res["primary"]["dropped_covariates"] == ["holiday"]
    assert "holiday" not in res["primary"]["formula"]
    lo, hi = res["primary"]["ci95"]
    assert lo <= 20 <= hi


def test_stopping_rule_small_group(csv, tmp_path):
    df = make_series()
    df["marketing_event"] = 0
    df.loc[:1, "marketing_event"] = 1  # only 2 event days < MIN_GROUP_SIZE
    res = run_pipeline(csv(df), tmp_path / "o", WIN)
    assert res["status"] == "insufficient_data"
    assert "primary" not in res


# ---------------------------------------------------------- output contract
def test_output_files_and_results_contract(csv, tmp_path):
    out = tmp_path / "o"
    run_pipeline(csv(make_series()), out, WIN)
    assert (out / "interim" / "demand_clean.csv").exists()
    proc = pd.read_csv(out / "processed" / "demand_features.csv")
    assert list(proc.columns) == PROCESSED_COLUMNS

    res = json.loads((out / "results.json").read_text())
    for key in ["prereg_version", "input_sha256", "window", "exclusions", "status",
                "descriptive", "primary", "decision", "secondary", "robustness",
                "diagnostics"]:
        assert key in res, key
    assert set(res["primary"]) == {"formula", "dropped_covariates", "coef", "ci95",
                                   "p_one_sided", "significant", "n"}
    assert 0 <= res["primary"]["p_one_sided"] <= 1
    assert res["decision"] in {"reject_H0", "fail_to_reject_H0"}
    assert set(res["robustness"]) == {"R1_mann_whitney", "R2_log_demand",
                                      "R3_exclude_holidays", "robust"}


def test_pipeline_is_deterministic(csv, tmp_path):
    p = csv(make_series())
    run_pipeline(p, tmp_path / "a", WIN)
    run_pipeline(p, tmp_path / "b", WIN)
    assert (tmp_path / "a/results.json").read_text() == (tmp_path / "b/results.json").read_text()


# --------------------------------------------------------- data-blind guard
def test_real_data_guard_blocks_without_flag(tmp_path):
    from pathlib import Path

    real = Path(__file__).resolve().parents[1] / "data" / "raw" / "daily-demand-series.csv"
    with pytest.raises(SystemExit, match="Refusing"):
        main(["--input", str(real), "--out", str(tmp_path / "o")])
    assert not (tmp_path / "o").exists()
