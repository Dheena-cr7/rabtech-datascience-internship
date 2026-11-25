"""Unit tests for data cleaning, statistics helpers and the profit model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn import business as biz
from churn.data import CATEGORICAL, NUMERIC, add_features, clean, load_dataset, load_raw
from churn.modeling import paired_bootstrap_auc_diff, preprocessor
from churn.stats_tests import cramers_v, holm


# ------------------------------------------------------------------ data
def test_raw_data_shape_and_ids():
    raw = load_raw()
    assert raw.shape == (7043, 21)
    assert raw["customerID"].is_unique


def test_blank_total_charges_only_for_new_customers():
    raw = load_raw()
    blank = raw["TotalCharges"].astype(str).str.strip() == ""
    assert blank.sum() == 11
    assert (raw.loc[blank, "tenure"] == 0).all()
    df = clean(raw)
    assert df["TotalCharges"].isna().sum() == 0
    assert (df.loc[blank.values, "TotalCharges"] == 0).all()


def test_clean_rejects_blank_total_for_tenured_customer():
    raw = load_raw().head(5).copy()
    raw.loc[0, "TotalCharges"] = " "
    raw.loc[0, "tenure"] = 12
    with pytest.raises(ValueError, match="tenure > 0"):
        clean(raw)


def test_engineered_features():
    df = add_features(clean(load_raw()))
    assert df["n_addon_services"].between(0, 6).all()
    assert set(df["auto_pay"]) == {"Yes", "No"}
    assert np.isfinite(df["charge_ratio"]).all()


def test_dataset_has_no_target_or_id_leakage():
    X, y, _ = load_dataset()
    assert list(X.columns) == NUMERIC + CATEGORICAL
    assert "Churn" not in X.columns and "customerID" not in X.columns
    assert set(y.unique()) == {0, 1}


def test_preprocessor_outputs_numeric_matrix():
    X, _, _ = load_dataset()
    Z = preprocessor().fit_transform(X.head(200))
    assert Z.shape[0] == 200 and np.isfinite(Z).all()


# ------------------------------------------------------------------ statistics
def test_holm_matches_known_values():
    adj = holm(np.array([0.01, 0.04, 0.03]))
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])


def test_cramers_v_bounds():
    perfect = pd.DataFrame([[50, 0], [0, 50]])
    none = pd.DataFrame([[25, 25], [25, 25]])
    assert cramers_v(perfect) == pytest.approx(1.0)
    assert cramers_v(none) == pytest.approx(0.0)


def test_paired_bootstrap_detects_better_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    good = y + rng.normal(0, 0.6, 600)
    bad = y + rng.normal(0, 2.0, 600)
    res = paired_bootstrap_auc_diff(y, good, bad, n=300)
    assert res["diff"] > 0 and res["ci95"][0] > 0 and res["p_one_sided"] < 0.05


# ------------------------------------------------------------------ profit model
A = biz.CampaignAssumptions(contact_cost=5, incentive_months=1, success_rate=0.3, retained_months=12)


def test_customer_profit_by_hand():
    # churner paying $100: 0.3 * (1200 - 100) - 5 = 325 ; non-churner: -100 - 5 = -105
    np.testing.assert_allclose(biz.customer_profit([1, 0], [100, 100], A), [325.0, -105.0])


def test_expected_profit_equals_realised_at_certainty():
    m = np.array([50.0, 80.0])
    np.testing.assert_allclose(biz.expected_profit(np.array([1.0, 0.0]), m, A),
                               biz.customer_profit([1, 0], m, A))


def test_best_threshold_targets_only_churners_with_perfect_scores():
    y = np.array([1, 1, 0, 0, 0])
    m = np.full(5, 70.0)
    p = y.astype(float)
    t, profit = biz.best_threshold(p, y, m, A)
    assert profit == pytest.approx(biz.campaign_profit(y == 1, y, m, A))


def test_profit_curve_starts_at_zero_and_ends_at_contact_all():
    y = np.array([1, 0, 1, 0])
    m = np.array([60.0, 40.0, 90.0, 30.0])
    frac, cum = biz.profit_curve(np.array([.9, .1, .8, .2]), y, m, A)
    assert frac[0] == 0 and cum[0] == 0 and frac[-1] == 1
    assert cum[-1] == pytest.approx(biz.campaign_profit(np.ones(4, bool), y, m, A))
