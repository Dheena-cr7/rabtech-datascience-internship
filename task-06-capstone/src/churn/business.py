"""Retention-campaign profit model (analysis plan §6)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class CampaignAssumptions:
    contact_cost: float = 5.0          # $ per contacted customer
    incentive_months: float = 1.0      # free months offered (× monthly charge)
    success_rate: float = 0.30         # share of contacted would-be churners retained
    retained_months: float = 12.0      # months of revenue kept per retained churner

    def as_dict(self) -> dict:
        return asdict(self)


def customer_profit(y: np.ndarray, monthly: np.ndarray, a: CampaignAssumptions) -> np.ndarray:
    """Realised profit of contacting each customer, given their true churn label."""
    y = np.asarray(y, dtype=float)
    m = np.asarray(monthly, dtype=float)
    incentive = a.incentive_months * m
    gain_if_churner = a.success_rate * (a.retained_months * m - incentive)
    return y * gain_if_churner - (1 - y) * incentive - a.contact_cost


def expected_profit(p: np.ndarray, monthly: np.ndarray, a: CampaignAssumptions) -> np.ndarray:
    """Expected profit of contacting each customer, given a churn probability p."""
    m = np.asarray(monthly, dtype=float)
    incentive = a.incentive_months * m
    return (p * a.success_rate * (a.retained_months * m - incentive)
            - (1 - p) * incentive - a.contact_cost)


def campaign_profit(target: np.ndarray, y: np.ndarray, monthly: np.ndarray,
                    a: CampaignAssumptions) -> float:
    target = np.asarray(target, dtype=bool)
    return float(customer_profit(y, monthly, a)[target].sum())


def profit_curve(score: np.ndarray, y: np.ndarray, monthly: np.ndarray,
                 a: CampaignAssumptions) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative profit when contacting customers in descending score order."""
    order = np.argsort(-np.asarray(score))
    cum = np.concatenate([[0.0], np.cumsum(customer_profit(y, monthly, a)[order])])
    frac = np.arange(len(cum)) / (len(cum) - 1)
    return frac, cum


def best_threshold(p: np.ndarray, y: np.ndarray, monthly: np.ndarray,
                   a: CampaignAssumptions, grid: np.ndarray | None = None) -> tuple[float, float]:
    """Probability threshold that maximises realised profit (fit on out-of-fold data only)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    profits = [campaign_profit(p >= t, y, monthly, a) for t in grid]
    i = int(np.argmax(profits))
    return float(grid[i]), float(profits[i])
