"""Synthetic fixtures with a KNOWN data-generating process.

The pipeline is checked against these instead of the real data, so the analysis
can be validated without seeing any real outcome values (data-blind check).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_series(
    n_days: int = 30,
    start: str = "2026-06-01",
    event_effect: float = 20.0,
    base: float = 100.0,
    trend: float = 0.5,
    weekend_effect: float = 8.0,
    holiday_effect: float = -10.0,
    noise_sd: float = 4.0,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    event = (rng.random(n_days) < 0.35).astype(int)
    holiday = (rng.random(n_days) < 0.08).astype(int)
    weekend = (dates.day_of_week >= 5).astype(int)
    t = np.arange(n_days)
    demand = (
        base
        + trend * t
        + weekend_effect * weekend
        + holiday_effect * holiday
        + event_effect * event
        + rng.normal(0, noise_sd, n_days)
    )
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "demand": np.round(demand).clip(min=0).astype(int),
            "marketing_event": event,
            "holiday": holiday,
        }
    )


def with_dirty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Append one row for each exclusion rule (prereg §4)."""
    dirty = pd.DataFrame(
        [
            {"date": "not-a-date", "demand": 100, "marketing_event": 0, "holiday": 0},
            {"date": "2026-07-15", "demand": 100, "marketing_event": 0, "holiday": 0},
            {"date": df["date"].iloc[0], "demand": 999, "marketing_event": 1, "holiday": 0},
            {"date": "2026-06-30", "demand": None, "marketing_event": 0, "holiday": 0},
            {"date": "2026-06-29", "demand": -5, "marketing_event": 2, "holiday": 0},
        ]
    )
    return pd.concat([df, dirty], ignore_index=True)
