"""Univariate churn-driver tests (analysis plan §5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


def holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def driver_tests(df: pd.DataFrame, target: str, categorical: list[str],
                 numeric: list[str]) -> pd.DataFrame:
    rows = []
    for col in categorical:
        table = pd.crosstab(df[col], df[target])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        rates = df.groupby(col)[target].mean()
        rows.append({"feature": col, "type": "categorical", "test": "chi-square",
                     "statistic": chi2, "df": dof, "p": p, "effect": cramers_v(table),
                     "effect_name": "Cramér's V",
                     "detail": f"highest churn: {rates.idxmax()} ({rates.max():.0%}); "
                               f"lowest: {rates.idxmin()} ({rates.min():.0%})"})
    for col in numeric:
        a = df.loc[df[target] == 1, col]
        b = df.loc[df[target] == 0, col]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rbc = 1 - 2 * u / (len(a) * len(b))       # rank-biserial (positive => churners lower)
        rows.append({"feature": col, "type": "numeric", "test": "Mann-Whitney U",
                     "statistic": u, "df": np.nan, "p": p, "effect": -rbc,
                     "effect_name": "rank-biserial r",
                     "detail": f"median churned {a.median():.1f} vs stayed {b.median():.1f}"})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy())
    out["significant"] = out["p_holm"] < ALPHA
    return out.sort_values("effect", key=np.abs, ascending=False).reset_index(drop=True)
