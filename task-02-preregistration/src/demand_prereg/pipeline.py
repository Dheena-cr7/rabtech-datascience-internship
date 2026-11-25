"""Preregistered analysis pipeline: raw -> interim -> processed -> results.json.

Every rule here mirrors docs/preregistration.md. Changing behaviour here
without logging a deviation in §9 of that document breaks the preregistration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.stattools import durbin_watson

RAW_COLUMNS = ["date", "demand", "marketing_event", "holiday"]
PROCESSED_COLUMNS = ["date", "demand", "marketing_event", "holiday", "weekend", "trend"]
PRIMARY_COVARIATES = ["holiday", "weekend", "trend"]
ALPHA = 0.05
MAX_EXCLUDED_SHARE = 0.20
MIN_GROUP_SIZE = 3
SMALLEST_EFFECT_OF_INTEREST = 0.05  # 5% lift over the non-event mean
REAL_DATA_SHA256 = "171eb7a471f10bfdb4674d34bfa78428c6118823de728245e24239409a053f89"
WINDOW = ("2026-06-01", "2026-06-30")


class SchemaError(ValueError):
    """The raw file does not match the expected input contract."""


@dataclass(frozen=True)
class Window:
    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def of(cls, start: str, end: str) -> "Window":
        return cls(pd.Timestamp(start), pd.Timestamp(end))


# ---------------------------------------------------------------- stage 1: raw
def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")
    return df[RAW_COLUMNS].copy()


# ------------------------------------------------------- stage 2: interim
def apply_exclusions(raw: pd.DataFrame, window: Window) -> tuple[pd.DataFrame, dict]:
    """Apply prereg §4 exclusion rules in order; return kept rows + counts."""
    df = raw.copy()
    log: dict[str, int] = {"n_raw": int(len(df))}

    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="%Y-%m-%d")
    bad_date = df["date"].isna()
    log["excluded_bad_date"] = int(bad_date.sum())
    df = df[~bad_date]

    outside = (df["date"] < window.start) | (df["date"] > window.end)
    log["excluded_outside_window"] = int(outside.sum())
    df = df[~outside]

    dup = df["date"].duplicated(keep="first")
    log["excluded_duplicate_date"] = int(dup.sum())
    df = df[~dup]

    for col in ["demand", "marketing_event", "holiday"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    missing = df[["demand", "marketing_event", "holiday"]].isna().any(axis=1)
    log["excluded_missing_value"] = int(missing.sum())
    df = df[~missing]

    invalid = (
        (df["demand"] < 0)
        | ~df["marketing_event"].isin([0, 1])
        | ~df["holiday"].isin([0, 1])
    )
    log["excluded_invalid_value"] = int(invalid.sum())
    df = df[~invalid]

    df = df.astype({"marketing_event": int, "holiday": int}).sort_values("date")
    df = df.reset_index(drop=True)
    log["n_kept"] = int(len(df))
    log["excluded_share"] = (
        round(1 - log["n_kept"] / log["n_raw"], 6) if log["n_raw"] else 1.0
    )
    return df, log


# ----------------------------------------------------- stage 3: processed
def build_features(interim: pd.DataFrame, window: Window) -> pd.DataFrame:
    df = interim.copy()
    df["weekend"] = (df["date"].dt.day_of_week >= 5).astype(int)
    df["trend"] = (df["date"] - window.start).dt.days.astype(int)
    return df[PROCESSED_COLUMNS]


# ------------------------------------------------------ stage 4: analysis
def _one_sided(coef: float, p_two_sided: float) -> float:
    return p_two_sided / 2 if coef > 0 else 1 - p_two_sided / 2


MIN_COVARIATE_LEVEL = 2  # prereg §5: binary covariate kept only if each level has >= 2 days


def _formula(df: pd.DataFrame, covariates: list[str]) -> tuple[str, list[str]]:
    """Build the model formula, dropping binary covariates with too little variation."""
    kept, dropped = [], []
    for c in covariates:
        if c in ("holiday", "weekend") and df[c].value_counts().reindex([0, 1], fill_value=0).min() < MIN_COVARIATE_LEVEL:
            dropped.append(c)
        else:
            kept.append(c)
    return "demand ~ " + " + ".join(["marketing_event", *kept]), dropped


def _ols_event(df: pd.DataFrame, covariates: list[str]) -> dict:
    formula, dropped = _formula(df, covariates)
    fit = smf.ols(formula, data=df).fit(cov_type="HC3")
    coef = float(fit.params["marketing_event"])
    lo, hi = (float(x) for x in fit.conf_int(alpha=ALPHA).loc["marketing_event"])
    p_one = _one_sided(coef, float(fit.pvalues["marketing_event"]))
    return {
        "formula": formula,
        "dropped_covariates": dropped,
        "coef": coef,
        "ci95": [lo, hi],
        "p_one_sided": float(p_one),
        "significant": bool(coef > 0 and p_one < ALPHA),
        "n": int(fit.nobs),
        "_fit": fit,
    }


def hedges_g(x1: np.ndarray, x0: np.ndarray) -> float:
    n1, n0 = len(x1), len(x0)
    sp = np.sqrt(((n1 - 1) * x1.var(ddof=1) + (n0 - 1) * x0.var(ddof=1)) / (n1 + n0 - 2))
    d = (x1.mean() - x0.mean()) / sp if sp > 0 else 0.0
    return float(d * (1 - 3 / (4 * (n1 + n0) - 9)))


def run_analysis(processed: pd.DataFrame, exclusion_log: dict) -> dict:
    event = processed.loc[processed["marketing_event"] == 1, "demand"].to_numpy(float)
    control = processed.loc[processed["marketing_event"] == 0, "demand"].to_numpy(float)
    descriptive = {
        "n_event_days": int(len(event)),
        "n_control_days": int(len(control)),
        "mean_event": float(event.mean()) if len(event) else None,
        "mean_control": float(control.mean()) if len(control) else None,
    }

    if (
        exclusion_log["excluded_share"] > MAX_EXCLUDED_SHARE
        or len(event) < MIN_GROUP_SIZE
        or len(control) < MIN_GROUP_SIZE
    ):
        return {"status": "insufficient_data", "descriptive": descriptive}

    primary = _ols_event(processed, PRIMARY_COVARIATES)
    fit = primary.pop("_fit")
    resid = fit.resid.to_numpy()

    welch = stats.ttest_ind(event, control, equal_var=False, alternative="greater")
    mwu = stats.mannwhitneyu(event, control, alternative="greater")
    lift = (event.mean() - control.mean()) / control.mean() if control.mean() else None

    log_df = processed[processed["demand"] > 0].assign(demand=lambda d: np.log(d["demand"]))
    r2 = _ols_event(log_df, PRIMARY_COVARIATES)
    r2.pop("_fit")

    no_hol = processed[processed["holiday"] == 0]
    if (no_hol["marketing_event"] == 1).sum() >= MIN_GROUP_SIZE and (
        no_hol["marketing_event"] == 0
    ).sum() >= MIN_GROUP_SIZE:
        r3 = _ols_event(no_hol, ["weekend", "trend"])
        r3.pop("_fit")
    else:
        r3 = {"status": "insufficient_data"}

    checks = [
        mwu.pvalue < ALPHA,
        r2["significant"],
        r3.get("significant", False),
    ]
    robust = all(c == primary["significant"] for c in checks)

    return {
        "status": "ok",
        "descriptive": descriptive,
        "primary": primary,
        "decision": "reject_H0" if primary["significant"] else "fail_to_reject_H0",
        "secondary": {
            "welch_t": float(welch.statistic),
            "welch_p_one_sided": float(welch.pvalue),
            "mean_difference": float(event.mean() - control.mean()),
            "pct_lift": float(lift) if lift is not None else None,
            "hedges_g": hedges_g(event, control),
            "practically_meaningful": bool(
                lift is not None and lift >= SMALLEST_EFFECT_OF_INTEREST
            ),
        },
        "robustness": {
            "R1_mann_whitney": {"U": float(mwu.statistic), "p_one_sided": float(mwu.pvalue)},
            "R2_log_demand": r2,
            "R3_exclude_holidays": r3,
            "robust": bool(robust),
        },
        "diagnostics": {
            "shapiro_resid_p": float(stats.shapiro(resid).pvalue),
            "durbin_watson": float(durbin_watson(resid)),
        },
    }


# ------------------------------------------------------------- orchestrator
def run_pipeline(raw_path: Path, out_dir: Path, window: Window) -> dict:
    out_dir = Path(out_dir)
    for sub in ("interim", "processed"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    raw = load_raw(raw_path)
    interim, excl = apply_exclusions(raw, window)
    interim.to_csv(out_dir / "interim" / "demand_clean.csv", index=False, date_format="%Y-%m-%d")
    processed = build_features(interim, window)
    processed.to_csv(out_dir / "processed" / "demand_features.csv", index=False, date_format="%Y-%m-%d")

    results = {
        "prereg_version": "prereg-v1",
        "input_sha256": sha256_of(raw_path),
        "window": [str(window.start.date()), str(window.end.date())],
        "exclusions": excl,
        **run_analysis(processed, excl),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", default=WINDOW[0])
    ap.add_argument("--end", default=WINDOW[1])
    ap.add_argument(
        "--confirm-prereg-locked",
        action="store_true",
        help="Required to analyse the real dataset (only after the prereg-v1 tag exists).",
    )
    args = ap.parse_args(argv)

    digest = sha256_of(args.input)
    if digest == REAL_DATA_SHA256 and not args.confirm_prereg_locked:
        raise SystemExit(
            "Refusing to analyse the real dataset: pass --confirm-prereg-locked "
            "only after docs/preregistration.md is frozen at tag prereg-v1."
        )
    res = run_pipeline(args.input, args.out, Window.of(args.start, args.end))
    print(json.dumps({k: res[k] for k in ("status", "exclusions")}, indent=2))


if __name__ == "__main__":
    main()
