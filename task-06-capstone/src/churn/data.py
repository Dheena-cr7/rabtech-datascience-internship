"""Loading, validation, cleaning and feature engineering (analysis plan §3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "telco-customer-churn.csv"
TARGET = "Churn"

ADDON_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                  "StreamingTV", "StreamingMovies"]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "n_addon_services", "charge_ratio"]
CATEGORICAL = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
               "InternetService", *ADDON_SERVICES, "Contract", "PaperlessBilling", "PaymentMethod",
               "auto_pay"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"customerID", TARGET, "tenure", "MonthlyCharges", "TotalCharges", "Contract"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"raw file is missing columns: {sorted(missing)}")
    if df["customerID"].duplicated().any():
        raise ValueError("duplicate customerID values")
    return df


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop(columns="customerID").copy()
    total = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", np.nan),
                          errors="coerce")
    blank = total.isna()
    if not (df.loc[blank, "tenure"] == 0).all():
        raise ValueError("TotalCharges is blank for a customer with tenure > 0")
    df["TotalCharges"] = total.fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["n_addon_services"] = (out[ADDON_SERVICES] == "Yes").sum(axis=1)
    out["auto_pay"] = np.where(out["PaymentMethod"].str.contains("automatic"), "Yes", "No")
    avg_monthly = np.where(out["tenure"] > 0, out["TotalCharges"] / out["tenure"].clip(lower=1),
                           out["MonthlyCharges"])
    out["charge_ratio"] = avg_monthly / out["MonthlyCharges"]
    return out


def load_dataset(path: Path = RAW_PATH) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, full frame incl. MonthlyCharges for the profit model)."""
    df = add_features(clean(load_raw(path)))
    return df[NUMERIC + CATEGORICAL], df[TARGET], df
