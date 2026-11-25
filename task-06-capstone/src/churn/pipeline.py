"""End-to-end capstone pipeline: data -> tests -> models -> evaluation -> business -> metrics.json.

Run:  uv run python -m churn.pipeline
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (confusion_matrix, precision_recall_curve, roc_curve)
from sklearn.model_selection import cross_val_predict, train_test_split

from . import business as biz
from .data import CATEGORICAL, NUMERIC, RAW_PATH, TARGET, load_dataset, load_raw
from .modeling import (CV, SEED, bootstrap_auc_ci, candidates, classification_metrics,
                       paired_bootstrap_auc_diff, tune)
from .stats_tests import driver_tests

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports" / "figures"
METRICS = ROOT / "reports" / "metrics.json"
COLORS = {"Dummy (majority)": "#95A5A6", "Logistic Regression": "#3182BD",
          "Random Forest": "#E07B39", "Gradient Boosting": "#2A8C82"}
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


def _clean_name(s: str) -> str:
    return s.replace("num__", "").replace("cat__", "").replace("_", " = ", 1) if s.startswith("cat__") \
        else s.replace("num__", "")


def run(test_size: float = 0.20, n_boot: int = 2000) -> dict:
    t0 = time.time()
    raw = load_raw()
    X, y, full = load_dataset()
    monthly = full["MonthlyCharges"].to_numpy()
    out: dict = {"data": {
        "n_customers": int(len(full)), "n_raw_columns": int(raw.shape[1]),
        "n_features_model": len(NUMERIC) + len(CATEGORICAL),
        "churn_rate": float(y.mean()), "n_churned": int(y.sum()),
        "blank_total_charges_fixed": int((raw["TotalCharges"].astype(str).str.strip() == "").sum()),
        "monthly_revenue_total": float(full["MonthlyCharges"].sum()),
        "monthly_revenue_churned": float(full.loc[y == 1, "MonthlyCharges"].sum()),
    }}

    # ---------------------------------------------------------------- RQ2: univariate drivers
    drivers = driver_tests(full, TARGET, CATEGORICAL, NUMERIC)
    out["drivers"] = drivers.drop(columns=["df"]).to_dict("records")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, col in zip(axes, ["Contract", "InternetService", "PaymentMethod"]):
        rate = full.groupby(col)[TARGET].mean().sort_values()
        ax.barh(rate.index, rate.values * 100, color="#2A8C82")
        for i, v in enumerate(rate.values):
            ax.text(v * 100 + 1, i, f"{v:.0%}", va="center", fontsize=9)
        ax.axvline(y.mean() * 100, color="#C0392B", ls="--", lw=1)
        ax.set(title=f"Churn rate by {col}", xlabel="Churn rate (%)", xlim=(0, 55))
    _save(fig, "01_churn_by_category.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col, xl in [(axes[0], "tenure", "Tenure (months)"), (axes[1], "MonthlyCharges", "Monthly charge ($)")]:
        sns.kdeplot(data=full.assign(Status=np.where(y == 1, "Churned", "Stayed")), x=col,
                    hue="Status", fill=True, common_norm=False, ax=ax,
                    palette={"Churned": "#C0392B", "Stayed": "#2A8C82"})
        ax.set(xlabel=xl, title=f"{xl.split(' (')[0]} distribution by churn status")
    _save(fig, "02_numeric_distributions.png")

    # ---------------------------------------------------------------- split
    X_tr, X_te, y_tr, y_te, m_tr, m_te = train_test_split(
        X, y, monthly, test_size=test_size, stratify=y, random_state=SEED)
    out["split"] = {"n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                    "churn_rate_train": float(y_tr.mean()), "churn_rate_test": float(y_te.mean())}

    # ---------------------------------------------------------------- RQ1: models
    models, oof, test_p, tuning = {}, {}, {}, {}
    for name, (pipe, space, n_iter) in candidates().items():
        est, params, cv_mean, cv_sd = tune(name, pipe, space, n_iter, X_tr, y_tr)
        models[name] = est
        oof[name] = cross_val_predict(est, X_tr, y_tr, cv=CV, method="predict_proba")[:, 1]
        test_p[name] = est.predict_proba(X_te)[:, 1]
        tuning[name] = {"best_params": params, "cv_roc_auc_mean": cv_mean, "cv_roc_auc_sd": cv_sd}
    out["tuning"] = tuning

    selected = max((n for n in models if n != "Dummy (majority)"),
                   key=lambda n: tuning[n]["cv_roc_auc_mean"])
    out["selected_model"] = selected

    # threshold that maximises F1 on out-of-fold predictions (classification view)
    perf = {}
    for name in models:
        grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
        from sklearn.metrics import f1_score
        f1s = [f1_score(y_tr, (oof[name] >= t).astype(int), zero_division=0) for t in grid]
        thr = float(grid[int(np.argmax(f1s))])
        m = classification_metrics(y_te, test_p[name], thr)
        if name != "Dummy (majority)":
            m["roc_auc_ci95"] = list(bootstrap_auc_ci(y_te, test_p[name], n=n_boot))
        perf[name] = m
    out["test_performance"] = perf
    out["h1_selected_vs_logistic"] = paired_bootstrap_auc_diff(
        y_te, test_p[selected], test_p["Logistic Regression"], n=n_boot)

    # lift in top decile
    order = np.argsort(-test_p[selected])
    top = order[: len(order) // 10]
    out["top_decile"] = {"churn_rate": float(np.asarray(y_te)[top].mean()),
                         "lift": float(np.asarray(y_te)[top].mean() / y_te.mean())}

    # ROC + PR
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in models:
        if name == "Dummy (majority)":
            continue
        fpr, tpr, _ = roc_curve(y_te, test_p[name])
        axes[0].plot(fpr, tpr, color=COLORS[name], lw=2 if name == selected else 1.3,
                     label=f"{name} (AUC {perf[name]['roc_auc']:.3f})")
        pr, rc, _ = precision_recall_curve(y_te, test_p[name])
        axes[1].plot(rc, pr, color=COLORS[name], lw=2 if name == selected else 1.3,
                     label=f"{name} (AP {perf[name]['pr_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random (AUC 0.500)")
    axes[1].axhline(y_te.mean(), ls="--", color="grey", lw=1, label=f"Random (AP {y_te.mean():.3f})")
    axes[0].set(title="ROC curves (test set)", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision-recall curves (test set)", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=8, loc="lower right" if ax is axes[0] else "upper right")
    _save(fig, "03_roc_pr.png")

    # calibration + confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for name in models:
        if name == "Dummy (majority)":
            continue
        fp, mp = calibration_curve(y_te, test_p[name], n_bins=10, strategy="quantile")
        axes[0].plot(mp, fp, "o-", ms=4, color=COLORS[name], label=name)
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfectly calibrated")
    axes[0].set(title="Calibration (reliability) curves", xlabel="Mean predicted probability",
                ylabel="Observed churn rate"); axes[0].legend(fontsize=8)
    cm = confusion_matrix(y_te, (test_p[selected] >= perf[selected]["threshold"]).astype(int))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1],
                xticklabels=["Predicted stay", "Predicted churn"], yticklabels=["Stayed", "Churned"])
    axes[1].set(title=f"{selected}: confusion matrix (threshold {perf[selected]['threshold']:.2f})")
    _save(fig, "04_calibration_confusion.png")
    out["confusion_matrix"] = cm.tolist()

    # ---------------------------------------------------------------- RQ2: model-based importance
    pi = permutation_importance(models[selected], X_te, y_te, scoring="roc_auc", n_repeats=15,
                                random_state=SEED, n_jobs=-1)
    imp = (pd.DataFrame({"feature": X_te.columns, "mean": pi.importances_mean, "sd": pi.importances_std})
           .sort_values("mean", ascending=False).reset_index(drop=True))
    out["permutation_importance"] = imp.to_dict("records")
    top_imp = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_imp.feature, top_imp["mean"], xerr=top_imp["sd"], color="#2A8C82")
    ax.set(title=f"Permutation importance: {selected} (drop in test ROC-AUC)", xlabel="Mean AUC decrease")
    _save(fig, "05_permutation_importance.png")

    lr = models["Logistic Regression"]
    names = lr.named_steps["prep"].get_feature_names_out()
    coefs = pd.DataFrame({"feature": [_clean_name(n) for n in names],
                          "odds_ratio": np.exp(lr.named_steps["clf"].coef_[0])})
    coefs["abs_log"] = np.abs(np.log(coefs.odds_ratio))
    top_or = coefs.sort_values("abs_log", ascending=False).head(14).sort_values("odds_ratio")
    out["logistic_odds_ratios"] = top_or.drop(columns="abs_log").to_dict("records")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top_or.feature, top_or.odds_ratio,
            color=["#C0392B" if v > 1 else "#2A8C82" for v in top_or.odds_ratio])
    ax.axvline(1, color="k", lw=.8); ax.set_xscale("log")
    ax.set(title="Logistic regression odds ratios (top 14, log scale)",
           xlabel="Odds ratio (numeric features: per 1 SD)")
    _save(fig, "06_odds_ratios.png")

    # ---------------------------------------------------------------- RQ3: business value
    A = biz.CampaignAssumptions()
    out["assumptions"] = A.as_dict()
    thr_profit, oof_profit = biz.best_threshold(oof[selected], y_tr.to_numpy(), m_tr, A)
    y_te_np = np.asarray(y_te)
    rng = np.random.default_rng(SEED)
    target_model = test_p[selected] >= thr_profit
    k = int(target_model.sum())
    random_profits = [biz.campaign_profit(np.isin(np.arange(len(y_te_np)),
                                                  rng.choice(len(y_te_np), k, replace=False)),
                                          y_te_np, m_te, A) for _ in range(500)]
    thr_lr, _ = biz.best_threshold(oof["Logistic Regression"], y_tr.to_numpy(), m_tr, A)
    strategies = {
        "No campaign": 0.0,
        "Contact everyone": biz.campaign_profit(np.ones(len(y_te_np), bool), y_te_np, m_te, A),
        f"Random (same {k} customers)": float(np.mean(random_profits)),
        "Logistic Regression targeting": biz.campaign_profit(test_p["Logistic Regression"] >= thr_lr,
                                                             y_te_np, m_te, A),
        f"{selected} targeting": biz.campaign_profit(target_model, y_te_np, m_te, A),
        "Perfect foresight (upper bound)": biz.campaign_profit(y_te_np == 1, y_te_np, m_te, A),
    }
    tp = int((target_model & (y_te_np == 1)).sum())
    out["business"] = {
        "profit_threshold": thr_profit, "oof_profit_at_threshold": oof_profit,
        "n_targeted_test": k, "share_targeted_test": k / len(y_te_np),
        "churners_reached": tp, "churners_total_test": int(y_te_np.sum()),
        "random_profit_p95": float(np.percentile(random_profits, 95)),
        "strategies_test_profit": strategies,
        "scale_factor_to_full_base": len(full) / len(y_te_np),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for name in ["Logistic Regression", selected]:
        frac, cum = biz.profit_curve(test_p[name], y_te_np, m_te, A)
        ax.plot(frac * 100, cum, color=COLORS[name], lw=2, label=f"Target by {name}")
    frac_r, cum_r = biz.profit_curve(rng.random(len(y_te_np)), y_te_np, m_te, A)
    ax.plot(frac_r * 100, cum_r, color="#95A5A6", lw=1.5, ls="--", label="Random order")
    ax.axhline(0, color="k", lw=.8)
    ax.axvline(out["business"]["share_targeted_test"] * 100, color="#C0392B", ls=":",
               label=f"Chosen threshold ({thr_profit:.2f}) → {out['business']['share_targeted_test']:.0%} contacted")
    ax.set(title="Profit curve on the test set: cumulative campaign profit vs share of customers contacted",
           xlabel="Customers contacted (%, highest risk first)", ylabel="Campaign profit ($)")
    ax.legend(fontsize=9)
    _save(fig, "07_profit_curve.png")

    sens_rows = []
    for g in [0.10, 0.20, 0.30, 0.40, 0.50]:
        for months in [6, 12, 18, 24]:
            a = biz.CampaignAssumptions(success_rate=g, retained_months=months)
            t, _ = biz.best_threshold(oof[selected], y_tr.to_numpy(), m_tr, a)
            sens_rows.append({"success_rate": g, "retained_months": months, "threshold": t,
                              "model_profit": biz.campaign_profit(test_p[selected] >= t, y_te_np, m_te, a),
                              "contact_all_profit": biz.campaign_profit(np.ones(len(y_te_np), bool),
                                                                        y_te_np, m_te, a)})
    sens = pd.DataFrame(sens_rows)
    out["sensitivity"] = sens.to_dict("records")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.heatmap(sens.pivot(index="success_rate", columns="retained_months", values="model_profit"),
                annot=True, fmt=",.0f", cmap="RdYlGn", center=0, ax=ax, cbar_kws={"label": "test profit ($)"})
    ax.set(title="Sensitivity: model-targeted test profit ($)", xlabel="Months of revenue retained",
           ylabel="Retention success rate γ")
    ax.set_yticklabels([f"{float(t.get_text()):.0%}" for t in ax.get_yticklabels()], rotation=0)
    _save(fig, "08_sensitivity.png")

    out["runtime_seconds"] = round(time.time() - t0, 1)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    res = run()
    print(json.dumps({k: res[k] for k in ("selected_model", "test_performance", "h1_selected_vs_logistic",
                                          "business")}, indent=1, default=float))
