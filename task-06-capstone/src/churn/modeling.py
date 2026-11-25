"""Model definitions, tuning and evaluation (analysis plan §4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC

SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
         CATEGORICAL),
    ])


def candidates() -> dict[str, tuple[Pipeline, dict, int]]:
    """name -> (pipeline, search space, n_iter)."""
    return {
        "Dummy (majority)": (
            Pipeline([("prep", preprocessor()), ("clf", DummyClassifier(strategy="prior"))]), {}, 1),
        "Logistic Regression": (
            Pipeline([("prep", preprocessor()),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))]),
            {"clf__C": loguniform(1e-3, 1e2)}, 20),
        "Random Forest": (
            Pipeline([("prep", preprocessor()),
                      ("clf", RandomForestClassifier(class_weight="balanced_subsample",
                                                     random_state=SEED, n_jobs=-1))]),
            {"clf__n_estimators": randint(200, 600), "clf__max_depth": randint(4, 16),
             "clf__min_samples_leaf": randint(1, 30), "clf__max_features": uniform(0.2, 0.6)}, 25),
        "Gradient Boosting": (
            Pipeline([("prep", preprocessor()),
                      ("clf", HistGradientBoostingClassifier(class_weight="balanced",
                                                             random_state=SEED))]),
            {"clf__learning_rate": loguniform(0.01, 0.3), "clf__max_iter": randint(100, 500),
             "clf__max_leaf_nodes": randint(8, 48), "clf__min_samples_leaf": randint(10, 80),
             "clf__l2_regularization": loguniform(1e-4, 10)}, 30),
    }


def tune(name: str, pipe: Pipeline, space: dict, n_iter: int, X: pd.DataFrame, y: pd.Series):
    if not space:
        pipe.fit(X, y)
        from sklearn.model_selection import cross_val_score
        cv_auc = cross_val_score(pipe, X, y, cv=CV, scoring="roc_auc")
        return pipe, {}, float(cv_auc.mean()), float(cv_auc.std())
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="roc_auc", cv=CV,
                                random_state=SEED, n_jobs=-1, refit=True)
    search.fit(X, y)
    i = search.best_index_
    params = {k.replace("clf__", ""): (float(v) if isinstance(v, (float, np.floating)) else int(v))
              for k, v in search.best_params_.items()}
    return (search.best_estimator_, params, float(search.cv_results_["mean_test_score"][i]),
            float(search.cv_results_["std_test_score"][i]))


def classification_metrics(y, p, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "threshold": threshold,
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "flagged_share": float(pred.mean())}


def bootstrap_auc_ci(y, p, n: int = 2000, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap_auc_diff(y, p_a, p_b, n: int = 2000, seed: int = SEED) -> dict:
    """AUC(a) - AUC(b) with a 95% CI and one-sided p (H1: a > b)."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    diffs = np.array(diffs)
    return {"diff": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_one_sided": float((diffs <= 0).mean())}
