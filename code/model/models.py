#!/usr/bin/env python3
"""Model fitters and the single `fit_predict` path used everywhere.

Rule
----
Tuning, the cross-validation runs, the final retrain and the test scoring all fit
through `fit_predict`. There is no second way to build a model, so the settings
searched are the settings used: the learning rate, `bagging_freq`, thread count
and seed come from `lgbm_params` / `xgb_params` in every case.

Tree counts are counted per library, because the libraries disagree: LightGBM's
`best_iteration` is already the number of iterations of the best model, while
XGBoost's is a zero-based index. Getting this wrong shifts every tree count by
one, which is invisible except at the cap.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


_PARALLEL = {"trials": 1, "folds": 1}


def set_parallel(trial_workers: int = 1, fold_workers: int = 1) -> None:
    """How many fits run at once, so each gets a share of the cores.

    Two levels multiply: `trial_workers` trials in flight, each fitting
    `fold_workers` folds. Every fit then gets `cpus // (trials * folds)` threads.
    """
    _PARALLEL["trials"] = max(1, int(trial_workers))
    _PARALLEL["folds"] = max(1, int(fold_workers))


def set_workers(n: int) -> None:
    """Kept for callers that only know about trial-level parallelism."""
    set_parallel(n, _PARALLEL["folds"])


def model_key(model: str) -> str:
    """The config section that holds this model's settings."""
    return {"lgbm": "lightgbm", "xgb": "xgboost", "rf": "random_forest"}[model]


def detect_cpus() -> int:
    """CPUs this process may use, from the scheduler if it says so.

    SciNet runs under SLURM, where the allocation is smaller than the node; using
    `os.cpu_count()` there would oversubscribe and slow every fit down.
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_NTASKS"):
        v = os.environ.get(var, "").strip()
        if v.isdigit() and int(v) > 0:
            return int(v)
    return os.cpu_count() or 1


def resolve_threads(cfg: dict, key: str) -> int:
    """Threads for one fit: the config value (or every core if negative), capped by
    the allocation and divided by the number of concurrent fits."""
    want = int(cfg[key]["n_jobs"])
    cpus = detect_cpus()
    n = cpus if want < 0 else min(want, cpus)
    return max(1, n // (_PARALLEL["trials"] * _PARALLEL["folds"]))


def parallelism() -> dict:
    """What the thread shares were computed from, for the run metadata."""
    return {"trial_workers": _PARALLEL["trials"], "fold_workers": _PARALLEL["folds"]}


def lgbm_params(cfg: dict, tuned: dict | None = None) -> dict:
    p = {"objective": "l2", "verbose": -1,
         "learning_rate": float(cfg["lightgbm"]["learning_rate"]),
         "bagging_freq": int(cfg["lightgbm"]["bagging_freq"]),
         "num_threads": resolve_threads(cfg, "lightgbm"),
         "seed": int(cfg["seeds"]["tune"])}
    if tuned:
        p.update(tuned)
    return p


def xgb_params(cfg: dict, tuned: dict | None = None) -> dict:
    p = {"objective": "reg:squarederror",
         "tree_method": cfg["xgboost"]["tree_method"],
         "learning_rate": float(cfg["xgboost"]["learning_rate"]),
         "nthread": resolve_threads(cfg, "xgboost"),
         "seed": int(cfg["seeds"]["tune"])}
    if tuned:
        p.update(tuned)
    return p


def rf_params(cfg: dict, tuned: dict | None = None) -> dict:
    p = {"n_estimators": int(cfg["random_forest"]["n_estimators"]),
         "criterion": cfg["random_forest"]["criterion"],
         "n_jobs": resolve_threads(cfg, "random_forest"),
         "random_state": int(cfg["seeds"]["tune"])}
    if tuned:
        p.update(tuned)
    return p


def fit_lgbm(train, inner, cols, params, cap, patience=100):
    import lightgbm as lgb
    dtr = lgb.Dataset(train[cols], label=train["pm25"], free_raw_data=True)
    dva = lgb.Dataset(inner[cols], label=inner["pm25"], reference=dtr, free_raw_data=True)
    return lgb.train(params, dtr, num_boost_round=cap, valid_sets=[dva],
                     valid_names=["inner"],
                     callbacks=[lgb.early_stopping(patience, verbose=False)])


def best_iter_lgbm(booster) -> int:
    """LightGBM counts from 1: `best_iteration` is already the tree count.

    When early stopping never triggers the attribute can be absent or zero; fall
    back to the number of iterations actually built, because predicting with 0
    trees would silently return the initial score.
    """
    it = int(booster.best_iteration or 0)
    if it <= 0:
        it = int(booster.current_iteration())
    return it


def predict_lgbm(booster, df, cols) -> np.ndarray:
    return booster.predict(df[cols], num_iteration=best_iter_lgbm(booster))


def fit_xgb(train, inner, cols, params, cap, patience=100):
    import xgboost as xgb
    dtr = xgb.DMatrix(train[cols], label=train["pm25"])
    dva = xgb.DMatrix(inner[cols], label=inner["pm25"])
    return xgb.train(params, dtr, num_boost_round=cap, evals=[(dva, "inner")],
                     early_stopping_rounds=patience, verbose_eval=False)


def best_iter_xgb(booster) -> int:
    """XGBoost's best_iteration is a zero-based index; add one for the count."""
    it = getattr(booster, "best_iteration", None)
    if it is None:
        return int(booster.num_boosted_rounds())
    return int(it) + 1


def predict_xgb(booster, df, cols) -> np.ndarray:
    import xgboost as xgb
    return booster.predict(xgb.DMatrix(df[cols]), iteration_range=(0, best_iter_xgb(booster)))


def fit_rf(train, cols, params):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(**params)
    rf.fit(train[cols].to_numpy(), train["pm25"].to_numpy())
    return rf


def predict_rf(rf, df, cols) -> np.ndarray:
    return rf.predict(df[cols].to_numpy())


def fit_predict(model: str, cfg: dict, tuned: dict, cols: list[str], train, inner, score,
                cap: int, patience: int) -> tuple[np.ndarray, int, bool]:
    """Fit one model and predict the held-out rows.

    Returns (predictions, tree count, not_stopped).

    `not_stopped` is True when early stopping did not fire, which is detected as
    `n > cap - patience`: stopping can only have happened if the best iteration was
    more than `patience` rounds before the cap. Comparing `n >= cap` would miss the
    usual case, where LightGBM stops one or two rounds short of the cap and reports
    a best iteration just below it.

    Random Forest has no early stopping, so it is fitted on the inner cells as well.
    They were cut out only to make the inner split, and discarding them would train
    R4 on 15% less data than its own final model uses.
    """
    if model == "lgbm":
        b = fit_lgbm(train, inner, cols, lgbm_params(cfg, tuned), cap, patience)
        n = best_iter_lgbm(b)
        return predict_lgbm(b, score, cols), n, n > cap - patience
    if model == "xgb":
        b = fit_xgb(train, inner, cols, xgb_params(cfg, tuned), cap, patience)
        n = best_iter_xgb(b)
        return predict_xgb(b, score, cols), n, n > cap - patience
    if model == "rf":
        full = pd.concat([train, inner], ignore_index=True)
        rf = fit_rf(full, cols, rf_params(cfg, tuned))
        n = int(rf_params(cfg, tuned)["n_estimators"])
        return predict_rf(rf, score, cols), n, False
    raise ValueError(f"unknown model {model!r}")