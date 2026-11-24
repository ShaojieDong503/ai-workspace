#!/usr/bin/env python3
"""Tuning on the grouped 10-fold scheme (plan step 3).

Rule
----
Folds are the parallel unit, not trials. The data is small (~145k rows, 62
columns), so one fit gains little beyond about 16 threads; a 192-core node has to
run several fits at once. A trial's folds therefore run concurrently, each with
`cpus / (trial_workers * fold_workers)` threads, while trials stay sequential so
TPE learns from finished results.

Consequences, all deliberate:

* results are collected by fold index, never completion order, so RMSE lists, tree
  counts and files do not depend on thread timing;
* with `fold_workers > 1` a fold cannot be pruned before the others finish, so the
  median pruner is dropped; it is kept only for `fold_workers == 1`;
* `--workers > 1` runs trials concurrently, which uses `constant_liar` so the
  sampler knows about running trials. Values above 8 degrade the search, because
  TPE then proposes most trials before any result is back.

Every fit goes through `models.fit_predict`, the same path the runs use.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time

import numpy as np
import pandas as pd

import cv
import data as D
import features as F
import folds as FLD
import models

OUT = D.OUT
OPTUNA = OUT / "optuna"


def evaluate(model: str, tuned: dict, dataset: str, cfg: dict, d, splits, quick: bool,
             trial=None, fold_workers: int = 1):
    """Mean and per-fold RMSE, tree counts and not-stopped counts, on group10."""
    cols = F.columns(dataset)
    cap = cv.round_cap(cfg, quick)
    patience = int(cfg["early_stopping"]["patience"])
    specs = FLD.plan("group10", splits, pd.read_parquet(OUT / "folds.parquet"))

    def one(spec):
        train, inner, score = cv.fold_rows(d, splits, cfg, spec, quick)
        yhat, n, not_stopped = models.fit_predict(model, cfg, tuned, cols, train, inner,
                                                  score, cap, patience)
        err = score["pm25"].to_numpy(float) - np.asarray(yhat, float)
        return float(np.sqrt(np.mean(err ** 2))), int(n), int(not_stopped)

    if fold_workers > 1 or trial is None:
        # parallel folds: a fold cannot be judged before its siblings finish
        res = cv.fit_folds(specs, one, fold_workers)
    else:
        res = []
        for i, spec in enumerate(specs):
            res.append(one(spec))
            trial.report(float(np.mean([r[0] for r in res])), step=i)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    rmses = [r[0] for r in res]
    return float(np.mean(rmses)), rmses, [r[1] for r in res], sum(r[2] for r in res)


def sample(trial, model: str, cfg: dict) -> dict:
    if model == "lgbm":
        g = cfg["lightgbm"]
        return {"learning_rate": trial.suggest_float("learning_rate", *g["learning_rate_range"],
                                                     log=True),
                "num_leaves": trial.suggest_int("num_leaves", *g["num_leaves"]),
                "max_depth": trial.suggest_int("max_depth", *g["max_depth"]),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", *g["min_data_in_leaf"]),
                "feature_fraction": trial.suggest_float("feature_fraction", *g["feature_fraction"]),
                "bagging_fraction": trial.suggest_float("bagging_fraction", *g["bagging_fraction"]),
                "lambda_l1": trial.suggest_float("lambda_l1", *g["lambda_l1"]),
                "lambda_l2": trial.suggest_float("lambda_l2", *g["lambda_l2"]),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", *g["min_gain_to_split"])}
    g = cfg["xgboost"]
    return {"learning_rate": trial.suggest_float("learning_rate", *g["learning_rate_range"],
                                                 log=True),
            "max_depth": trial.suggest_int("max_depth", *g["max_depth"]),
            "min_child_weight": trial.suggest_float("min_child_weight", *g["min_child_weight"]),
            "subsample": trial.suggest_float("subsample", *g["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *g["colsample_bytree"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *g["reg_lambda"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *g["reg_alpha"]),
            "gamma": trial.suggest_float("gamma", *g["gamma"])}


def feature_fraction(value, n_features: int) -> float:
    """`max_features` as a fraction of the columns, so values can be compared."""
    if isinstance(value, str) and value == "sqrt":
        return 1.0 / math.sqrt(n_features)
    return float(value)


def simplest(trial, model: str, n_features: int):
    p = trial.params
    if model == "lgbm":
        return (p.get("num_leaves", 1e9), -p.get("min_data_in_leaf", 0))
    return (p.get("max_depth", 1e9), -p.get("min_child_weight", 0))


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune one model")
    ap.add_argument("--model", required=True, choices=["lgbm", "xgb", "rf"])
    ap.add_argument("--dataset", default="A", choices=sorted(F.DATASETS))
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent trials; above 8 degrades the search")
    ap.add_argument("--fold-workers", type=int, default=None,
                    help="folds fitted at once; defaults to one per fold, capped by cpus")
    args = ap.parse_args()

    cfg = D.load_config()
    d = D.load()
    splits = pd.read_parquet(OUT / "splits.parquet")
    n_features = len(F.columns(args.dataset))
    n_folds = len(FLD.plan("group10", splits, pd.read_parquet(OUT / "folds.parquet")))
    fold_workers = args.fold_workers or cv.default_fold_workers(n_folds)
    # Random Forest has no per-fold early stopping to parallelise against a trial
    # budget, and its fits already use every thread, so trials stay serial
    trial_workers = 1 if args.model == "rf" else max(1, args.workers)
    models.set_parallel(trial_workers, fold_workers)

    out_dir = OUT / "quick" if args.quick else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.model == "rf":
        result = tune_grid(cfg, d, splits, args.dataset, n_features, args.quick, fold_workers)
    else:
        result = tune_optuna(cfg, d, splits, args.model, args.dataset, n_features,
                             args.quick, args.trials, trial_workers, fold_workers)

    result.update({"model": args.model, "dataset": args.dataset, "quick": args.quick,
                   "wall_seconds": round(time.time() - t0, 1),
                   "fold_workers": int(fold_workers),
                   "trial_workers": int(trial_workers),
                   "threads_per_fit": models.resolve_threads(
                       cfg, models.model_key(args.model)),
                   "config_hash": D.config_hash(cfg)})
    name = out_dir / f"params_{args.model}.json"
    name.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{args.model}: chosen {result['chosen']['value']:.4f}  "
          f"best {result['best']['value']:.4f}  trees {result['n_trees_final']}  "
          f"folds x{fold_workers}, trials x{trial_workers}, "
          f"{result['threads_per_fit']} threads/fit -> {name}")
    return 0


def tune_optuna(cfg, d, splits, model, dataset, n_features, quick, trials,
                trial_workers=1, fold_workers=1):
    import optuna
    from optuna.samplers import TPESampler

    n_trials = trials or (int(cfg["quick"]["trials"]) if quick
                          else int(cfg["tuning"]["lgbm_trials" if model == "lgbm" else "xgb_trials"]))
    OPTUNA.mkdir(parents=True, exist_ok=True)
    study_name = (f"{model}_{dataset}_{'quick' if quick else 'full'}_"
                  f"{D.config_hash(cfg)}_{cv.code_hash()}")
    pruner = (optuna.pruners.MedianPruner(n_warmup_steps=3, n_startup_trials=5)
              if fold_workers == 1 else None)
    sampler = TPESampler(seed=int(cfg["seeds"]["tune"]),       # replaced below, once the
                         constant_liar=bool(trial_workers > 1))  # finished count is known
    # one SQLite file per model, so the LightGBM and XGBoost tuning jobs can run at the
    # same time without two processes writing one database
    study = optuna.create_study(
        study_name=study_name, direction="minimize", sampler=sampler, pruner=pruner,
        storage=f"sqlite:///{(OPTUNA / f'studies_{model}.db').as_posix()}", load_if_exists=True)

    def objective(trial):
        p = sample(trial, model, cfg)
        mean, per_fold, iters, hits = evaluate(model, p, dataset, cfg, d, splits, quick,
                                               trial=trial, fold_workers=fold_workers)
        trial.set_user_attr("per_fold", per_fold)
        trial.set_user_attr("best_iters", iters)
        trial.set_user_attr("not_stopped", hits)
        return mean

    # a trial still RUNNING here was cut off by a wall time or a crash in an earlier job:
    # only one job per model ever runs, so nothing else can own it. Marked FAIL so the
    # database says what happened; it was never counted or chosen either way.
    stale = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    for t in stale:
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)
    if stale:
        print(f"marked {len(stale)} interrupted trial(s) as FAIL: {[t.number for t in stale]}")

    # a pruned trial used a slot, so it counts towards the requested total; otherwise
    # a resubmitted job with pruning on would run more trials than asked for
    finished = sum(1 for t in study.trials
                   if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED))
    remaining = max(0, n_trials - finished)
    # every process starts a fresh sampler. With the same seed each time, a job that
    # resumes inside the random start-up trials would re-propose the trials that already
    # ran. Offsetting by the finished count keeps the sequence new and still reproducible
    # for a given chunk plan (tuning run in chunks, or resubmitted after the wall time).
    study.sampler = TPESampler(seed=int(cfg["seeds"]["tune"]) + finished,
                               constant_liar=bool(trial_workers > 1))
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=max(1, trial_workers))

    done = [t for t in study.trials if t.value is not None]
    # counted after optimize, over the whole study, so a resubmitted job reports the
    # real total rather than the difference between a before and an after count
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    best = min(done, key=lambda t: t.value)
    se = float(np.std(best.user_attrs["per_fold"], ddof=1) / np.sqrt(len(best.user_attrs["per_fold"])))
    band = [t for t in done if t.value <= best.value + se]
    chosen = sorted(band, key=lambda t: simplest(t, model, n_features))[0]
    n_failed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
    return {"study": study_name, "n_trials": len(done), "n_pruned": n_pruned,
            "n_failed_or_interrupted": n_failed,
            "se": se, "band_size": len(band),
            "workers": int(trial_workers),
            "pruner": "median" if pruner is not None else "none",
            "sampler": "tpe_constant_liar" if trial_workers > 1 else "tpe",
            "best": {"trial": best.number, "value": best.value, "params": best.params},
            "chosen": {"trial": chosen.number, "value": chosen.value, "params": chosen.params},
            "per_fold_chosen": chosen.user_attrs["per_fold"],
            "not_stopped_chosen": chosen.user_attrs["not_stopped"],
            "n_trees_final": int(np.median(chosen.user_attrs["best_iters"]))}


def tune_grid(cfg, d, splits, dataset, n_features, quick, fold_workers=1):
    """The Random Forest grid: every point of max_features x min_samples_leaf.

    Points run one after another and each is cached, so a crash resumes rather than
    restarting 120 fits. Within a point the folds are parallel, like everywhere else.
    """
    g = cfg["random_forest"]
    points = [{"max_features": mf, "min_samples_leaf": int(leaf)}
              for mf, leaf in itertools.product(g["max_features"], g["min_samples_leaf"])]
    # keyed like the Optuna studies, by config and code: a cached point fitted by older
    # fit code must not be mixed with new ones after a resubmit
    cache = (OUT / "rf_grid" /
             f"{dataset}_{'quick' if quick else 'full'}_{D.config_hash(cfg)}_{cv.code_hash()}")
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(points):
        f = cache / f"point_{i:02d}.json"
        if f.exists():
            rows.append(json.loads(f.read_text(encoding="utf-8")))
            print(f"  rf {p} -> cached {rows[-1]['value']:.4f}")
            continue
        mean, per_fold, iters, hits = evaluate("rf", p, dataset, cfg, d, splits, quick,
                                               trial=None, fold_workers=fold_workers)
        row = {"params": p, "value": mean, "per_fold": per_fold,
               "not_stopped": hits, "n_trees": iters[0]}
        f.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        print(f"  rf {p} -> {mean:.4f}")
    best = min(rows, key=lambda r: r["value"])
    se = float(np.std(best["per_fold"], ddof=1) / np.sqrt(len(best["per_fold"])))
    band = [r for r in rows if r["value"] <= best["value"] + se]
    chosen = sorted(band, key=lambda r: (-r["params"]["min_samples_leaf"],
                                         feature_fraction(r["params"]["max_features"], n_features)))[0]
    return {"n_trials": len(rows), "se": se, "band_size": len(band),
            "pruner": "none", "sampler": "grid",
            "best": {"value": best["value"], "params": best["params"]},
            "chosen": {"value": chosen["value"], "params": chosen["params"]},
            "per_fold_chosen": chosen["per_fold"],
            "not_stopped_chosen": chosen["not_stopped"],
            "n_trees_final": int(chosen["n_trees"]),
            "grid": rows}


if __name__ == "__main__":
    raise SystemExit(main())