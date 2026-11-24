#!/usr/bin/env python3
"""Final retrain and 90% split-conformal intervals. Guarded; Phase 3 only.

Rule
----
Refuses unless `data/interim/model/OPEN_TEST` exists and `--open-test` is passed.

The run decides the model, the dataset and the parameter file, so `--run R3` fits
XGBoost on dataset A and `--run R4` fits the Random Forest -- not a LightGBM
wearing another run's label.

The interval is split-conformal: the ceil((n+1) * 0.9)-th smallest absolute
residual on the calibration cells, pooled and per region.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import data as D
import features as F
import models

OUT = D.OUT
OPEN = OUT / "OPEN_TEST"

RUNS = {"R1": ("lgbm", "A", "lgbm"), "R2": ("lgbm", "C", "lgbm"),
        "R3": ("xgb", "A", "xgb"), "R4": ("rf", "A", "rf")}


def guard(open_test: bool) -> None:
    if not OPEN.exists():
        raise SystemExit(f"refusing: {OPEN} does not exist")
    if not open_test:
        raise SystemExit("refusing: pass --open-test to confirm the test sets may be opened")


def conformal_q(resid: np.ndarray, level: float = 0.9) -> float:
    if len(resid) == 0:
        return float("nan")
    r = np.sort(np.abs(resid))
    k = int(np.ceil((len(r) + 1) * level)) - 1
    return float(r[min(k, len(r) - 1)])


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrain and calibrate intervals")
    ap.add_argument("--run", default="R1", choices=sorted(RUNS))
    ap.add_argument("--open-test", action="store_true")
    args = ap.parse_args()
    guard(args.open_test)

    model, dataset, pname = RUNS[args.run]
    cfg = D.load_config()
    d = D.load()
    splits = __import__("pandas").read_parquet(OUT / "splits.parquet")
    cols = F.columns(dataset)
    par = json.loads((OUT / f"params_{pname}.json").read_text(encoding="utf-8"))
    if par.get("quick"):
        raise SystemExit(f"refusing: params_{pname}.json was written in quick mode; "
                         f"final models need the full parameter set")
    tuned = par["chosen"]["params"]
    n_trees, n_trees_source, per_fold_trees = choose_trees(args.run, par)

    train = D.training_rows(d, splits)
    calib_cells = set(splits.loc[splits["role"] == "calibration", "cell_id"])
    calib = d[d["cell_id"].isin(calib_cells) & d["year"].between(D.DEV_FIRST_YEAR, D.DEV_LAST_YEAR)]

    predictor, model_file = fit_final(args.run, model, cfg, tuned, cols, train, n_trees)
    resid = calib["pm25"].to_numpy(float) - predictor(calib)
    per_region = {int(r): conformal_q(g["pm25"].to_numpy(float) - predictor(g))
                  for r, g in calib.groupby("on_region")}

    (OUT / f"final_{args.run}.json").write_text(json.dumps(
        {"run": args.run, "model": model, "dataset": dataset, "n_trees": n_trees,
         "n_trees_source": n_trees_source, "n_trees_per_fold": per_fold_trees,
         "model_file": model_file,
         "n_features": len(cols), "q90": conformal_q(resid),
         "q90_by_region": per_region, "n_calib": int(len(calib)),
         "config_hash": D.config_hash(cfg)}, indent=2), encoding="utf-8")
    print(f"{args.run} ({model}, dataset {dataset}): q90 {conformal_q(resid):.2f} "
          f"on {len(calib):,} calibration rows")
    return 0


def choose_trees(run: str, par: dict) -> tuple[int, str, dict]:
    """Tree count for the retrain: the median of the per-FOLD counts, and where it came from.

    The median is taken over folds, not over scored rows. Folds differ in size, so a
    row-median would weight the larger folds more and could disagree with the
    `n_trees_final` that `tune.py` computed for R1.

    R2 must not inherit R1's count: it trains on dataset C, where early stopping
    chose different trees. `params_*.json` is the fallback only when the OOF file is
    absent, which for R2 is refused outright.
    """
    oof = OUT / "full" / run / "group10" / "oof.parquet"
    if oof.exists():
        import pandas as pd
        d = pd.read_parquet(oof, columns=["fold", "n_trees"])
        per_fold = d.groupby("fold")["n_trees"].first()
        med = int(per_fold.median())
        return med, f"oof_median_per_fold:{run}/group10",             {str(k): int(v) for k, v in per_fold.items()}
    if run == "R2":
        raise SystemExit(f"refusing: {oof} missing, and R2 must not borrow R1's tree count")
    return int(par["n_trees_final"]), "params_file", {}


def fit_final(run: str, model: str, cfg: dict, tuned: dict, cols: list[str], train,
              n_trees: int):
    """Retrain on all development rows; returns (predict, file name).

    The file is named per run, so finalising R2 cannot overwrite R1's booster.
    """
    if model == "lgbm":
        import lightgbm as lgb
        b = lgb.train(models.lgbm_params(cfg, tuned),
                      lgb.Dataset(train[cols], label=train["pm25"]), num_boost_round=n_trees)
        name = f"final_{run}.txt"
        b.save_model(str(OUT / name))
        return (lambda df: b.predict(df[cols], num_iteration=n_trees)), name
    if model == "xgb":
        import xgboost as xgb
        b = xgb.train(models.xgb_params(cfg, tuned),
                      xgb.DMatrix(train[cols], label=train["pm25"]), num_boost_round=n_trees)
        name = f"final_{run}.json"
        b.save_model(str(OUT / name))
        return (lambda df: b.predict(xgb.DMatrix(df[cols]), iteration_range=(0, n_trees))), name
    rf = models.fit_rf(train, cols, models.rf_params(cfg, tuned))
    import joblib
    name = f"final_{run}.joblib"
    joblib.dump(rf, OUT / name)
    return (lambda df: rf.predict(df[cols].to_numpy())), name


if __name__ == "__main__":
    raise SystemExit(main())