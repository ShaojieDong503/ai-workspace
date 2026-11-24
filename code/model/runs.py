#!/usr/bin/env python3
"""One entry per run, R1-R4.

Rule
----
Each run names its model, dataset and parameter file; `cv.run_cv` fits through
`models.fit_predict`. R2 shares R1's LightGBM settings on dataset C and re-picks
only the number of trees, by early stopping.

Parameter files are mode-specific: quick tuning writes `quick/params_*.json`, full
tuning writes `params_*.json`. Loading the wrong one is refused rather than
silently used.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

import cv
import data as D
import features as F
import folds as FLD

OUT = D.OUT

RUNS = {
    "R1": {"model": "lgbm", "dataset": "A", "params": "lgbm"},
    "R2": {"model": "lgbm", "dataset": "C", "params": "lgbm"},
    "R3": {"model": "xgb",  "dataset": "A", "params": "xgb"},
    "R4": {"model": "rf",   "dataset": "A", "params": "rf"},
}


def load_params(name: str, quick: bool) -> dict:
    base = OUT / "quick" if quick else OUT
    p = base / f"params_{name}.json"
    if not p.exists():
        raise SystemExit(f"missing {p}; run tune.py --model {name}"
                         f"{' --quick' if quick else ''} first")
    j = json.loads(p.read_text(encoding="utf-8"))
    if bool(j.get("quick")) != bool(quick):
        raise SystemExit(f"{p} was written in "
                         f"{'quick' if j.get('quick') else 'full'} mode but this run is "
                         f"{'quick' if quick else 'full'}")
    return j["chosen"]["params"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one experiment")
    ap.add_argument("run", choices=sorted(RUNS))
    ap.add_argument("--scheme", default="all", choices=["all", "group10", "loro", "loyo"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="discard folds from a different run key")
    ap.add_argument("--fold-workers", type=int, default=None,
                    help="folds fitted at once; defaults to one per fold, capped by cpus")
    args = ap.parse_args()

    cfg = D.load_config()
    d = D.load()
    splits = pd.read_parquet(OUT / "splits.parquet")
    schemes = ["group10", "loro", "loyo"] if args.scheme == "all" else [args.scheme]
    spec = RUNS[args.run]
    cols = F.columns(spec["dataset"])
    params = load_params(spec["params"], args.quick)

    for scheme in schemes:
        n_folds = len(FLD.plan(scheme, splits, pd.read_parquet(OUT / "folds.parquet")))
        fold_workers = args.fold_workers or cv.default_fold_workers(n_folds)
        base = cv.run_cv(tag=args.run, dataset=spec["dataset"], scheme=scheme, cfg=cfg, d=d,
                         splits=splits, model=spec["model"], params=params, cols=cols,
                         quick=args.quick, fresh=args.fresh, fold_workers=fold_workers)
        print(f"{args.run} {scheme} done -> {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())