#!/usr/bin/env python3
"""Verify the prediction table and its consistency with the training table.

Per written month:

  1. the column set is exactly the model inputs plus allowed extras, so a
     dropped input cannot hide inside the intersection of two column sets;
  2. rows = n_days x n_cells, (cell_id, date) unique;
  3. every model input outside ALLOWED_NAN has zero NaN;
  4. every training row of the month has a prediction row (inner join), and the
     feature values are identical, NaN pattern included.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from build_prediction import contract
from model_inputs import MODEL_INPUTS, check_columns
from utils import ROOT, days_in_month, fingerprint, read_inputs
from verify_common import cells, check, summary

PRED = ROOT / "data" / "processed" / "prediction"
FEAT = ROOT / "data" / "interim" / "features" / "all"
TARGET = ROOT / "data" / "processed" / "target" / "target_table.parquet"


def same(a: np.ndarray, b: np.ndarray) -> tuple[bool, float, int]:
    a = np.asarray(a, "float64"); b = np.asarray(b, "float64")
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    n_mis = int((nan_a != nan_b).sum())
    both = ~nan_a & ~nan_b
    diff = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
    return (n_mis == 0 and diff == 0.0), diff, n_mis


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the prediction table")
    ap.add_argument("--pred", default=str(PRED))
    ap.add_argument("--target", default=str(TARGET))
    args = ap.parse_args()

    pred_dir = ROOT / args.pred if not str(args.pred).startswith(str(ROOT)) else args.pred
    from pathlib import Path
    pred_dir = Path(pred_dir)
    target_path = Path(args.target)

    n_cells = len(cells())
    files = sorted(pred_dir.glob("*/prediction_*.parquet"))
    check(len(files) > 0, f"prediction months found: {len(files)}")
    if not files:
        return summary()

    tgt = pd.read_parquet(target_path) if target_path.exists() else None
    if tgt is not None:
        problems = check_columns(tgt.columns)
        check(not problems, "training table carries the model inputs"
                            + (f" - {problems}" if problems else ""))

    for f in files:
        y, m = int(f.stem.split("_")[-2]), int(f.stem.split("_")[-1])
        names = pq.read_schema(f).names
        problems = check_columns(names)
        check(not problems, f"{y}-{m:02d} column set complete and classified"
                            + (f" - {problems}" if problems else ""))

        # the table must have been built from the features file now on disk
        recorded = read_inputs(f) or {}
        current = fingerprint(FEAT / f"features_{y}_{m:02d}.parquet")
        check(recorded.get("features") == current,
              f"{y}-{m:02d} prediction built from the current features file"
              + (f" - recorded {recorded.get('features')}, now {current}" if
                 recorded.get("features") != current else ""))

        d = pd.read_parquet(f)
        nd = len(days_in_month(y, m))
        check(len(d) == nd * n_cells, f"{y}-{m:02d} rows = {nd*n_cells:,} (got {len(d):,})")
        check(not d.duplicated(subset=["cell_id", "date"]).any(), f"{y}-{m:02d} keys unique")

        required, _ = contract(d.columns)
        nan = {c: int(d[c].isna().sum()) for c in required}
        broken = {c: v for c, v in nan.items() if v}
        check(not broken, f"{y}-{m:02d} {len(required)} contract columns complete"
                          + (f" - broken: {broken}" if broken else ""))

        if tgt is None:
            continue
        tm = tgt[(tgt["date"].dt.year == y) & (tgt["date"].dt.month == m)]
        if tm.empty:
            continue
        shared = [c for c in MODEL_INPUTS if c in names and c in tm.columns]
        mrg = tm[["cell_id", "date"] + shared].merge(
            d[["cell_id", "date"] + shared], on=["cell_id", "date"],
            how="inner", suffixes=("_tr", "_pr"))
        missing_rows = len(tm) - len(mrg)
        check(missing_rows == 0,
              f"{y}-{m:02d} every training row has a prediction row "
              f"({len(mrg)}/{len(tm)} matched, {missing_rows} unmatched)")

        bad = []
        for c in shared:
            ok, diff, nmis = same(mrg[f"{c}_tr"], mrg[f"{c}_pr"])
            if not ok:
                bad.append((c, round(diff, 8), nmis))
        check(not bad, f"{y}-{m:02d} {len(shared)} model inputs identical between the tables"
                       + (f" - bad: {bad[:3]}" if bad else ""))

    return summary()


if __name__ == "__main__":
    raise SystemExit(main())