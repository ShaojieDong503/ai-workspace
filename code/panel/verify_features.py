#!/usr/bin/env python3
"""Verify the context features: brute force, backward-only, warm-up continuity.

Three checks, each against an independent recomputation:

  brute force    recompute aod_nbr_50km_d0 for every cell with an observation on
                 one day, by looping over the observed cells and averaging those
                 within a 10-cell lattice disk - no convolution.
  backward-only  recompute the month from a frame truncated at day D and require
                 every row for D to be unchanged, in every context column.
  warm-up        the features written for month M must equal a continuous run
                 over M-1 and M, in every context column. A month boundary is
                 the only place the warm-up can go wrong, and a within-month
                 truncation test cannot see it.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import build_features as bf
from model_inputs import CONTEXT
from utils import ROOT
from verify_common import BASE, check, compare, summary

FEAT = ROOT / "data" / "interim" / "features"
CELLS = ROOT / "data" / "interim" / "spine" / "cells.parquet"


def load_base(kind: str, year: int, month: int) -> pd.DataFrame | None:
    p = BASE / kind / f"base_{kind}_{year}_{month:02d}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d["date"] = pd.to_datetime(d["date"])
    return d


def brute_force(cs, kind: str, year: int, month: int) -> None:
    base = load_base(kind, year, month)
    feat = pd.read_parquet(FEAT / kind / f"features_{year}_{month:02d}.parquet")
    feat["date"] = pd.to_datetime(feat["date"])

    day = pd.Timestamp(base["date"].unique()[10])
    obs = base[(base["date"] == day) & base["aod055"].notna()][["cell_id", "aod055"]]
    target = feat[feat["date"] == day]

    # int64 before squaring: row/col are int16 and dr*dr overflows past 181
    rr = cs.loc[obs["cell_id"], "row"].to_numpy("int64")
    cc = cs.loc[obs["cell_id"], "col"].to_numpy("int64")
    vals = obs["aod055"].to_numpy("float64")

    # the loop is exact but quadratic; sample when the lattice is full
    if len(target) > 2000:
        target = target.sample(n=2000, random_state=0)
    got = target["aod_nbr_50km_d0"].to_numpy("float64")
    want = np.empty(len(target))
    for i, cid in enumerate(target["cell_id"].to_numpy()):
        dr = rr - int(cs.loc[cid, "row"])
        dc = cc - int(cs.loc[cid, "col"])
        near = (dr * dr + dc * dc) <= 100
        want[i] = vals[near].mean() if near.any() else np.nan
    compare(f"aod_nbr_50km_d0 on {day.date()} ({len(target)} rows)", got, want)


def backward_only(kind: str, year: int, month: int) -> None:
    base = load_base(kind, year, month)
    cells = pd.read_parquet(CELLS).reset_index(drop=True)
    day = pd.Timestamp(base["date"].unique()[10])

    full = bf.context_month(base, cells)
    part = bf.context_month(base[base["date"] <= day].copy(), cells)

    rows_full = np.flatnonzero((base["date"] == day).to_numpy())
    rows_part = np.flatnonzero((base["date"] <= day).to_numpy())
    offset = len(rows_part) - len(rows_full)     # same day is last in the truncated frame

    bad = []
    for k in full:
        a = np.asarray(full[k])[rows_full]
        b = np.asarray(part[k])[rows_part][offset:]
        if not np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-12):
            bad.append(k)
    check(not bad, f"backward-only: all {len(rows_full)} rows for {day.date()} unchanged "
                   f"({len(full)} columns)" + (f" - differ: {bad}" if bad else ""))


def warmup_continuity(kind: str, year: int, month: int) -> None:
    prev_y, prev_m = (year - 1, 12) if month == 1 else (year, month - 1)
    cur = load_base(kind, year, month)
    prev = load_base(kind, prev_y, prev_m)
    if prev is None:
        print(f"no {prev_y}-{prev_m:02d} base file - warm-up test skipped")
        return
    feat = pd.read_parquet(FEAT / kind / f"features_{year}_{month:02d}.parquet")
    feat["date"] = pd.to_datetime(feat["date"])

    cont = pd.concat([prev, cur], ignore_index=True).sort_values(["date", "cell_id"])
    ctx = bf.context_month(cont, pd.read_parquet(CELLS).reset_index(drop=True))
    keep = (cont["date"] >= cur["date"].min()).to_numpy()

    bad = []
    for k in CONTEXT:
        a = np.asarray(ctx[k])[keep]
        b = feat[k].to_numpy("float64")
        if len(a) != len(b) or not np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-12):
            bad.append(k)
    check(not bad, f"warm-up: {int(keep.sum()):,} rows of {year}-{month:02d} equal a "
                   f"continuous {prev_y}-{prev_m:02d}+{year}-{month:02d} run, "
                   f"{len(CONTEXT)} columns" + (f" - differ: {bad}" if bad else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the context features")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--year", type=int, default=2015)
    ap.add_argument("--month", type=int, default=7)
    args = ap.parse_args()

    cs = pd.read_parquet(CELLS).set_index("cell_id")
    brute_force(cs, args.cells, args.year, args.month)
    backward_only(args.cells, args.year, args.month)
    warmup_continuity(args.cells, args.year, args.month)
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())