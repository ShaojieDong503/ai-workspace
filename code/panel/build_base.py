#!/usr/bin/env python3
"""Base panel driver: one month of raw values, no filling, no aggregates.

Every extractor returns arrays of shape (n_days, n_cells) in cell_idx order, so
a month is built by concatenating them positionally and writing once. Rows are
date-major, which is the (date, cell_id) sort the build requires.

Resumable: a month is skipped when its output exists and its row count matches
n_days * n_cells. A month that raises is logged with its traceback and the run
continues; the failure list is written at the end and the exit code is non-zero.

Usage
-----
    python build_base.py --cells station
    python build_base.py --cells all --years 2010 2011 --workers 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import extract_fire
import extract_hms
import extract_maiac
import extract_merra
from utils import END, ROOT, START, Failures, days_in_month, months, setup_logging

CELLS = ROOT / "data" / "interim" / "spine" / "cells.parquet"
STATIONS = ROOT / "data" / "interim" / "stations" / "stations.parquet"
OUT = ROOT / "data" / "interim" / "base"

EXTRACTORS = {
    "maiac": extract_maiac.extract,
    "merra": extract_merra.extract,
    "hms": extract_hms.extract,
    "fire": extract_fire.extract,
}

# the columns every month must carry, so a schema drift is caught on resume
REQUIRED = [
    "merra_pblh", "merra_tlml", "merra_u10m", "merra_totexttau",
    "hms_any", "hms_n_polygons", "hms_density_max",
    "fire_dist_nearest_km", "fire_count_100km", "fire_frp_sum_100km",
    "aod055", "aod047", "injection_height", "smoke_aod055",
    "smoke_mask_frac", "aod055_npix", "maiac_npix_total", "maiac_covered",
]


def cell_subset(kind: str) -> np.ndarray:
    if kind == "all":
        return pd.read_parquet(CELLS)["cell_idx"].to_numpy()
    st = pd.read_parquet(STATIONS)
    return np.sort(st["cell_idx"].unique())


def build_month(year: int, month: int, cell_idx: np.ndarray, log) -> pd.DataFrame:
    days = days_in_month(year, month)
    parts, meta = {}, {}
    for name, fn in EXTRACTORS.items():
        res = fn(year, month, cell_idx)
        for k in [k for k in res if k.startswith("_")]:
            meta[k] = res.pop(k)
        parts.update(res)
        log.info("  %s: %d columns", name, len(res))

    n, m = len(days), len(cell_idx)
    for k, v in list(parts.items()):
        if v.shape == (m,):
            # a static per-cell column (maiac_npix_total, maiac_covered): the
            # table is long, so repeat it for every day
            parts[k] = np.broadcast_to(v, (n, m)).copy()
        elif v.shape != (n, m):
            raise RuntimeError(f"{k}: shape {v.shape} != {(n, m)}")

    # cell_idx is a position in the spine, cell_id is the lattice id. Writing
    # cell_idx into the cell_id column silently breaks every downstream join.
    spine_ids = pd.read_parquet(CELLS).sort_values("cell_idx")["cell_id"].to_numpy()
    ids = spine_ids[cell_idx]
    df = pd.DataFrame({"cell_id": np.tile(ids.astype("int32"), n)})
    dates = np.repeat(np.array(days, dtype="datetime64[D]"), m)
    df.insert(1, "date", dates.astype("datetime64[ns]"))
    for k, v in parts.items():
        df[k] = v.ravel()
    for k, v in meta.items():
        if k == "_missing_days" and len(v):
            log.warning("  HMS missing days: %s", list(v)[:10])
        if k == "_empty_tile_days" and int(v[0]):
            log.warning("  MAIAC empty tile-days in this month: %d", int(v[0]))
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the base panel month by month")
    ap.add_argument("--cells", choices=["station", "all"], required=True)
    ap.add_argument("--years", type=int, nargs=2, default=[START.year, END.year])
    ap.add_argument("--month", type=int, nargs="*", default=None,
                    help="restrict to these month numbers, for a one-month canary")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out) / args.cells
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out / "build.log")
    fails = Failures()
    cell_idx = cell_subset(args.cells)
    log.info("cells=%s n_cells=%d", args.cells, len(cell_idx))

    written = skipped = 0
    for y, m in months(START, END):
        if not (args.years[0] <= y <= args.years[1]):
            continue
        if args.month and m not in args.month:
            continue
        path = out / f"base_{args.cells}_{y}_{m:02d}.parquet"
        n_days = len(days_in_month(y, m))
        if path.exists():
            try:
                if len(pd.read_parquet(path, columns=["cell_id"])) == n_days * len(cell_idx):
                    skipped += 1
                    continue
            except Exception:
                pass                                     # rebuild a corrupt month
        log.info("building %d-%02d", y, m)
        df = fails.guard(log, f"{y}-{m:02d}", build_month, y, m, cell_idx, log)
        if df is None:
            continue
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            log.error("%d-%02d missing columns %s", y, m, missing)
            fails.items.append({"unit": f"{y}-{m:02d}", "error": f"missing {missing}"})
            continue
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(path)                      # atomic publish
        written += 1

    fails.write(out / "failures.json")
    log.info("written=%d skipped=%d failed=%d", written, skipped, len(fails.items))
    return 1 if fails.items else 0


if __name__ == "__main__":
    raise SystemExit(main())
