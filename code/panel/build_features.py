#!/usr/bin/env python3
"""Feature stage: AOD context features built from the observed field only.

Scope
-----
There is deliberately **no AOD gap-fill** in this build. The raw `aod055` and
`aod047` keep their native missingness and the model meets NaN directly
(LightGBM routes it). This module adds deterministic *context* around the
observations - neighbourhood means and the age of the last observation - which
help exactly when the cell itself is cloud-masked.

Direction and continuity
------------------------
`--window backward` (default) uses day D and earlier only. Each month is warmed
up on the previous month's last seven days, so days 1-7 of a month see the same
history they would in a continuous run; the warm-up rows are dropped before
writing. Without that, a month boundary would quietly reset `aod_last_obs`.

Everything is computed on the lattice (a rows x cols array) with box/disk
kernels, never with a per-cell loop.
"""
from __future__ import annotations

import argparse

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.ndimage import convolve

from utils import ROOT, Failures, fingerprint, months, rebuild_reason, setup_logging

CELLS = ROOT / "data" / "interim" / "spine" / "cells.parquet"
BASE = ROOT / "data" / "interim" / "base"
OUT = ROOT / "data" / "interim" / "features"

RADII_KM = {"25": 25, "50": 50, "100": 100}
CELL_KM = 5.0
AGES = 7


def grid_layout(cells: pd.DataFrame):
    r0, c0 = int(cells["row"].min()), int(cells["col"].min())
    r1, c1 = int(cells["row"].max()), int(cells["col"].max())
    return (r1 - r0 + 1, c1 - c0 + 1,
            (cells["row"].to_numpy() - r0).astype("int64"),
            (cells["col"].to_numpy() - c0).astype("int64"))


def disk(radius_km: float) -> np.ndarray:
    """Lattice disk, radius in cells. 50 km is 10 cells of 5 km, not a geodesic disk."""
    r = int(np.ceil(radius_km / CELL_KM))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return ((x * x + y * y) <= (radius_km / CELL_KM) ** 2).astype("float32")


def _neighbour(vals, valid, kernel):
    num = convolve(np.where(valid, vals, 0.0), kernel, mode="constant", cval=0.0)
    den = convolve(valid.astype("float32"), kernel, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan), den


def context_month(df: pd.DataFrame, cells: pd.DataFrame, prev_days: int = 3):
    """Context features for every row of df, using day D and earlier only.

    `df` may span more than the month of interest (the warm-up rows); the caller
    slices the result.
    """
    present = cells[cells["cell_id"].isin(df["cell_id"].unique())]
    h, w, rr, cc = grid_layout(present)
    pos = {cid: i for i, cid in enumerate(present["cell_id"].to_numpy())}
    days = np.sort(df["date"].unique())
    n = len(days)

    aod_grid = np.full((n, h, w), np.nan, "float32")
    for i, d in enumerate(days):
        sub = df[df["date"] == d]
        # map each row to its cell position by id: the rows of a day are not
        # guaranteed to follow the spine order
        idx = np.fromiter((pos[c] for c in sub["cell_id"].to_numpy()),
                          dtype="int64", count=len(sub))
        v = np.full(h * w, np.nan, "float32")
        v[rr[idx] * w + cc[idx]] = sub["aod055"].to_numpy()
        aod_grid[i] = v.reshape(h, w)

    n_cells = len(present)
    out = {f"aod_nbr_{k}km_d0": np.full((n, n_cells), np.nan, "float32") for k in RADII_KM}
    out.update({f"aod_nbr_{k}km_d0_n": np.zeros((n, n_cells), "float32") for k in RADII_KM})
    out["aod_nbr_50km_d1_3"] = np.full((n, n_cells), np.nan, "float32")
    out["aod_nbr_50km_d1_3_n"] = np.zeros((n, n_cells), "float32")
    out["aod_last_obs"] = np.full((n, n_cells), np.nan, "float32")
    out["aod_last_obs_age_d"] = np.full((n, n_cells), np.nan, "float32")

    kernels = {k: disk(r) for k, r in RADII_KM.items()}
    last_val = np.full(n_cells, np.nan, "float32")
    last_age = np.full(n_cells, np.nan, "float32")

    for i in range(n):
        g = aod_grid[i]
        valid = np.isfinite(g)

        for k, kern in kernels.items():
            mean, cnt = _neighbour(g, valid, kern)
            out[f"aod_nbr_{k}km_d0"][i] = mean[rr, cc]
            out[f"aod_nbr_{k}km_d0_n"][i] = cnt[rr, cc]

        lo = max(0, i - prev_days)
        if i > lo:
            kern = kernels["50"]
            num = np.zeros((h, w), "float32")
            den = np.zeros((h, w), "float32")
            for j in range(lo, i):
                vj = np.isfinite(aod_grid[j])
                num += convolve(np.where(vj, aod_grid[j], 0.0), kern, mode="constant")
                den += convolve(vj.astype("float32"), kern, mode="constant")
            with np.errstate(invalid="ignore", divide="ignore"):
                out["aod_nbr_50km_d1_3"][i] = np.where(den > 0, num / den, np.nan)[rr, cc]
            out["aod_nbr_50km_d1_3_n"][i] = den[rr, cc]

        # age is carried forward first, then reset where today is observed, so a
        # value seen today never also counts as yesterday's
        last_age = np.where(np.isnan(last_age), np.nan, last_age + 1.0)
        cur = np.isfinite(g[rr, cc])
        last_val = np.where(cur, g[rr, cc], last_val)
        last_age = np.where(cur, 0.0, last_age)
        expired = ~(last_age <= AGES)
        last_age = np.where(expired, np.nan, last_age)
        last_val = np.where(expired, np.nan, last_val)
        out["aod_last_obs"][i] = last_val
        out["aod_last_obs_age_d"][i] = last_age

    # re-expand to the row order of df, through the same id map
    row_of = np.array([pos[c] for c in df["cell_id"].to_numpy()])
    day_of = {d: i for i, d in enumerate(days)}
    day_ix = np.array([day_of[d] for d in df["date"].to_numpy()])
    return {k: v[day_ix, row_of] for k, v in out.items()}


def _previous_month(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def _inputs(kind: str, year: int, month: int) -> dict:
    """The files this month is built from: its base, and its predecessor's."""
    src = BASE / kind
    cur = src / f"base_{kind}_{year}_{month:02d}.parquet"
    py, pm = _previous_month(year, month)
    prev = src / f"base_{kind}_{py}_{pm:02d}.parquet"
    return {"base": fingerprint(cur), "prev": fingerprint(prev)}


def _write(df: pd.DataFrame, out: Path, inputs: dict) -> None:
    """Write the month atomically with its inputs recorded in the metadata."""
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    md = dict(tbl.schema.metadata or {})
    md[b"inputs"] = json.dumps(inputs).encode()
    md[b"warm_start"] = b"true" if inputs.get("prev") else b"false"
    tmp = out.with_suffix(".tmp")
    pq.write_table(tbl.replace_schema_metadata(md), tmp, compression="zstd")
    tmp.replace(out)


def cmd_context(args):
    cells_all = pd.read_parquet(CELLS)
    src = BASE / args.cells
    dst = OUT / args.cells
    dst.mkdir(parents=True, exist_ok=True)
    log = setup_logging(dst / "context.log")
    fails = Failures()

    for y, m in months():
        f = src / f"base_{args.cells}_{y}_{m:02d}.parquet"
        if not f.exists():
            continue
        out = dst / f"features_{y}_{m:02d}.parquet"
        cur_inputs = _inputs(args.cells, y, m)
        reason = rebuild_reason(out, cur_inputs)
        if reason is None:
            continue
        if cur_inputs["prev"] is None and (y, m) != (2010, 1):
            log.warning("%d-%02d has no %d-%02d base: cold start, will be rebuilt "
                        "once that month exists", y, m, *_previous_month(y, m))
        log.info("features %d-%02d: %s", y, m, reason)

        def work():
            df = pd.read_parquet(f)
            df["date"] = pd.to_datetime(df["date"])
            first = df["date"].min()

            if cur_inputs["prev"] is not None:
                py_, pm_ = _previous_month(y, m)
                prev_path = src / f"base_{args.cells}_{py_}_{pm_:02d}.parquet"
                pdf = pd.read_parquet(prev_path)
                pdf["date"] = pd.to_datetime(pdf["date"])
                warm = pdf[pdf["date"] >= first - pd.Timedelta(days=AGES)]
                frame = pd.concat([warm, df], ignore_index=True)
            else:
                frame = df

            ctx = context_month(frame, cells_all)
            keep = (frame["date"] >= first).to_numpy()
            res = pd.DataFrame({"cell_id": frame["cell_id"].to_numpy()[keep],
                                "date": frame["date"].to_numpy()[keep]})
            for k, v in ctx.items():
                res[k] = v[keep]
            _write(res, out, cur_inputs)

        fails.guard(log, f"context {y}-{m:02d}", work)

    fails.write(dst / "failures.json")
    log.info("failures: %d", len(fails.items))
    return 1 if fails.items else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("context", help="neighbourhood context from observed AOD")
    c.add_argument("--cells", choices=["station", "all"], required=True,
                   help="all is the one the tables use; station exists for QA only")
    c.set_defaults(fn=cmd_context)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())