#!/usr/bin/env python3
"""Training table: one row per cell-day with at least one station.

Rule
----
The unit is the cell-day. Stations are mapped to the cell containing them;
where several share a cell their UTC-6 means are averaged and `n_stations`
records how many contributed. Station-days with a negative daily mean are
dropped before the cell average.

The features come from the **all-cell** run, then are subset to the station
cell-days: a station cell is a biased sample of the domain, so neighbourhood
features computed there would differ from the ones the prediction table carries.

Partial builds are refused. A month that has targets but no base or features is
a station-day loss, not something to skip quietly, so the script prints what
would be lost and exits non-zero unless --allow-partial is given.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from model_inputs import MODEL_INPUTS, check_columns
from panel_calendar import add_calendar
from province import add_province
from utils import ROOT

BASE_ALL = ROOT / "data" / "interim" / "base" / "all"
FEAT_ALL = ROOT / "data" / "interim" / "features" / "all"
STATION_DAY = ROOT / "data" / "interim" / "stations" / "station_day.parquet"
CELL_ONTARIO = ROOT / "data" / "interim" / "spine" / "cell_ontario.parquet"
OUT = ROOT / "data" / "processed" / "target"


def cell_day_targets() -> pd.DataFrame:
    sd = pd.read_parquet(STATION_DAY)
    neg = sd["pm25"] < 0
    print(f"negative station-days dropped: {int(neg.sum()):,} of {len(sd):,}")
    sd = sd[~neg]
    g = (sd.groupby(["cell_id", "date"], as_index=False)
           .agg(pm25=("pm25", "mean"),
                n_stations=("station_uid", "size"),
                n_hours_min=("n_valid_hours", "min")))
    g["pm25"] = g["pm25"].astype("float32")
    g["n_stations"] = g["n_stations"].astype("int8")
    g["n_hours_min"] = g["n_hours_min"].astype("int8")
    g["date"] = pd.to_datetime(g["date"])
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the cell-day training table")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--allow-partial", action="store_true",
                    help="write a table even when months are missing (loss is printed)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    targets = cell_day_targets()
    targets["ym"] = targets["date"].dt.strftime("%Y_%m")
    total = len(targets)

    frames, missing = [], []
    for ym, tgt in targets.groupby("ym"):
        fb = BASE_ALL / f"base_all_{ym}.parquet"
        ff = FEAT_ALL / f"features_{ym}.parquet"
        if not (fb.exists() and ff.exists()):
            missing.append((ym, len(tgt)))
            continue
        d = pd.read_parquet(fb).merge(pd.read_parquet(ff), on=["cell_id", "date"], how="inner")
        d["date"] = pd.to_datetime(d["date"])
        frames.append(tgt.drop(columns="ym").merge(d, on=["cell_id", "date"], how="inner"))

    if missing:
        lost = sum(n for _, n in missing)
        print(f"months without base or features: {len(missing)} "
              f"(first: {[m for m, _ in missing[:3]]})")
        print(f"station-days lost: {lost:,} of {total:,} ({100*lost/total:.1f}%)")
        if not args.allow_partial:
            print("refusing to write a partial training table; pass --allow-partial")
            return 1

    if not frames:
        print("no months with both base/all and features/all")
        return 1

    df = pd.concat(frames, ignore_index=True)
    df["date"] = df["date"].astype("datetime64[ns]")
    df = add_province(df, name_col="province_state")
    df = add_calendar(df, "date")
    df = df.merge(pd.read_parquet(CELL_ONTARIO), on="cell_id", how="left", validate="m:1")
    if df["dist_ontario_km"].isna().any():
        print("cells without dist_ontario_km; run ontario_geo.py")
        return 1

    problems = check_columns(df.columns)
    if problems:
        print("; ".join(problems))
        return 1

    cols = list(df.columns)
    cols.remove("province_code")
    cols.insert(2, "province_code")
    df = df[cols].sort_values(["date", "cell_id"]).reset_index(drop=True)
    tmp = out / "target_table.tmp"
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(out / "target_table.parquet")

    print(f"rows: {len(df):,}  cells: {df.cell_id.nunique():,}  "
          f"model inputs: {sum(c in df.columns for c in MODEL_INPUTS)}/{len(MODEL_INPUTS)}")
    print(f"dates: {df.date.min().date()} .. {df.date.max().date()}")
    print(f"duplicate (cell_id, date): {int(df.duplicated(['cell_id','date']).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())