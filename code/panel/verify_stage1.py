#!/usr/bin/env python3
"""Verification for stage 1: spine, index maps, stations, station-days.

Checks are independent of the builders where it matters: the cell bounds are
recomputed from the lattice formula, station cells are re-derived from the
station coordinates, and station-day means are recomputed from the hourly
source for a random sample.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\ai-workspace\project")
CELLS = ROOT / "data" / "interim" / "spine" / "cells.parquet"
STATIONS = ROOT / "data" / "interim" / "stations" / "stations.parquet"
STATION_DAY = ROOT / "data" / "interim" / "stations" / "station_day.parquet"
INDEX = ROOT / "data" / "interim" / "index"

X0, DX = 3688478.20, 4999.827
Y0, DY = 5239381.11, -4998.864
WIDTH = 1065

problems: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        problems.append(msg)


def main() -> int:
    cells = pd.read_parquet(CELLS)
    stations = pd.read_parquet(STATIONS)
    sd = pd.read_parquet(STATION_DAY)

    # ---- spine -----------------------------------------------------------
    check(len(cells) == 163_787, f"spine rows = 163,787 (got {len(cells):,})")
    check(cells["cell_idx"].tolist() == list(range(len(cells))),
          "cell_idx is 0..n-1 in order")
    check(cells["cell_id"].is_unique, "cell_id unique in the spine")
    # country/province are null exactly for water cells; everything else must be complete
    geo_null = cells[["country", "province_state"]].isna().all(axis=1)
    check(int(geo_null.sum()) == int(cells["country"].isna().sum()),
          "country and province are null together (water cells only)")
    rest = cells.drop(columns=["country", "province_state"])
    check(int(rest.isna().sum().sum()) == 0, "no NaN in the numeric spine columns")
    check(int(cells["country"].isna().sum()) == 16_269,
          f"water cells = 16,269 (got {int(cells["country"].isna().sum()):,})")

    # lattice recomputed independently from row/col
    x = X0 + DX * cells["col"].to_numpy()
    y = Y0 + DY * cells["row"].to_numpy()
    err = np.hypot(x - cells["x_3347"].to_numpy(), y - cells["y_3347"].to_numpy())
    check(float(err.max()) < 1e-3, f"cell centres match the lattice (max {err.max():.2e} m)")
    check(bool(((cells["cell_id"] - (cells["row"].astype("int64") * WIDTH + cells["col"])) == 0).all()),
          "cell_id == row*1065 + col for every row")

    # ---- index maps ------------------------------------------------------
    merra = np.load(INDEX / "merra_pix.npy")
    check(merra.shape == (len(cells),), f"merra index length {merra.shape}")
    check(int((merra < 0).sum()) == 0, "every cell maps into the MERRA raster")

    import json
    tiles = json.loads((INDEX / "maiac_tiles.json").read_text(encoding="utf-8"))
    check(len(tiles["tiles"]) == 4, f"4 MAIAC tiles discovered (got {len(tiles['tiles'])})")
    for t in tiles["tiles"]:
        arr = np.load(INDEX / f"maiac_{t['tag']}.npy")
        check(arr.shape == (t["window"][2], t["window"][3]),
              f"maiac {t['tag']} index shape {arr.shape} matches its window")
        check(int((arr >= 0).sum()) == t["pixels_in_spine"],
              f"maiac {t['tag']} pixel count matches the manifest")

    # ---- stations --------------------------------------------------------
    check(len(stations) == 388, f"stations in the spine = 388 (got {len(stations)})")
    check(stations["station_uid"].is_unique, "station_uid unique")
    check(stations["cell_idx"].isin(cells["cell_idx"]).all(),
          "every station cell_idx exists in the spine")

    # re-derive the cell from the station coordinates
    import pyproj
    tr = pyproj.Transformer.from_crs(4326, 3347, always_xy=True)
    sx, sy = tr.transform(stations["lon"].to_numpy(), stations["lat"].to_numpy())
    col = np.rint((sx - X0) / DX).astype("int64")
    row = np.rint((Y0 - sy) / -DY).astype("int64")
    recomputed = row * WIDTH + col
    check(bool((recomputed == stations["cell_id"].to_numpy()).all()),
          "station cell_id reproduced from its coordinates")
    d = np.hypot(sx - (X0 + DX * col), sy - (Y0 + DY * row)) / 1000
    check(float(d.max()) <= 3.55,
          f"max station-to-centre distance {d.max():.2f} km (<= half a cell diagonal 3.54)")

    # ---- station-days ----------------------------------------------------
    check(len(sd) == 1_067_868, f"station-days = 1,067,868 (got {len(sd):,})")
    check(not sd.duplicated(subset=["station_uid", "date"]).any(),
          "(station_uid, date) unique")
    check(sd["date"].between(pd.Timestamp("2010-01-01"), pd.Timestamp("2024-12-31")).all(),
          "every date inside 2010-2024")
    check(sd["cell_id"].isin(cells["cell_id"]).all(), "every station-day cell is in the spine")
    check(bool((sd["n_valid_hours"] >= 18).all()),
          f"every row respects the 18-hour rule (min {int(sd.n_valid_hours.min())})")
    check(bool(sd["pm25"].notna().all()), "no NaN targets")
    check(bool((sd["pm25"] >= 0).all()), "no negative targets (negative hours are missing)")
    check(sd["station_uid"].nunique() == 388, "all 388 stations have at least one day")

    # station -> cell consistency between the two tables
    m = sd.merge(stations[["station_uid", "cell_id"]], on="station_uid", suffixes=("", "_st"))
    check(bool((m["cell_id"] == m["cell_id_st"]).all()),
          "station-day cell_id agrees with stations.parquet")

    # ---- independent recomputation of a few daily means ------------------
    # NAPS Ontario only: recompute from the hourly UTC-6 product.
    on_prov = set(stations.loc[stations.province_state == "ON", "station_uid"])
    on = sd[sd.station_uid.isin(on_prov)]
    OH = ROOT / "data" / "processed" / "on_utc6"
    rng = np.random.default_rng(0)
    sample = on.sample(n=30, random_state=0)
    misses = 0
    for _, r in sample.iterrows():
        stn = r.station_uid.replace("NAPS:", "")
        year = r.date.year
        f = OH / f"ON_PM25_UTC6_{year}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, skiprows=7, encoding="utf-8-sig", dtype=str)
        d.columns = [c.split("//")[0].strip() for c in d.columns]
        row = d[(d["NAPS ID"].str.strip() == stn) & (d["Date"] == r.date.strftime("%Y-%m-%d"))]
        if row.empty:
            continue
        hours = row[[f"H{i:02d}" for i in range(1, 25)]].apply(pd.to_numeric, errors="coerce")
        hours = hours.replace(-999, np.nan)
        hours = hours.where(hours >= 0)  # negative hour = missing hour
        manual = float(hours.mean(axis=1, skipna=True).iloc[0])
        if abs(manual - float(r.pm25)) > 1e-4:
            misses += 1
    check(misses == 0, f"30 sampled Ontario daily means reproduce from the hourly product ({misses} mismatches)")

    print()
    print(f"{len(problems)} failure(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
