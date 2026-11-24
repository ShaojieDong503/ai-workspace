#!/usr/bin/env python3
"""Station inventory: NAPS + AQS sites, and the lattice cell each one falls in.

Rule
----
A station belongs to the cell whose 5 km footprint contains it. The lattice is
exact (x = 3688478.20 + 4999.827*col, y = 5239381.11 - 4998.864*row), so the
mapping is a rounding, not a spatial join. Stations that land on a water cell,
outside the square, or outside the spine are flagged rather than dropped.

NAPS sites come from the already-verified domain extract; AQS sites are read
from the annual hourly zips with pyarrow, keeping only the six location
columns, because parsing 15 x 0.85 GB CSVs with pandas costs far more.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import pyarrow as pa
import pyarrow.csv as pv

ROOT = Path(r"D:\ai-workspace\project")
CELLS = ROOT / "data" / "interim" / "spine" / "cells.parquet"
NAPS_SITES = ROOT / "data" / "interim" / "domain" / "stations_inside_square.csv"
AQS_DIR = ROOT / "data" / "raw" / "AQS"
OUT = ROOT / "data" / "interim" / "stations"

X0, DX = 3688478.20, 4999.827
Y0, DY = 5239381.11, -4998.864
WIDTH = 1065
MAIAC_SOUTH = 41.0          # MAIAC coverage starts here; CMR-verified tile bounds

TO_3347 = pyproj.Transformer.from_crs(4326, 3347, always_xy=True)


def to_cell(lat, lon, lookup):
    """Round a lon/lat to its lattice cell; returns (cell_id, cell_idx, dist_km, dx, dy)."""
    x, y = TO_3347.transform(np.asarray(lon), np.asarray(lat))
    col = np.rint((x - X0) / DX).astype("int64")
    row = np.rint((Y0 - y) / -DY).astype("int64")
    cid = row * WIDTH + col
    idx = np.array([lookup.get(int(c), -1) for c in cid], dtype="int32")
    # distance from the station to the chosen cell centre, on the projected plane
    cx = X0 + DX * col
    cy = Y0 + DY * row
    dist = np.hypot(x - cx, y - cy) / 1000.0
    return cid, idx, dist


def naps_stations(lookup):
    d = pd.read_csv(NAPS_SITES, encoding="utf-8-sig", dtype={"stn": str})
    d = d.rename(columns={"stn": "site_id", "city": "name", "prov": "province_state"})
    cid, idx, dist = to_cell(d["lat"].values, d["lon"].values, lookup)
    out = pd.DataFrame({
        "station_uid": "NAPS:" + d["site_id"].astype(str),
        "network": "NAPS",
        "name": d["name"],
        "lat": d["lat"].astype("float32"),
        "lon": d["lon"].astype("float32"),
        "cell_id": cid.astype("int32"),
        "cell_idx": idx,
        "province_state": d["province_state"],
        "dist_to_cell_centre_km": dist.astype("float32"),
    })
    return out


def aqs_stations(lookup):
    frames = []
    for z in sorted(AQS_DIR.glob("hourly_88101_*.zip")):
        with zipfile.ZipFile(z) as zf:
            member = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(member) as fh:
                tbl = pv.read_csv(
                    fh,
                    convert_options=pv.ConvertOptions(
                        include_columns=["State Code", "County Code", "Site Num",
                                         "Latitude", "Longitude", "State Name"],
                        column_types={"State Code": pa.string(), "County Code": pa.string(),
                                      "Site Num": pa.string()},
                    ),
                )
        frames.append(tbl.to_pandas().drop_duplicates())
    d = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["State Code", "County Code", "Site Num"])
    d["site_id"] = d["State Code"] + "-" + d["County Code"] + "-" + d["Site Num"]
    d["lat"] = pd.to_numeric(d["Latitude"], errors="coerce")
    d["lon"] = pd.to_numeric(d["Longitude"], errors="coerce")
    d = d.dropna(subset=["lat", "lon"])

    cid, idx, dist = to_cell(d["lat"].values, d["lon"].values, lookup)
    out = pd.DataFrame({
        "station_uid": "AQS:" + d["site_id"].astype(str),
        "network": "AQS",
        "name": d["State Name"].astype(str),
        "lat": d["lat"].astype("float32"),
        "lon": d["lon"].astype("float32"),
        "cell_id": cid.astype("int32"),
        "cell_idx": idx,
        "province_state": d["State Name"].astype(str),
        "dist_to_cell_centre_km": dist.astype("float32"),
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Station -> cell map")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cells = pd.read_parquet(CELLS)
    lookup = dict(zip(cells["cell_id"].tolist(), cells["cell_idx"].tolist()))
    prov = dict(zip(cells["cell_id"].tolist(),
                    cells["province_state"].fillna("water").tolist()))

    naps = naps_stations(lookup)
    aqs = aqs_stations(lookup)
    allst = pd.concat([naps, aqs], ignore_index=True)

    # keep only stations that resolve to a cell in the spine
    before = len(allst)
    allst = allst[allst["cell_idx"] >= 0].copy()
    allst["cell_province"] = allst["cell_id"].map(prov)
    allst["maiac_covered"] = (allst["lat"] >= MAIAC_SOUTH).astype("int8")
    allst["on_water"] = (allst["cell_province"] == "water").astype("int8")

    allst["station_uid"] = allst["station_uid"].astype(str)
    allst = allst.sort_values(["network", "station_uid"]).reset_index(drop=True)
    allst.to_parquet(out / "stations.parquet", index=False, compression="zstd")

    print(f"stations read {before:,} -> inside the spine {len(allst):,}")
    print(allst.groupby(["network", "province_state"]).size().to_string())
    print("\non water cells:", int(allst.on_water.sum()))
    print("south of 41N (no MAIAC):", int((allst.maiac_covered == 0).sum()))
    multi = allst.groupby("cell_id").size()
    multi = multi[multi > 1]
    print(f"cells with more than one station: {len(multi)} "
          f"({int(multi.sum())} stations)")
    print("\ndistance to cell centre, km:")
    print(allst.dist_to_cell_centre_km.describe().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
