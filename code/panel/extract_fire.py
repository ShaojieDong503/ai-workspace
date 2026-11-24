#!/usr/bin/env python3
"""MODIS active fire, one month at a time, on the cell spine.

Rule
----
acq_date + acq_time are UTC; the UTC-6 day is the date six hours earlier.
Only type == 0 (presumed vegetation fire) is kept. Exact duplicates across the
Canada and United States files are dropped. Fires up to 500 km outside the
square still count, so both national files are used in full.

Spatial definitions, both great-circle:
  fire_dist_nearest_km  distance to the nearest detection that day, NaN if none
  fire_count_<r>km      detections within r km, 0 if none
  fire_frp_sum_<r>km    summed FRP within r km, 0 if none

Nearest distance uses a haversine BallTree over the fires, queried once for all
cells. Radius counts and sums use a BallTree over the cells, queried per fire and
accumulated with np.add.at - fires are far fewer than cells, so this is the cheap
direction. Both are exact; no FFT approximation is used.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from utils import ROOT, days_in_month
import datetime as _dt

SRC = ROOT / "data" / "MODIS_fire"
CELLS = ROOT / "data" / "interim" / "spine" / "cells.parquet"
RADII_KM = (25, 50, 100, 250, 500)
EARTH_KM = 6371.0088


def _load_year(year: int, years: tuple[int, ...] | None = None) -> pd.DataFrame:
    frames, missing = [], []
    for y in (years or (year,)):
        for name in (f"modis_{y}_Canada.csv", f"modis_{y}_United_States.csv"):
            p = SRC / name
            if p.exists():
                frames.append(pd.read_csv(p, low_memory=False))
            else:
                missing.append(name)
    if missing:
        # the last December has no next-year file, so its 00:00-06:00 UTC
        # detections on 1 January are absent; say so instead of failing quietly
        print(f"  WARNING: fire files missing, that window cannot be covered: {missing}")
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = d.drop_duplicates()                       # exact duplicates across countries
    return d


def extract(year: int, month: int, cell_idx: np.ndarray) -> dict[str, np.ndarray]:
    from sklearn.neighbors import BallTree

    cells = pd.read_parquet(CELLS).set_index("cell_idx").loc[cell_idx]
    cell_rad = np.radians(cells[["lat", "lon"]].to_numpy(dtype="float64"))

    days = days_in_month(year, month)
    n, m = len(days), len(cell_idx)
    out = {f"fire_count_{r}km": np.zeros((n, m), "int16") for r in RADII_KM}
    out.update({f"fire_frp_sum_{r}km": np.zeros((n, m), "float32") for r in RADII_KM})
    out["fire_dist_nearest_km"] = np.full((n, m), np.nan, "float32")

    # 31 December needs the next year's file: the UTC-6 window runs to
    # 1 January 06:00 UTC and the Terra night overpass sits inside it
    raw = _load_year(year, (year, year + 1) if month == 12 else (year,))
    if raw.empty:
        return out

    dropped = raw["type"].eq(0) == False          # noqa: E712
    type_counts = raw.loc[dropped, "type"].value_counts().to_dict()
    fires = raw[~dropped].copy()

    ts = pd.to_datetime(fires["acq_date"].astype(str) + " " +
                        fires["acq_time"].astype(str).str.zfill(4),
                        format="%Y-%m-%d %H%M", errors="coerce")
    fires["day"] = (ts - pd.Timedelta(hours=6)).dt.date
    fires = fires.dropna(subset=["day", "latitude", "longitude"])

    cell_tree = BallTree(cell_rad, metric="haversine")

    for i, day in enumerate(days):
        f = fires[fires["day"] == day]
        if f.empty:
            continue                              # counts stay 0, distance stays NaN
        fr = np.radians(f[["latitude", "longitude"]].to_numpy(dtype="float64"))
        frp = pd.to_numeric(f["frp"], errors="coerce").fillna(0.0).to_numpy("float64")

        dist, _ = BallTree(fr, metric="haversine").query(cell_rad, k=1)
        out["fire_dist_nearest_km"][i] = (dist[:, 0] * EARTH_KM).astype("float32")

        for r in RADII_KM:
            idx = cell_tree.query_radius(fr, r=float(r) / EARTH_KM)
            flat = np.concatenate(idx) if len(idx) else np.empty(0, "int64")
            # counts are per CELL, accumulated from the per-fire neighbour lists
            out[f"fire_count_{r}km"][i] = np.bincount(flat, minlength=m).astype("int16")
            counts = np.fromiter((len(x) for x in idx), dtype="int64", count=len(idx))
            if flat.size:
                vals = np.repeat(frp, counts)
                acc = np.zeros(m, "float64")
                np.add.at(acc, flat, vals)
                out[f"fire_frp_sum_{r}km"][i] = acc.astype("float32")

    out["_dropped_types"] = np.array([type_counts], dtype=object)
    return out
