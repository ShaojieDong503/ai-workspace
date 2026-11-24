#!/usr/bin/env python3
"""MERRA-2 UTC-6 daily means, one month at a time, on the cell spine.

Rule
----
The files in data/raw/merra2_utc6_ee are Earth Engine exports of the mean of
the *hourly* MERRA-2 images over the UTC-6 day (D 06:00 .. D+1 06:00 UTC).
They are on a 69 x 44 EPSG:4326 grid at 0.4940734 deg; a cell takes the value
of the pixel containing its centre, via merra_pix.npy.

Bands are looked up by description (YYYYMMDD_<VAR>), never by position, so a
re-export that adds or reorders bands cannot silently shift the variables.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio

from utils import ROOT

SRC = ROOT / "data" / "raw" / "merra2_utc6_ee"
INDEX = ROOT / "data" / "interim" / "index" / "merra_pix.npy"

VARIABLES = [
    "BCCMASS", "OCCMASS", "SO4CMASS", "SSCMASS25", "DUCMASS25",
    "BCSMASS", "OCSMASS", "SO4SMASS", "SSSMASS25", "DUSMASS25",
    "BCEXTTAU", "OCEXTTAU", "SUEXTTAU", "DUEXTTAU", "TOTEXTTAU",
    "PBLH", "QLML", "TLML", "ULML", "VLML", "SPEED", "SPEEDMAX",
    "U2M", "V2M", "U10M", "V10M", "U50M", "V50M",
    "U250", "V250", "U500", "V500", "U850", "V850",
]
COLLECTIONS = {"AER": "aer", "FLX": "flx", "SLV": "slv"}


BAND_RE = re.compile(r"^(\d{8})_([A-Z0-9]+)$")


def _band_map(descriptions):
    """description -> band index (1-based), for YYYYMMDD_VAR names.

    Names run 13-17 characters (20150701_PBLH, 20150701_TOTEXTTAU), so a fixed
    length test matches nothing. Match the shape and split on the underscore.
    """
    out = {}
    for i, d in enumerate(descriptions, 1):
        if d and BAND_RE.fullmatch(d):
            out[d] = i
    return out


def extract(year: int, month: int, cell_idx: np.ndarray) -> dict[str, np.ndarray]:
    """Returns {merra_<var>: float32 array of shape (n_days, n_cells)}."""
    pix = np.load(INDEX)[cell_idx]
    out: dict[str, np.ndarray] = {}

    for coll, suffix in COLLECTIONS.items():
        path = SRC / f"source_region_merra_{suffix}_daily_stack_{year}_{month:02d}.tif"
        with rasterio.open(path) as src:
            bmap = _band_map(src.descriptions)
            h, w = src.height, src.width

            for var in VARIABLES:
                bands = sorted((d, i) for d, i in bmap.items()
                               if d.split("_", 1)[1] == var)
                if not bands:
                    continue
                days = [d for d, _ in bands]
                arr = src.read([i for _, i in bands]).astype("float32")   # (n_days, h, w)
                arr = arr.reshape(arr.shape[0], h * w)[:, pix]            # (n_days, n_cells)
                arr[arr == -999] = np.nan
                out[f"merra_{var.lower()}"] = arr
                if len(days) != arr.shape[0]:
                    raise RuntimeError(f"{path.name}: band/day mismatch for {var}")

    missing = [v for v in VARIABLES if f"merra_{v.lower()}" not in out]
    if missing:
        raise RuntimeError(f"MERRA variables not found in {year}-{month:02d}: {missing}")
    return out
