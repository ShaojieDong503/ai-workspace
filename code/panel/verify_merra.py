#!/usr/bin/env python3
"""Independent check of the MERRA-2 extractor.

Recomputes sampled cells straight from the monthly GeoTIFF: find the band by its
description, convert the cell centre to a pixel index with rasterio's own
transform, read that single pixel. The extractor instead uses the precomputed
merra_pix map and a flattened array, so the two paths share only the file.
"""
from __future__ import annotations

import argparse

import numpy as np
import rasterio

from utils import ROOT
from verify_common import cells, check, compare, sample_pairs, summary

SRC = ROOT / "data" / "raw" / "merra2_utc6_ee"
FILES = {"AER": "aer", "FLX": "flx", "SLV": "slv"}


def read_pixel(path, band_desc, lon, lat):
    with rasterio.open(path) as src:
        if band_desc not in src.descriptions:
            return np.nan
        i = src.descriptions.index(band_desc) + 1
        col = int(np.floor((lon - src.transform.c) / src.transform.a))
        row = int(np.floor((lat - src.transform.f) / src.transform.e))
        if not (0 <= row < src.height and 0 <= col < src.width):
            return np.nan
        v = float(src.read(i, window=rasterio.windows.Window(col, row, 1, 1))[0, 0])
    return np.nan if v == -999 else v


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the MERRA extractor")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--year", type=int, default=2015)
    ap.add_argument("--month", type=int, default=7)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    df = sample_pairs(args.cells, args.year, args.month, args.n)
    cs = cells()

    for var, coll in (("pblh", "FLX"), ("totexttau", "AER"), ("u10m", "SLV")):
        col = f"merra_{var}"
        if col not in df:
            check(False, f"{col} missing from the written month")
            continue
        want = np.array([
            read_pixel(SRC / f"source_region_merra_{FILES[coll]}_daily_stack_{args.year}_{args.month:02d}.tif",
                       f"{r.date:%Y%m%d}_{var.upper()}", cs.loc[r.cell_id, "lon"], cs.loc[r.cell_id, "lat"])
            for r in df.itertuples()])
        compare(col, df[col].to_numpy("float64"), want)

    # plausibility
    for col, lo, hi in (("merra_pblh", 0, 6000), ("merra_tlml", 180, 340),
                        ("merra_totexttau", 0, 10)):
        if col in df:
            v = df[col].to_numpy("float64")
            v = v[np.isfinite(v)]
            check(bool(v.min() >= lo and v.max() <= hi),
                  f"{col} within [{lo}, {hi}] (got {v.min():.3g} .. {v.max():.3g})")
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())