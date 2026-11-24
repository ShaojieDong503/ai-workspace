#!/usr/bin/env python3
"""Static index maps: raster pixel -> lattice cell, computed once.

Two different directions are needed and they are not symmetric:

  MERRA (69 x 44, coarse):   cell -> pixel.  Each cell centre falls inside one
      MERRA pixel, so the map is an int32 array of length n_cells.

  MAIAC (1 km, fine):        pixel -> cell.  Each cell contains ~25 MAIAC
      pixels, so the map is one int32 array per tile window, -1 outside the
      spine.  That lets a day-band be reduced with two bincounts instead of a
      per-cell loop or any polygon test.

Both are derived from the geotransform only. Nothing here assumes a raster
shape; the MAIAC tiles are discovered from their own bounds so a later
re-export that adds the 37.5-41 N strip drops in without code changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyproj
import rasterio
from rasterio.windows import from_bounds

ROOT = Path(r"D:\ai-workspace\project")
CELLS = ROOT / "data" / "interim" / "spine" / "cells.parquet"
OUT = ROOT / "data" / "interim" / "index"
MAIAC = ROOT / "data" / "raw" / "maiac_daily_stack"
MERRA_SAMPLE = ROOT / "data" / "raw" / "merra2_utc6_ee" / "source_region_merra_aer_daily_stack_2015_07.tif"

# lattice, from the grid README (fitted, zero residual)
X0, DX = 3688478.20, 4999.827
Y0, DY = 5239381.11, -4998.864
WIDTH = 1065

# the square, W S E N
SQUARE = (-98.2, 38.5, -66.9, 59.1)

TO_3347 = pyproj.Transformer.from_crs(4326, 3347, always_xy=True)


def cell_lookup(cells):
    """(row, col) -> cell_idx, via cell_id = row*1065 + col."""
    ids = cells["cell_id"].to_numpy()
    idx = cells["cell_idx"].to_numpy()
    return dict(zip(ids.tolist(), idx.tolist()))


def merra_index(cells, out: Path) -> dict:
    with rasterio.open(MERRA_SAMPLE) as src:
        t = src.transform
        h, w = src.height, src.width
    # rasterio's src.index() is scalar-only; do the affine maths on the arrays
    cols = np.floor((cells["lon"].to_numpy() - t.c) / t.a).astype("int64")
    rows = np.floor((cells["lat"].to_numpy() - t.f) / t.e).astype("int64")
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    pix = (rows * w + cols).astype("int32")
    pix[~inside] = -1
    np.save(out / "merra_pix.npy", pix)
    print(f"merra: {inside.sum():,}/{len(cells):,} cells inside the raster "
          f"({w}x{h}); outside stored as -1")
    return {"shape": [h, w], "inside": int(inside.sum()), "file": "merra_pix.npy"}


def maiac_index(cells, sample_month: str, out: Path) -> dict:
    lookup = cell_lookup(cells)
    xs = cells["x_3347"].to_numpy(); ys = cells["y_3347"].to_numpy()

    files = sorted(MAIAC.glob(f"source_region_maiac_daily_stack_{sample_month}-*.tif"))
    if not files:
        raise SystemExit(f"no MAIAC tiles for {sample_month}")
    info = {"month_probed": sample_month, "tiles": [], "n_cells_touched": 0}

    for f in files:
        with rasterio.open(f) as src:
            b = src.bounds
            # does this tile reach the square at all?
            if b.right < SQUARE[0] or b.left > SQUARE[2] or b.top < SQUARE[1] or b.bottom > SQUARE[3]:
                continue
            win = from_bounds(max(b.left, SQUARE[0]), max(b.bottom, SQUARE[1]),
                              min(b.right, SQUARE[2]), min(b.top, SQUARE[3]),
                              src.transform).round_offsets().round_lengths()
            t = src.window_transform(win)
            w, h = int(win.width), int(win.height)

        # pixel centres of the window, in lon/lat, then projected once
        lon = t.c + (np.arange(w) + 0.5) * t.a
        lat = t.f + (np.arange(h) + 0.5) * t.e
        LON, LAT = np.meshgrid(lon, lat)
        X, Y = TO_3347.transform(LON.ravel(), LAT.ravel())

        col = np.rint((X - X0) / DX).astype("int64")
        row = np.rint((Y0 - Y) / -DY).astype("int64")
        cid = row * WIDTH + col
        idx = np.fromiter((lookup.get(int(c), -1) for c in cid), dtype="int32", count=cid.size)
        idx = idx.reshape(h, w)

        tag = f.name.split("-")[1].replace(".tif", "") + "_" + f.name.split("-")[2].replace(".tif", "")
        np.save(out / f"maiac_{tag}.npy", idx)
        touched = int((idx >= 0).sum())
        info["tiles"].append({"file": f.name, "tag": tag, "window": [int(win.row_off), int(win.col_off), h, w],
                              "pixels_in_spine": touched})
        info["n_cells_touched"] += touched
        print(f"maiac {tag}: window {h}x{w} at ({int(win.row_off)},{int(win.col_off)}), "
              f"{touched:,} pixels map to a cell")

    (out / "maiac_tiles.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="Build static index maps")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--maiac-month", default="2015_07")
    args = ap.parse_args()

    import pandas as pd
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_parquet(CELLS)

    m = merra_index(cells, out)
    a = maiac_index(cells, args.maiac_month, out)
    (out / "index_manifest.json").write_text(
        json.dumps({"merra": m, "maiac": a}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
