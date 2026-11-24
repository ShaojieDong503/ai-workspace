#!/usr/bin/env python3
"""MAIAC AOD, one month at a time, reduced to the cell spine.

Rule
----
Every MAIAC pixel centre in the square is assigned to a lattice cell by
rounding in EPSG:3347 (done once in index_maps.py). A day-band is then reduced
with two bincounts - sum of values and count of pixels - which gives the
footprint mean without a per-cell loop and without any polygon test.

I/O pattern
-----------
The tiles are pixel-interleaved LZW GeoTIFFs in 256 x 256 blocks. Reading one
band at a time costs ~2.2 s per window and the month needs 620 of those; reading
all 155 bands of a 256-row chunk costs ~0.9 s, so the month takes about half a
minute instead of twenty. The chunk loop below is the only reason that is
affordable: 155 x 256 x 2117 x 4 B = 336 MB per chunk, well inside the worker
budget.

The export writes days with no granule as constant 0 in all five bands, not as
NoData. A tile-day whose five bands are all exactly 0 therefore contributes
nothing.

Coverage stops at 41 N. Cells south of it get NaN and maiac_covered = 0; the
tiles are discovered from their bounds, so a later strip export drops in with no
code change.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from utils import ROOT, days_in_month

SRC = ROOT / "data" / "raw" / "maiac_daily_stack"
IDX = ROOT / "data" / "interim" / "index"
CELLS = ROOT / "data" / "interim" / "spine" / "cells.parquet"

BANDS = {
    "aod055": "AOD055",
    "aod047": "AOD047",
    "injection_height": "INJECTION_HEIGHT",
    "smoke_aod055": "SMOKE_AOD055",
    "smoke_mask_frac": "SMOKE_MASK",
}
CENTROID_CACHE = IDX / "maiac_centroid.npz"
CHUNK_ROWS = 256


def _centroid_map(cells: pd.DataFrame) -> dict[str, np.ndarray]:
    """cell_idx, flat window pixel of the pixel containing each cell centre."""
    if CENTROID_CACHE.exists():
        z = np.load(CENTROID_CACHE)
        return {k: z[k] for k in z.files}

    meta = json.loads((IDX / "maiac_tiles.json").read_text(encoding="utf-8"))
    out = {}
    for t in meta["tiles"]:
        with rasterio.open(SRC / t["file"]) as src:
            r_off, c_off, h, w = t["window"]
            tr = src.window_transform(Window(c_off, r_off, w, h))
        col = np.floor((cells["lon"].to_numpy() - tr.c) / tr.a).astype("int64")
        row = np.floor((cells["lat"].to_numpy() - tr.f) / tr.e).astype("int64")
        inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        out[t["tag"]] = np.column_stack([cells["cell_idx"].to_numpy()[inside],
                                         row[inside] * w + col[inside]])
    np.savez_compressed(CENTROID_CACHE, **out)
    return out


def extract(year: int, month: int, cell_idx: np.ndarray) -> dict[str, np.ndarray]:
    cells = pd.read_parquet(CELLS)
    meta = json.loads((IDX / "maiac_tiles.json").read_text(encoding="utf-8"))
    cent = _centroid_map(cells)

    days = days_in_month(year, month)
    n, m = len(days), len(cell_idx)
    n_cells = len(cells)
    pos = np.full(n_cells, -1, "int64")
    pos[cell_idx] = np.arange(m)

    sums = {k: np.zeros((n, m), "float64") for k in BANDS}
    cnts = {k: np.zeros((n, m), "float64") for k in BANDS}
    npix_tot = np.zeros(m, "float64")
    cent_vals = {"aod055_centroid": np.full((n, m), np.nan, "float32"),
                 "aod047_centroid": np.full((n, m), np.nan, "float32")}
    empty_days = 0

    for t in meta["tiles"]:
        idx = np.load(IDX / f"maiac_{t['tag']}.npy")           # (h, w) cell_idx or -1
        h, w = idx.shape
        r_off, c_off = t["window"][0], t["window"][1]

        # static per-pixel target position, flattened once
        target = np.where(idx.ravel() >= 0, pos[np.maximum(idx.ravel(), 0)], -1)
        npix_tot += np.bincount(target[target >= 0], minlength=m)

        path = SRC / t["file"].replace("2015_07", f"{year}_{month:02d}")
        if not path.exists():
            continue

        # centroid pixels of this tile, grouped by row for the chunk loop
        c_pix = cent[t["tag"]][:, 1]
        c_row, c_col = c_pix // w, c_pix % w
        c_loc = pos[cent[t["tag"]][:, 0]]
        c_ok = c_loc >= 0

        with rasterio.open(path) as src:
            desc = {d: i + 1 for i, d in enumerate(src.descriptions) if d}
            wanted = {day: {k: f"{day:%Y%m%d}_{v}" for k, v in BANDS.items()}
                      for day in days}

            for r0 in range(0, h, CHUNK_ROWS):
                r1 = min(r0 + CHUNK_ROWS, h)
                block = src.read(window=Window(c_off, r_off + r0, w, r1 - r0))
                rows = r1 - r0
                tgt = target[r0 * w:r1 * w]
                usable = tgt >= 0
                tgt_ok = tgt[usable]

                in_chunk = c_ok & (c_row >= r0) & (c_row < r1)
                if in_chunk.any():
                    cr = c_row[in_chunk] - r0
                    cc = c_col[in_chunk]
                    cl = c_loc[in_chunk]

                for i, day in enumerate(days):
                    want = wanted[day]
                    if any(v not in desc for v in want.values()):
                        continue
                    b = {k: block[desc[v] - 1] for k, v in want.items()}

                    # empty-day trap: the export wrote an all-zero image
                    if all(np.all(v == 0) for v in b.values()):
                        if r0 == 0:
                            empty_days += 1
                        continue

                    for k, arr in b.items():
                        v = arr.ravel()[usable]
                        ok = np.isfinite(v)
                        if ok.any():
                            sums[k][i] += np.bincount(tgt_ok[ok], weights=v[ok], minlength=m)
                            cnts[k][i] += np.bincount(tgt_ok[ok], minlength=m)

                    if in_chunk.any():
                        a055 = b["aod055"]
                        a047 = b["aod047"]
                        cent_vals["aod055_centroid"][i, cl] = a055[cr, cc]
                        cent_vals["aod047_centroid"][i, cl] = a047[cr, cc]

    out = {}
    for k in BANDS:
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = np.where(cnts[k] > 0, sums[k] / cnts[k], np.nan).astype("float32")
    out["aod055_npix"] = np.where(cnts["aod055"] > 0, cnts["aod055"], 0).astype("int16")
    out["maiac_npix_total"] = np.minimum(npix_tot, 32767).astype("int16")
    out.update(cent_vals)
    out["maiac_covered"] = (npix_tot > 0).astype("int8")
    out["_empty_tile_days"] = np.array([empty_days])
    return out