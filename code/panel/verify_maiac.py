#!/usr/bin/env python3
"""Independent check of the MAIAC extractor.

The extractor reduces each day-band with bincounts over a precomputed
pixel->cell map. This script does the opposite: for a sampled cell it builds the
cell's 5 km box in EPSG:3347, projects every pixel centre in that box back to
3347 to decide membership, and averages the values itself. No shared code path.

Also reports the two properties that would expose a monthly-composite defect:
the number of all-zero ("empty") days per year and the median number of distinct
values per cell per month.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.windows import from_bounds

from utils import DX, DY, ROOT, X0, Y0
from verify_common import BASE, cells, check, compare, sample_pairs, summary

SRC = ROOT / "data" / "raw" / "maiac_daily_stack"
META = ROOT / "data" / "interim" / "index" / "maiac_tiles.json"
TO_WGS = pyproj.Transformer.from_crs(3347, 4326, always_xy=True)
TO_3347 = pyproj.Transformer.from_crs(4326, 3347, always_xy=True)


def recompute(cs, cell_id, day):
    """Every MAIAC column for one cell-day, recomputed from the raster.

    Accumulates across all tiles so seam cells are right, and applies the same
    five-bands-all-zero empty-day rule as the builder.
    """
    cx, cy = cs.loc[cell_id, "x_3347"], cs.loc[cell_id, "y_3347"]
    cell_col = int(round((cx - X0) / DX))
    cell_row = int(round((Y0 - cy) / -DY))
    hx, hy = DX / 2, abs(DY) / 2
    lon0, lat0 = TO_WGS.transform(cx - hx, cy + hy)
    lon1, lat1 = TO_WGS.transform(cx + hx, cy - hy)

    bands = ("AOD055", "AOD047", "INJECTION_HEIGHT", "SMOKE_AOD055", "SMOKE_MASK")
    sums = {b: 0.0 for b in bands}
    cnts = {b: 0 for b in bands}
    total = 0
    cent = {}

    meta = json.loads(META.read_text(encoding="utf-8"))
    for t in meta["tiles"]:
        path = SRC / t["file"].replace("2015_07", f"{day:%Y_%m}")
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            desc = {d: i + 1 for i, d in enumerate(src.descriptions) if d}
            want = {b: f"{day:%Y%m%d}_{b}" for b in bands}
            if not all(v in desc for v in want.values()):
                continue
            left, right = max(lon0, src.bounds.left), min(lon1, src.bounds.right)
            top, bottom = min(lat0, src.bounds.top), max(lat1, src.bounds.bottom)
            if right <= left or top <= bottom:
                continue

            w0 = from_bounds(left, bottom, right, top, src.transform)
            w0 = w0.round_offsets().round_lengths()
            c0 = max(0, int(w0.col_off) - 4)
            r0 = max(0, int(w0.row_off) - 4)
            c1 = min(src.width, int(w0.col_off + w0.width) + 4)
            r1 = min(src.height, int(w0.row_off + w0.height) + 4)
            win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
            arr = {b: src.read(desc[want[b]], window=win).astype("float32") for b in bands}

            if all(np.all(v == 0) for v in arr.values()):
                continue                      # empty tile-day

            tr = src.window_transform(rasterio.windows.Window(
                t["window"][1], t["window"][0], t["window"][3], t["window"][2]))
            yy, xx = np.mgrid[0:arr[bands[0]].shape[0], 0:arr[bands[0]].shape[1]]
            gx = (c0 - t["window"][1]) + xx
            gy = (r0 - t["window"][0]) + yy
            px, py = TO_3347.transform((tr.c + (gx + 0.5) * tr.a).ravel(),
                                        (tr.f + (gy + 0.5) * tr.e).ravel())
            col = np.rint((px - X0) / DX).astype("int64")
            row = np.rint((Y0 - py) / -DY).astype("int64")
            inside = (col == cell_col) & (row == cell_row)
            if not inside.any():
                continue
            total += int(inside.sum())
            for b in bands:
                v = arr[b].ravel()[inside]
                v = v[np.isfinite(v)]
                sums[b] += float(v.sum())
                cnts[b] += int(v.size)
            # centroid = the pixel CONTAINING the cell centre, not the first
            # pixel of the footprint
            lon_c, lat_c = TO_WGS.transform(cx, cy)
            col_c = int(np.floor((lon_c - tr.c) / tr.a))
            row_c = int(np.floor((lat_c - tr.f) / tr.e))
            # row_c/col_c are tile-relative; r0/c0 are raster-absolute
            rr_c = (row_c + t["window"][0]) - r0
            cc_c = (col_c + t["window"][1]) - c0
            if 0 <= rr_c < arr[bands[0]].shape[0] and 0 <= cc_c < arr[bands[0]].shape[1]:
                for b, key in (("AOD055", "aod055_centroid"), ("AOD047", "aod047_centroid")):
                    if key not in cent:
                        cent[key] = float(arr[b][rr_c, cc_c])

    out = {}
    for b, key in (("AOD055", "aod055"), ("AOD047", "aod047"),
                   ("INJECTION_HEIGHT", "injection_height"),
                   ("SMOKE_AOD055", "smoke_aod055"), ("SMOKE_MASK", "smoke_mask_frac")):
        out[key] = sums[b] / cnts[b] if cnts[b] else np.nan
    out["aod055_npix"] = cnts["AOD055"]
    out["maiac_npix_total"] = total
    out["aod055_centroid"] = cent.get("aod055_centroid", np.nan)
    out["aod047_centroid"] = cent.get("aod047_centroid", np.nan)
    out["maiac_covered"] = 1 if total else 0
    return out


def empty_days(kind: str) -> pd.DataFrame:
    """Days on which essentially every covered cell is NaN -> a dead tile-day."""
    rows = []
    for f in sorted((BASE / kind).glob("base_*.parquet")):
        d = pd.read_parquet(f, columns=["date", "aod055", "maiac_covered"])
        cov = d["maiac_covered"] == 1
        if not cov.any():
            continue
        g = d[cov].groupby("date")["aod055"].apply(lambda s: s.isna().mean())
        rows.append(pd.DataFrame({"date": g.index, "nan_share": g.values}))
    if not rows:
        return pd.DataFrame(columns=["date", "nan_share"])
    out = pd.concat(rows)
    out["year"] = pd.to_datetime(out["date"]).dt.year
    return out


def distinct_median(kind: str, year: int, month: int) -> float:
    d = pd.read_parquet(BASE / kind / f"base_{kind}_{year}_{month:02d}.parquet",
                        columns=["cell_id", "date", "aod055"])
    d = d[d["aod055"].notna()]
    if d.empty:
        return np.nan
    return float(d.groupby("cell_id")["aod055"].nunique().median())


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the MAIAC extractor")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--year", type=int, default=2015)
    ap.add_argument("--month", type=int, default=7)
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()

    df = sample_pairs(args.cells, args.year, args.month, args.n)
    cs = cells()
    want = [recompute(cs, r.cell_id, pd.Timestamp(r.date)) for r in df.itertuples()]

    for col in ("aod055", "aod047", "injection_height", "smoke_aod055",
                "smoke_mask_frac"):
        got = df[col].to_numpy("float64")
        exp = np.array([w[col] for w in want], "float64")
        compare(col, got, exp, tol=1e-5)

    for col in ("aod055_npix", "maiac_npix_total", "maiac_covered"):
        got = df[col].to_numpy("float64")
        exp = np.array([w[col] for w in want], "float64")
        compare(col, got, exp, tol=0.0)

    for col in ("aod055_centroid", "aod047_centroid"):
        got = df[col].to_numpy("float64")
        exp = np.array([w[col] for w in want], "float64")
        compare(col, got, exp, tol=1e-5)

    med = distinct_median(args.cells, args.year, args.month)
    check(bool(med > 1), f"median distinct aod055 per cell in {args.year}-{args.month:02d} = {med:.0f} (>1)")

    emp = empty_days(args.cells)
    if not emp.empty:
        per_year = emp.groupby("year").agg(days=("date", "size"),
                                           dead=("nan_share", lambda s: int((s > 0.999).sum())))
        print("\nall-NaN days per year (dead tile-days):")
        print(per_year.to_string())

    v = df["aod055"].to_numpy("float64")
    v = v[np.isfinite(v)]
    if v.size:
        check(bool(v.min() >= -0.1 and v.max() <= 5.0),
              f"aod055 within [-0.1, 5] (got {v.min():.3g} .. {v.max():.3g})")
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())