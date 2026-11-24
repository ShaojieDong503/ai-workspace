#!/usr/bin/env python3
"""Verify MERRA-2 gap-fill exports against the study square.

Run this after downloading the Earth Engine exports. It answers three
questions per product, year and variable:

  1. Are any days missing from the stacks?
  2. What fraction of pixels inside the study square is empty?
  3. Do the new files sit on the same pixel lattice as the existing archives?

Emptiness is read as NaN, -999, or <= 0, but the <= 0 rule is applied only to
the aerosol mass and extinction bands. Winds, temperature and humidity may be
legitimately negative or zero, so for those only NaN and -999 count as empty.

Usage
-----
    python check_merra_exports.py --dir "D:\\path\\to\\SourceRegion_MERRA_2010_2024_v2"
    python check_merra_exports.py --dir ... --years 2010 2024 --csv report.csv
    python check_merra_exports.py --dir ... --reference "D:\\...\\CAN_SRC_MERRA_AER_2013.tif"
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

# Study square, west/south/east/north.
SQUARE = (-98.2, 38.5, -66.9, 59.1)

# The lattice every export must sit on, read from
# CAN_SRC_MERRA_AER_2013.tif (affine: xscale, xskew, xorigin, yskew, yscale, yorigin).
REF_TRANSFORM = (0.4940734062657368, 0.0, -136.8583335356091,
                 0.0, -0.4940734062657368, 69.66435028346889)
TRANSFORM_TOL = 1e-9

# Bands for which a value of zero (or below) means "no data" rather than a
# real measurement. Winds/temperature/humidity are deliberately excluded.
POSITIVE_ONLY = ("SMASS", "CMASS", "EXTTAU")

BAND_RE = re.compile(r"^(\d{8})_([A-Z0-9]+)$")


def product_of(path: Path) -> str | None:
    """Guess the collection from the file name (works for old and new names)."""
    n = path.name.lower()
    for key, tag in (("aer", "AER"), ("flx", "FLX"), ("slv", "SLV")):
        if key in n:
            return tag
    return None


def is_positive_only(var: str) -> bool:
    return any(k in var for k in POSITIVE_ONLY)


def square_window(src, bbox=SQUARE):
    """Window covering the study square, clipped to the raster."""
    left, bottom, right, top = bbox
    b = src.bounds
    left, right = max(left, b.left), min(right, b.right)
    bottom, top = max(bottom, b.bottom), min(top, b.top)
    if right <= left or top <= bottom:
        return None
    return from_bounds(left, bottom, right, top, src.transform)


def transform_delta(src) -> float:
    """Largest absolute difference between this raster's affine and the reference."""
    t = src.transform
    vals = (t.a, t.b, t.c, t.d, t.e, t.f)
    return float(max(abs(v - r) for v, r in zip(vals, REF_TRANSFORM)))


def main() -> int:
    ap = argparse.ArgumentParser(description="QA for MERRA-2 gap-fill exports")
    ap.add_argument("--dir", required=True, help="folder holding the exported GeoTIFFs")
    ap.add_argument("--years", type=int, nargs=2, metavar=("FROM", "TO"),
                    help="expected year range (default: whatever the folder contains)")
    ap.add_argument("--reference", help="existing GeoTIFF to compare the grid against")
    ap.add_argument("--csv", default="merra_export_check.csv", help="per-variable report")
    args = ap.parse_args()

    root = Path(args.dir)
    files = sorted(p for p in root.rglob("*.tif"))
    if not files:
        print(f"no .tif files under {root}")
        return 2
    print(f"scanning {len(files)} files under {root}\n")

    # product -> variable -> set of dates, and product -> set of dates
    have: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    grid: dict[str, float] = {}
    unparsed: list[tuple[str, str]] = []

    for path in files:
        prod = product_of(path)
        if prod is None:
            unparsed.append((path.name, "product not in filename"))
            continue
        with rasterio.open(path) as src:
            grid[path.name] = transform_delta(src)
            desc = src.descriptions
        if not any(desc):
            unparsed.append((path.name, "no band descriptions"))
            continue
        for name in desc:
            if not name:
                continue
            m = BAND_RE.match(name)
            if not m:
                continue
            day, var = m.group(1), m.group(2)
            have[prod][var].add(date(int(day[:4]), int(day[4:6]), int(day[6:])))
            have[prod]["__all__"].add(date(int(day[:4]), int(day[4:6]), int(day[6:])))

    # year range to check
    all_days = set().union(*(have[p]["__all__"] for p in have)) if have else set()
    if not all_days:
        print("no parseable YYYYMMDD_VAR band names found")
        return 2
    y0, y1 = (args.years if args.years else (min(all_days).year, max(all_days).year))
    expected = [date(y0, 1, 1) + timedelta(days=i)
                for i in range((date(y1, 12, 31) - date(y0, 1, 1)).days + 1)]
    print(f"expected days {expected[0]} .. {expected[-1]}  ({len(expected)} days)\n")

    # ---- 1. missing days -------------------------------------------------
    print("== missing days ==")
    for prod in sorted(have):
        missing = [d for d in expected if d not in have[prod]["__all__"]]
        print(f"  {prod}: {len(missing)} missing", end="")
        if missing:
            head = ", ".join(d.isoformat() for d in missing[:8])
            print(f"  e.g. {head}{' ...' if len(missing) > 8 else ''}")
        else:
            print()
    print()

    # ---- 2. empty fraction inside the square ------------------------------
    print("== empty fraction inside the square ==")
    rows = []
    by_var: dict[tuple[str, str], list[float]] = defaultdict(list)
    for path in files:
        prod = product_of(path)
        if prod is None:
            continue
        with rasterio.open(path) as src:
            win = square_window(src)
            if win is None:
                continue
            desc = src.descriptions
            for var in sorted({n.split("_", 1)[1] for n in desc if n and BAND_RE.match(n)}):
                idx = [i + 1 for i, n in enumerate(desc) if n and n.endswith("_" + var)]
                if not idx:
                    continue
                arr = src.read(idx, window=win).astype("float32")
                empty = ~np.isfinite(arr) | (arr == -999)
                if is_positive_only(var):
                    empty |= arr <= 0
                # Measure against the WHOLE square, not just the part the raster
                # happens to cover: area outside the file is empty too.
                px, py = abs(src.transform.a), abs(src.transform.e)
                expected_px = ((SQUARE[2] - SQUARE[0]) / px) * ((SQUARE[3] - SQUARE[1]) / py)
                # per-band valid count, then averaged: arr holds one band per day
                valid_per_band = (~empty).sum(axis=(1, 2))
                frac = float(1.0 - valid_per_band.mean() / expected_px)
                covered = (win.width * win.height) / expected_px
                years = {desc[i - 1][:4] for i in idx}
                for yr in years:
                    rows.append({"product": prod, "year": yr, "variable": var,
                                 "file": path.name,
                                 "empty_fraction": round(frac, 6),
                                 "coverage": round(covered, 6)})
                    by_var[(prod, var)].append(frac)

    for (prod, var), fracs in sorted(by_var.items()):
        print(f"  {prod:3s} {var:12s} mean {np.mean(fracs):6.1%}  max {np.max(fracs):6.1%}"
              f"  ({len(fracs)} files)")

    # ---- 3. grid alignment ------------------------------------------------
    print("\n== grid alignment versus reference ==")
    worst = max(grid.items(), key=lambda kv: kv[1]) if grid else ("-", 0)
    n_ok = sum(1 for v in grid.values() if v <= TRANSFORM_TOL)
    print(f"  {n_ok}/{len(grid)} files match the reference transform exactly")
    print(f"  worst deviation: {worst[1]:.3e} deg  ({worst[0]})")
    if args.reference:
        with rasterio.open(args.reference) as ref:
            print(f"  reference file: {Path(args.reference).name}  {ref.width}x{ref.height}")
    else:
        print("  (reference default is CAN_SRC_MERRA_AER_2013.tif; pass --reference to override)")

    if unparsed:
        print("\n== files skipped ==")
        for name, why in unparsed[:10]:
            print(f"  {name}: {why}")

    # ---- detail csv -------------------------------------------------------
    out = Path(args.csv)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["product", "year", "variable", "file",
                                          "empty_fraction", "coverage"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
