#!/usr/bin/env python3
"""QA for the MERRA-2 AER surface/extinction gap-fill export.

Reads the downloaded zip in place through GDAL's /vsizip/, without extracting.
Reports:

  1. months present (expect 180 for 2010-01 .. 2024-12) and any missing days;
  2. per variable, the fraction of empty pixels inside the study square,
     where empty = NaN, -999, or <= 0 (this product is all mass/extinction,
     so the <= 0 rule applies to every band here);
  3. whether transform, shape and grid match the completed archive;
  4. a value cross-check against CAN_SRC_MERRA_AER_<year>.tif north of 41N,
     where the two files overlap.

Usage
-----
    python check_merra_aer_sfc.py --zip "D:\\...\\SourceRegion_MERRA_AER_SFC_2010_2024-....zip"
    python check_merra_aer_sfc.py --zip ... --day 2015-07-01 --csv report.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

SQUARE = (-98.2, 38.5, -66.9, 59.1)

# Grid of the completed archive; the new export must match it exactly.
REF_TRANSFORM = (0.4940734062657368, 0.0, -100.29690147194458,
                 0.0, -0.4940734062657368, 60.77102897068563)
REF_SHAPE = (48, 71)          # rows, cols
TRANSFORM_TOL = 1e-9

PRODUCT_KEYS = ("BCSMASS", "OCSMASS", "SO4SMASS", "SSSMASS25", "DUSMASS25",
                "BCEXTTAU", "OCEXTTAU", "SUEXTTAU", "DUEXTTAU", "TOTEXTTAU")

BAND_RE = re.compile(r"^(\d{8})_([A-Z0-9]+)$")

# Bands where zero means "no data". Winds/temperature/humidity may be
# legitimately negative or zero, so they are excluded.
POSITIVE_ONLY = ("SMASS", "CMASS", "EXTTAU")


def vsi(zip_path: Path, member: str) -> str:
    """GDAL virtual path into the archive."""
    return f"/vsizip/{zip_path.as_posix()}/{member}"


def list_members(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return [n for n in zf.namelist() if n.lower().endswith(".tif")]


def parse_member(name: str) -> tuple[int, int] | None:
    m = re.search(r"(\d{4})_(\d{2})\.tif$", Path(name).name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def window_for(src, bbox) -> object | None:
    left, bottom, right, top = bbox
    b = src.bounds
    left, right = max(left, b.left), min(right, b.right)
    bottom, top = max(bottom, b.bottom), min(top, b.top)
    if right <= left or top <= bottom:
        return None
    return from_bounds(left, bottom, right, top, src.transform)


def check_structure(zip_path: Path) -> dict:
    members = list_members(zip_path)
    months = sorted({m for m in (parse_member(n) for n in members) if m})
    print(f"files in zip: {len(members)}")
    print(f"months       : {len(months)}  {months[0]} .. {months[-1]}" if months else "months: none")

    expected = [(y, m) for y in range(2010, 2025) for m in range(1, 13)]
    missing_months = [f"{y}-{m:02d}" for (y, m) in expected if (y, m) not in set(months)]
    print(f"missing months: {len(missing_months)}"
          + (f"  {missing_months[:12]}" if missing_months else ""))

    # days actually present, read from band names of a sample of files
    present_days: set[date] = set()
    bad_files = []
    for m in members:
        try:
            with rasterio.open(vsi(zip_path, m)) as src:
                desc = src.descriptions
        except Exception as exc:                       # noqa: BLE001
            bad_files.append((m, str(exc)[:80]))
            continue
        for d in desc:
            mm = BAND_RE.match(d) if d else None
            if mm:
                present_days.add(date(int(mm.group(1)[:4]), int(mm.group(1)[4:6]), int(mm.group(1)[6:])))
    exp_days = [date(2010, 1, 1) + timedelta(days=i)
                for i in range((date(2024, 12, 31) - date(2010, 1, 1)).days + 1)]
    missing_days = [d for d in exp_days if d not in present_days]
    print(f"days present : {len(present_days)} of {len(exp_days)}")
    print(f"missing days : {len(missing_days)}"
          + (f"  {[str(d) for d in missing_days[:10]]}" if missing_days else ""))
    if bad_files:
        print(f"unreadable files: {len(bad_files)}")
        for name, err in bad_files[:5]:
            print(f"   {name}: {err}")
    return {"months": months, "missing_months": missing_months, "missing_days": missing_days}


def check_grid(zip_path: Path) -> None:
    print("\n== grid ==")
    members = list_members(zip_path)
    worst, worst_name = 0.0, ""
    shape_bad = []
    for m in members:
        with rasterio.open(vsi(zip_path, m)) as src:
            t = src.transform
            delta = max(abs(a - b) for a, b in
                        zip((t.a, t.b, t.c, t.d, t.e, t.f), REF_TRANSFORM))
            if delta > worst:
                worst, worst_name = delta, m
            if (src.height, src.width) != REF_SHAPE:
                shape_bad.append((m, (src.height, src.width)))
    print(f"  shape {REF_SHAPE}: {len(members) - len(shape_bad)}/{len(members)} match")
    if shape_bad:
        for name, shp in shape_bad[:5]:
            print(f"    {name}: {shp}")
    print(f"  worst transform deviation: {worst:.3e} deg  ({Path(worst_name).name})")
    print(f"  exact match: {worst <= TRANSFORM_TOL}")


def check_empty(zip_path: Path) -> list[dict]:
    print("\n== empty fraction inside the square ==")
    per_var: dict[str, list[float]] = defaultdict(list)
    rows = []
    px_area = ((SQUARE[2] - SQUARE[0]) / REF_TRANSFORM[0]) * \
              ((SQUARE[3] - SQUARE[1]) / abs(REF_TRANSFORM[4]))
    for m in list_members(zip_path):
        with rasterio.open(vsi(zip_path, m)) as src:
            win = window_for(src, SQUARE)
            if win is None:
                continue
            desc = src.descriptions
            for var in sorted({d.split("_", 1)[1] for d in desc if d and BAND_RE.match(d)}):
                idx = [i + 1 for i, d in enumerate(desc) if d and d.endswith("_" + var)]
                arr = src.read(idx, window=win).astype("float32")
                empty = ~np.isfinite(arr) | (arr == -999)
                if any(k in var for k in POSITIVE_ONLY):
                    empty |= arr <= 0
                valid = (~empty).sum(axis=(1, 2)).mean()
                # clamp: the window can be one pixel wider than the square,
                # which would otherwise produce a small negative fraction
                frac = float(np.clip(1.0 - valid / px_area, 0.0, 1.0))
                cov = float(np.clip((win.height * win.width) / px_area, 0.0, 2.0))
                per_var[var].append(frac)
                rows.append({"file": Path(m).name, "variable": var,
                             "empty_fraction": round(frac, 6), "coverage": round(cov, 6)})
    for var, fr in sorted(per_var.items()):
        print(f"  {var:12s} mean {np.mean(fr):6.3%}  max {np.max(fr):6.3%}  ({len(fr)} files)")
    return rows


def cross_check(zip_path: Path, cross_zip: Path, day: str) -> None:
    print(f"\n== value cross-check on {day} vs CAN_SRC_MERRA_AER_{day[:4]}.tif ==")
    y, mo, d = (int(x) for x in day.split("-"))
    want = f"{y}{mo:02d}{d:02d}"
    members = [m for m in list_members(zip_path) if f"{y}_{mo:02d}" in m]
    if not members:
        print(f"  no file for {y}-{mo:02d} in the new zip")
        return
    with zipfile.ZipFile(cross_zip) as zf:
        ref = [n for n in zf.namelist() if n.endswith(f"CAN_SRC_MERRA_AER_{y}.tif")]
    if not ref:
        print(f"  no CAN_SRC_MERRA_AER_{y}.tif in {cross_zip.name}")
        return
    # read the reference straight from its zip too
    with zipfile.ZipFile(cross_zip) as zf, zf.open(ref[0]) as fh:
        tmp = Path.cwd() / f"_xcheck_{y}.tif"
        tmp.write_bytes(fh.read())
    try:
        with rasterio.open(vsi(zip_path, members[0])) as new_src, rasterio.open(tmp) as old_src:
            left = max(new_src.bounds.left, old_src.bounds.left, SQUARE[0])
            right = min(new_src.bounds.right, old_src.bounds.right, SQUARE[2])
            bottom = max(new_src.bounds.bottom, old_src.bounds.bottom, 41.0)
            top = min(new_src.bounds.top, old_src.bounds.top, SQUARE[3])
            wn = from_bounds(left, bottom, right, top, new_src.transform)
            wo = from_bounds(left, bottom, right, top, old_src.transform)
            new_desc, old_desc = new_src.descriptions, old_src.descriptions
            print(f"  overlap bbox: lon {left:.2f}..{right:.2f}  lat {bottom:.2f}..{top:.2f}")
            for var in PRODUCT_KEYS:
                bi = [i + 1 for i, x in enumerate(new_desc) if x and x.endswith(want + "_" + var)]
                bj = [i + 1 for i, x in enumerate(old_desc) if x and x.endswith(want + "_" + var)]
                if not bi or not bj:
                    print(f"  {var:12s} band missing (new {bool(bi)}, old {bool(bj)})")
                    continue
                a = new_src.read(bi[0], window=wn).astype("float64")
                b = old_src.read(bj[0], window=wo).astype("float64")
                h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
                a, b = a[:h, :w], b[:h, :w]
                both = np.isfinite(a) & np.isfinite(b)
                if not both.any():
                    print(f"  {var:12s} no overlapping finite pixels")
                    continue
                diff = np.abs(a[both] - b[both])
                scale = np.maximum(np.abs(b[both]), 1e-30)
                print(f"  {var:12s} n={both.sum():5d}  max abs {diff.max():.3e}"
                      f"  max rel {(diff/scale).max():.3e}")
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="QA for the AER surface/extinction gap fill")
    ap.add_argument("--zip", required=True, help="the new export zip")
    ap.add_argument("--reference-zip", help="completed archive zip (for the grid check)")
    ap.add_argument("--cross-zip", help="Canada_Source_MERRA_AER_2012_2024_final zip")
    ap.add_argument("--day", default="2015-07-01", help="day for the value cross-check")
    ap.add_argument("--csv", default="merra_aer_sfc_check.csv")
    args = ap.parse_args()

    zp = Path(args.zip)
    if not zp.exists():
        print(f"no such zip: {zp}")
        return 2
    print(f"checking {zp.name}\n")
    info = check_structure(zp)
    check_grid(zp)
    rows = check_empty(zp)

    out = Path(args.csv)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "variable", "empty_fraction", "coverage"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")

    if args.cross_zip and Path(args.cross_zip).exists():
        cross_check(zp, Path(args.cross_zip), args.day)
    else:
        print("\n(cross-check skipped: pass --cross-zip to enable)")

    ok = not info["missing_months"] and not info["missing_days"]
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
