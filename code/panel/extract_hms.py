#!/usr/bin/env python3
"""HMS smoke, one month at a time, rasterized onto the lattice.

Rule
----
Each polygon is assigned to every UTC-6 day its [Start, End] interval overlaps,
using the timestamps (YYYYDDD HHMM, UTC) - never the file name. The UTC-6 day D
covers [D 06:00, D+1 06:00), so it can hold polygons from the files dated D-1, D
and D+1.

Rasterizing with rasterio.features.rasterize(all_touched=False) burns a pixel
when its *centre* is inside the polygon, which is the centre-in-polygon rule the
build needs. Polygon counts accumulate with merge_alg=add; the default replace
mode would cap hms_n_polygons at 1.

A missing file means a missing day: the three columns are NaN, not 0. Five days
in 2010-2024 have no file.

Fourteen of the 5,474 shapefiles contain unclosed rings, which pyogrio's normal
reader rejects with a GEOS error. They are read as raw WKB and repaired with
shapely.from_wkb(on_invalid="fix"), which recovered all 11 polygons in the first
file tested. Unparseable timestamps are counted, and the month fails if more
than 1% of polygons are skipped, so a silent format change cannot turn into
hms_any == 0 everywhere.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pyproj
import rasterio.enums
import rasterio.features as rfeatures
from affine import Affine
from shapely import from_wkb
from shapely.ops import transform as shp_transform

from utils import DX, DY, ROOT, X0, Y0, days_in_month

SRC = ROOT / "data" / "HMS_SMOKE"
CELLS = ROOT / "data" / "interim" / "spine" / "cells.parquet"

DENSITY = {"light": 5.0, "medium": 16.0, "heavy": 27.0}
MAX_SKIP_FRAC = 0.01
TO_3347 = pyproj.Transformer.from_crs(4326, 3347, always_xy=True)


def _parse_stamp(s) -> dt.datetime:
    """'YYYYDDD HHMM' (UTC) -> datetime."""
    txt = str(s).strip()
    year, rest = int(txt[:4]), txt[4:]
    ddd = int(rest.split(" ")[0][:3])
    hhmm = rest.split(" ")[1]
    base = dt.datetime(year, 1, 1) + dt.timedelta(days=ddd - 1)
    return base.replace(hour=int(hhmm[:2]), minute=int(hhmm[2:]))


def _density_value(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in DENSITY:
        return DENSITY[s]
    try:
        return float(s)
    except ValueError:
        return None


def _read_polygons(path: Path):
    """[(geometry in 3347, density or None, start, end)], plus a skipped count."""
    import pyogrio

    out, skipped = [], 0
    meta, _, wkb, fields = pyogrio.raw.read(f"/vsizip/{path.as_posix()}",
                                            read_geometry=True)
    if wkb is None or len(wkb) == 0:
        return out, skipped
    geoms = from_wkb(wkb, on_invalid="fix")
    # pyogrio returns the attribute columns as a list of arrays, with the names
    # in meta["fields"]; older versions returned a dict
    if isinstance(fields, dict):
        col = fields
    else:
        names = meta.get("fields")
        names = list(names) if names is not None else []
        col = dict(zip(names, fields))
    starts = col.get("Start")
    ends = col.get("End")
    dens = col.get("Density")

    for i, geom in enumerate(geoms):
        if geom is None or geom.is_empty:
            skipped += 1
            continue
        try:
            s = _parse_stamp(starts[i])
            e = _parse_stamp(ends[i])
        except Exception:
            skipped += 1
            continue
        g3347 = shp_transform(TO_3347.transform, geom)
        out.append((g3347, _density_value(dens[i] if dens is not None else None), s, e))
    return out, skipped


def _window(cells):
    r0, r1 = int(cells["row"].min()), int(cells["row"].max())
    c0, c1 = int(cells["col"].min()), int(cells["col"].max())
    # A cell centre sits at (X0 + DX*col, Y0 + DY*row), so the raster origin has
    # to be half a pixel up-left of the cell centre: without this the polygon
    # test happens 2.5 km away from the cell it is attributed to.
    transform = Affine(DX, 0.0, X0 + c0 * DX - DX / 2, 0.0, DY, Y0 + r0 * DY - DY / 2)
    return transform, r0, c0, r1 - r0 + 1, c1 - c0 + 1


def extract(year: int, month: int, cell_idx: np.ndarray) -> dict[str, np.ndarray]:
    import pandas as pd

    cells = pd.read_parquet(CELLS)
    transform, r0, c0, height, width = _window(cells)
    sub = cells.set_index("cell_idx").loc[cell_idx]
    rr = (sub["row"].to_numpy() - r0).astype("int64")
    cc = (sub["col"].to_numpy() - c0).astype("int64")

    days = days_in_month(year, month)
    n = len(days)
    poly_cnt = np.full((n, len(cell_idx)), np.nan, "float32")
    dens_max = np.full((n, len(cell_idx)), np.nan, "float32")
    missing, total_polys, total_skipped = [], 0, 0

    cache: dict[dt.date, tuple | None] = {}

    def load(d: dt.date):
        if d not in cache:
            z = SRC / f"hms_smoke{d:%Y%m%d}.zip"
            cache[d] = _read_polygons(z) if z.exists() else None
        return cache[d]

    for i, day in enumerate(days):
        if load(day) is None:
            missing.append(day.isoformat())      # whole day stays NaN
            continue

        start = dt.datetime.combine(day, dt.time(6, 0))
        end = start + dt.timedelta(days=1)

        polys = []
        for delta in (-1, 0, 1):
            got = load(day + dt.timedelta(days=delta))
            if got is None:
                continue                          # partial day: keep what we have
            found, skipped = got
            total_skipped += skipped
            total_polys += len(found) + skipped
            for geom, dens, s, e in found:
                if s < end and e >= start:
                    polys.append((geom, dens))

        if not polys:
            poly_cnt[i] = 0.0                     # file present, no smoke
            continue

        burned = rfeatures.rasterize(
            [(g, 1.0) for g, _ in polys], out_shape=(height, width),
            transform=transform, fill=0.0, all_touched=False, dtype="float32",
            merge_alg=rasterio.enums.MergeAlg.add)
        poly_cnt[i] = burned[rr, cc]

        acc = np.full((height, width), np.nan, "float32")
        known = [(g, d) for g, d in polys if d is not None]
        for val in sorted({d for _, d in known}):
            b = rfeatures.rasterize(
                [(g, val) for g, d in known if d == val], out_shape=(height, width),
                transform=transform, fill=np.nan, all_touched=False, dtype="float32")
            touch = np.isfinite(b)
            acc = np.where(touch & (np.isnan(acc) | (b > acc)), b, acc)
        dens_max[i] = acc[rr, cc]

    frac = total_skipped / total_polys if total_polys else 0.0
    if frac > MAX_SKIP_FRAC:
        raise RuntimeError(
            f"HMS {year}-{month:02d}: {total_skipped}/{total_polys} polygons skipped "
            f"({frac:.1%}) - refusing to write a silently empty month")

    out = {
        # float32 rather than int: a missing day must be NaN, and an integer
        # column cannot hold it
        "hms_n_polygons": poly_cnt.astype("float32"),
        "hms_any": np.where(np.isfinite(poly_cnt), (poly_cnt > 0).astype("float32"), np.nan).astype("float32"),
        "hms_density_max": dens_max.astype("float32"),
    }
    out["_missing_days"] = np.array(missing, dtype=object)
    out["_skipped_polygons"] = np.array([total_skipped, total_polys])
    return out