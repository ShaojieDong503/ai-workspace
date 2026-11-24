#!/usr/bin/env python3
"""Independent check of the HMS extractor, by shapely containment.

For each sampled cell-day it re-reads the relevant zipped shapefiles, converts
the polygons to EPSG:3347, and tests the cell centre directly - no rasterization,
which is what the extractor used.
"""
from __future__ import annotations

import argparse
import datetime as dt
import zipfile

import numpy as np
import pandas as pd
import pyproj
from shapely.ops import transform as shp_transform

from utils import ROOT
from verify_common import cells, check, compare, sample_pairs, summary

SRC = ROOT / "data" / "HMS_SMOKE"
DENSITY = {"light": 5.0, "medium": 16.0, "heavy": 27.0}
TO_3347 = pyproj.Transformer.from_crs(4326, 3347, always_xy=True)


def parse_stamp(s):
    y, rest = int(str(s)[:4]), str(s)[4:]
    ddd = int(rest.split(" ")[0][:3])
    hhmm = rest.split(" ")[1]
    return (dt.datetime(y, 1, 1) + dt.timedelta(days=ddd - 1)).replace(
        hour=int(hhmm[:2]), minute=int(hhmm[2:]))


def read_day(day: dt.date):
    p = SRC / f"hms_smoke{day:%Y%m%d}.zip"
    if not p.exists():
        return None
    out = []
    import pyogrio
    from shapely import from_wkb

    # fourteen of the 5,474 files carry unclosed rings; read raw WKB and repair
    meta, _, wkb, fields = pyogrio.raw.read(f"/vsizip/{p.as_posix()}", read_geometry=True)
    if wkb is None or len(wkb) == 0:
        return out
    geoms = from_wkb(wkb, on_invalid="fix")
    if isinstance(fields, dict):
        col = fields
    else:
        names = meta.get("fields")
        col = dict(zip(list(names) if names is not None else [], fields))

    for i, g in enumerate(geoms):
        if g is None or g.is_empty:
            continue
        try:
            s, e = parse_stamp(col["Start"][i]), parse_stamp(col["End"][i])
        except Exception:
            continue
        dens = DENSITY.get(str(col["Density"][i]).strip().lower())
        out.append((shp_transform(TO_3347.transform, g), dens, s, e))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the HMS extractor")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--year", type=int, default=2015)
    ap.add_argument("--month", type=int, default=7)
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    df = sample_pairs(args.cells, args.year, args.month, args.n)
    cs = cells()
    cache = {}

    def get(day):
        if day not in cache:
            cache[day] = read_day(day)
        return cache[day]

    got_any, want_any = [], []
    got_n, want_n = [], []
    got_d, want_d = [], []
    missing_days = 0

    for row in df.itertuples():
        day = pd.Timestamp(row.date).date()
        got_any.append(row.hms_any)
        got_n.append(row.hms_n_polygons)
        got_d.append(row.hms_density_max)

        if get(day) is None:
            missing_days += 1
            want_any.append(np.nan); want_n.append(np.nan); want_d.append(np.nan)
            continue

        start = dt.datetime.combine(day, dt.time(6))
        end = start + dt.timedelta(days=1)
        polys = []
        for delta in (-1, 0, 1):
            got_day = get(day + dt.timedelta(days=delta))
            if not got_day:
                continue
            polys += [(g, d) for g, d, s, e in got_day if s < end and e >= start]

        pt = cs.loc[row.cell_id]
        hit = [(g, d) for g, d in polys if g.covers(__import__("shapely").Point(pt.x_3347, pt.y_3347))]
        want_any.append(1 if hit else 0)
        want_n.append(len(hit))
        known = [d for _, d in hit if d is not None]
        want_d.append(max(known) if known else np.nan)

    compare("hms_any", np.array(got_any, "float64"), np.array(want_any, "float64"))
    compare("hms_n_polygons", np.array(got_n, "float64"), np.array(want_n, "float64"))
    compare("hms_density_max", np.array(got_d, "float64"), np.array(want_d, "float64"), tol=1e-6)
    print(f"sampled cell-days whose HMS file is missing: {missing_days}")
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())