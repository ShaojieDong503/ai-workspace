#!/usr/bin/env python3
"""Cell table for the Ontario 200 km square, aligned to the CanOSSEM 5 km grid.

The CanOSSEM crosswalk (RASTER_CanOSSEM_CROSSWALK.RData) is a 1,065-column
raster of 5 km cells in EPSG:3347 (Statistics Canada Lambert).  Its cell id is
row*1065+col, and the centre of any cell is

    x = 3,688,478.20 + 4,999.827 * col
    y = 5,239,381.11 - 4,998.864 * row        (EPSG:3347 metres)

both fitted to all 74,346 valid crosswalk cells with zero residual (one cell,
id 675210 in Newfoundland, is mis-indexed by exactly 1,065 ids and excluded
from the fit).

Two problems are handled by construction rather than by patching the source:
  1. gaps in Canada   - the crosswalk is sparse, so every lattice cell in the
     square is emitted and those absent from the crosswalk are flagged
     (`in_canossem = False`), not dropped.
  2. no US coverage   - the lattice is extended south past row 910, which is
     where the Canadian data stop; those cells are flagged `country = USA`.

Cell membership is decided on the centre, so each cell belongs to exactly one
side of the boundary; `intersects_square` records the cells whose footprint
straddles the edge.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa          # noqa: F401  (imported for the parquet engine)
import pyarrow.parquet as pq  # noqa: F401
import pyreadr
from pyproj import CRS, Transformer
from shapely.geometry import shape, MultiPolygon, box, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree
import shapely

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(r"D:\ai-workspace\project")
CROSSWALK = ROOT / "CanOSSEM_V4" / "RASTER_CanOSSEM_CROSSWALK.RData"
NE = ROOT / ".ne"                      # Natural Earth boundaries (admin-1)
OUT = ROOT / "data" / "interim" / "grid"
OUT.mkdir(parents=True, exist_ok=True)

# Lattice constants, all fitted from the crosswalk (see module docstring).
W = 1065                               # cells per row; also the id stride
DX, X0 = 4999.827, 3688478.20          # x = X0 + DX * col   (metres, EPSG:3347)
DY, Y0 = -4998.864, 5239381.11         # y = Y0 + DY * row   (negative: row 0 is north)

ALBERS = CRS.from_epsg(3347)           # Statistics Canada Lambert, the grid CRS
CONUS = CRS.from_epsg(5070)            # the projection the square was defined in
to_wgs = Transformer.from_crs(ALBERS, 4326, always_xy=True)
TO_ALB = Transformer.from_crs(4326, ALBERS, always_xy=True)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_crosswalk():
    """Read the CanOSSEM crosswalk and decode its cell ids into (row, col).

    pyreadr returns a dict of objects keyed by R name; this file holds exactly
    one data frame, so the first value is the table we want.  The id column
    arrives as float64 -- it must be cast with int64, never via str(), or
    170949.0 and 170949 would become different keys.
    """
    d = list(pyreadr.read_r(str(CROSSWALK)).values())[0]
    d["id"] = d["CanOSSEM_RASTER_CELL"].astype(np.int64)
    d["row"] = d["id"] // W            # integer division: which raster row
    d["col"] = d["id"] % W             # remainder: which column in that row
    return d


def square_bounds():
    """Rebuild the study square exactly as ontario_square.py defined it.

    The square was defined in EPSG:5070 (Ontario's bounding box expanded 200 km
    on each side, then squared up).  It is re-derived here in 5070 and only then
    mapped into 3347, so the domain matches the earlier figure instead of being
    re-invented in a different projection.
    """
    to_conus = Transformer.from_crs(4326, CONUS, always_xy=True)
    conus_to_alb = Transformer.from_crs(CONUS, ALBERS, always_xy=True)

    # Ontario outline, projected to 5070.
    feats = json.load(open(NE / "admin1.geojson", encoding="utf-8"))["features"]
    ont = shape([f for f in feats if f["properties"].get("name") == "Ontario"
                 and f["properties"].get("admin") == "Canada"][0]["geometry"])
    # A MultiPolygon is flattened so buffer/area work on one geometry.
    ont = unary_union(list(ont.geoms)) if isinstance(ont, MultiPolygon) else ont
    o = shapely.transform(ont, lambda a: np.column_stack(to_conus.transform(a[:, 0], a[:, 1])))

    # Expand the bounding box by 200 km per side, then square it up by growing
    # the shorter axis to the longer one -- the centre stays put.
    mnx, mny, mxx, mxy = o.bounds
    side = max(mxx - mnx, mxy - mny) + 400_000   # +200 km on both ends
    cx, cy = (mnx + mxx) / 2, (mny + mxy) / 2
    sq_conus = box(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    return shapely.transform(sq_conus,
                             lambda a: np.column_stack(conus_to_alb.transform(a[:, 0], a[:, 1]))), side


def admin_lookup():
    """Province/state polygons for Canada and the US, projected to EPSG:3347.

    Returns the polygon list, a parallel list of (country, name, postal) and an
    STRtree for fast point-in-polygon.  The tree must be built on the *projected*
    geometry -- querying metre coordinates against lon/lat polygons silently
    returns nothing.
    """
    feats = json.load(open(NE / "admin1.geojson", encoding="utf-8"))["features"]
    polys, meta = [], []
    for f in feats:
        p = f["properties"]
        if p.get("admin") not in ("Canada", "United States of America"):
            continue                       # skip Mexico and everything overseas
        g = shapely.transform(shape(f["geometry"]),
                              lambda a: np.column_stack(TO_ALB.transform(a[:, 0], a[:, 1])))
        # One entry per polygon part, so a state split by water still matches.
        for part in (g.geoms if isinstance(g, MultiPolygon) else [g]):
            polys.append(part)
            meta.append((p.get("admin"), p.get("name"), p.get("postal")))
    tree = STRtree(polys)
    return polys, meta, tree


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def main():
    cw = load_crosswalk()
    SQ, side = square_bounds()
    polys, meta, tree = admin_lookup()
    print(f"square side {side/1000:,.1f} km")

    # Convert the square's extent into lattice indices, with a one-cell margin
    # so edge cells are not lost to rounding.
    c0 = int(np.floor((SQ.bounds[0] - X0) / DX)) - 1        # west-most column
    c1 = int(np.ceil((SQ.bounds[2] - X0) / DX)) + 1         # east-most column
    r0 = int(np.ceil((Y0 - SQ.bounds[3]) / -DY)) - 1        # north-most row
    r1 = int(np.floor((Y0 - SQ.bounds[1]) / -DY)) + 1       # south-most row
    print(f"lattice window cols {c0}..{c1}, rows {r0}..{r1}")

    # Enumerate every candidate cell in that window.  This is the step that
    # makes the result gap-free: the full grid is generated first and only then
    # filtered, so no cell can go missing because it was absent upstream.
    cols = np.arange(c0, c1 + 1)
    rows = np.arange(r0, r1 + 1)
    cc, rr = np.meshgrid(cols, rows)
    cc, rr = cc.ravel(), rr.ravel()
    x = X0 + DX * cc
    y = Y0 + DY * rr
    lon, lat = to_wgs.transform(x, y)

    # Membership by cell centre: a cell whose centre is inside the square is
    # "in"; the rest are only kept if their 5 km footprint touches the edge.
    pts = shapely.points(x, y)
    in_sq = shapely.contains(SQ, pts)
    keep = in_sq
    inside = shapely.box(x - abs(DX) / 2, y - abs(DY) / 2,
                         x + abs(DX) / 2, y + abs(DY) / 2)
    touches = shapely.intersects(inside, SQ) & ~keep   # straddlers only

    idx = np.flatnonzero(keep | touches)

    # Assemble the table.  Dtypes are explicit: ids and indices as integers,
    # coordinates as float64 (centimetre precision matters for the join),
    # lon/lat as float32 (they are only labels here).
    df = pd.DataFrame({
        "cell_id": (rr[idx] * W + cc[idx]).astype(np.int64),
        "row": rr[idx].astype(np.int32),
        "col": cc[idx].astype(np.int32),
        "x_3347": x[idx].astype(np.float64),
        "y_3347": y[idx].astype(np.float64),
        "lon": lon[idx].astype(np.float32),
        "lat": lat[idx].astype(np.float32),
        "in_square": keep[idx],
        "intersects_square": (keep[idx] | touches[idx]),
    }).sort_values("cell_id").reset_index(drop=True)

    # Country / province / state by point-in-polygon.  STRtree.query with
    # predicate="within" returns (point_index, polygon_index) pairs; the first
    # hit wins, which matters only where a centre sits exactly on a border.
    pt = shapely.points(df["x_3347"].to_numpy(), df["y_3347"].to_numpy())
    hits = tree.query(pt, predicate="within")
    country = np.full(len(df), None, dtype=object)
    province = np.full(len(df), None, dtype=object)
    code = np.full(len(df), None, dtype=object)
    for cell_idx, poly_idx in zip(*hits):
        if country[cell_idx] is None:
            country[cell_idx], province[cell_idx], code[cell_idx] = meta[poly_idx]
    df["country"] = country            # null = water: no admin polygon covers it
    df["province_state"] = province
    df["postal"] = code

    # Flag which cells the crosswalk already knows about, then classify them.
    cw_ids = set(cw["id"].tolist())
    df["in_canossem"] = df["cell_id"].isin(cw_ids)
    df["status"] = np.select(
        [df["country"].eq("United States of America"),
         df["country"].eq("Canada") & ~df["in_canossem"],
         df["in_canossem"]],
        ["US cell (new)", "Canada gap (added)", "CanOSSEM cell"],
        default="other")               # "other" is water / outside both countries

    df.to_parquet(OUT / "cell_table_square.parquet", index=False, compression="zstd")

    # ------------------------------------------------------------------
    # QA: prove the fitted lattice reproduces the crosswalk without error.
    # ------------------------------------------------------------------
    chk = cw[cw["id"].isin(df["cell_id"])].merge(
        df[["cell_id", "x_3347", "y_3347"]], left_on="id", right_on="cell_id", how="left")
    xpred = X0 + DX * (chk["id"] % W)          # lattice prediction from the id alone
    ypred = Y0 + DY * (chk["id"] // W)
    err = np.sqrt((chk["x_3347"] - xpred) ** 2 + (chk["y_3347"] - ypred) ** 2)
    print(f"crosswalk cells inside the table: {len(chk):,}; max lattice error {err.max():.3f} m")

    # Human-readable summary of what was produced.
    summary = df["status"].value_counts().to_dict()
    print("\ncell status:", summary)
    print("total cells:", f"{len(df):,}")
    print("in_square (centroid):", int(df.in_square.sum()),
          "| edge cells added:", int((~df.in_square).sum()))
    print("\nby country:", df["country"].value_counts(dropna=False).to_dict())
    print("\nUS states present:")
    print(df[df.country == "United States of America"]["province_state"].value_counts().to_string())
    print("\nCanada provinces present:")
    print(df[df.country == "Canada"]["province_state"].value_counts().to_string())
    print(f"\nunassigned (outside Canada/US polygons): {int(df.country.isna().sum())}")
    print("bbox lon/lat:", round(df.lon.min(), 3), round(df.lon.max(), 3),
          round(df.lat.min(), 3), round(df.lat.max(), 3))
    print("wrote", OUT / "cell_table_square.parquet")


if __name__ == "__main__":
    main()
