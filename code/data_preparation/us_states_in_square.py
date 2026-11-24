#!/usr/bin/env python3
"""Which US states overlap the square domain, and by how much?

The square is the Ontario bounding box in EPSG:5070 expanded 200 km per side
(ontario_square.py).  Each US state polygon is intersected with it in the same
equal-area projection, so the areas are true km2.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import shape, MultiPolygon, box
from shapely.ops import unary_union
import shapely

ROOT = Path(r"D:\ai-workspace\project")
NE = ROOT / ".ne"
OUT = ROOT / "data" / "interim" / "domain"
OUT.mkdir(parents=True, exist_ok=True)

ALBERS = CRS.from_epsg(5070)
to_alb = Transformer.from_crs(4326, ALBERS, always_xy=True)


def prj(g):
    return shapely.transform(g, lambda a: np.column_stack(to_alb.transform(a[:, 0], a[:, 1])))


feats = json.load(open(NE / "admin1.geojson", encoding="utf-8"))["features"]
ont = shape([f for f in feats if f["properties"].get("name") == "Ontario"
             and f["properties"].get("admin") == "Canada"][0]["geometry"])
if isinstance(ont, MultiPolygon):
    ont = unary_union(list(ont.geoms))
o = prj(ont)
minx, miny, maxx, maxy = o.bounds
side = max(maxx - minx, maxy - miny) + 400_000
cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
SQ = box(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
SQ_AREA = SQ.area / 1e6

# Intersect each US state polygon with the square; only states that overlap
# produce a row.
rows = []
for f in feats:
    p = f["properties"]
    if p.get("admin") != "United States of America":
        continue
    g = prj(shape(f["geometry"]))
    if not g.intersects(SQ):
        continue
    # Clip the state to the square; the area of the clip is the overlap.
    inter = g.intersection(SQ)
    rows.append({
        "state": p.get("name"),
        "postal": p.get("postal"),
        "intersect_km2": inter.area / 1e6,
        "state_km2": g.area / 1e6,
        "share_of_square_pct": 100 * inter.area / SQ.area,
        "share_of_state_pct": 100 * inter.area / g.area,
    })

# Rank states by how much of the square they occupy.
df = pd.DataFrame(rows).sort_values("intersect_km2", ascending=False)
df.round(1).to_csv(OUT / "us_states_in_square.csv", index=False, encoding="utf-8-sig")

print(f"square area {SQ_AREA:,.0f} km2, side {side/1000:,.0f} km")
print(f"US states overlapping: {len(df)}")
print(f"US land inside the square: {df['intersect_km2'].sum():,.0f} km2 "
      f"({100*df['intersect_km2'].sum()/SQ_AREA:.1f}% of the square)")
print()
print(df.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
