#!/usr/bin/env python3
"""Axis-aligned square domain around Ontario with >= 200 km clearance.

Rule
----
Ontario is projected to NAD83 / Conus Albers (EPSG:5070).  Its bounding box
(W x H) is expanded by 200 km on every side and then made square by growing
the shorter axis to the longer one, keeping the same centre.  Every edge of
the resulting square is therefore at least 200 km from the nearest point of
Ontario, and exactly 200 km on the binding axis.  The clearance is verified
by measuring polygon-to-edge distance, not assumed from the construction.

Because the square is defined in a projected CRS, its edges are straight in
Albers but slightly curved on the ellipsoid; the reported lon/lat box is the
bounding box of the sampled edge, so it is very slightly larger than the
square itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
from pyproj import CRS, Transformer
from shapely.geometry import shape, MultiPolygon, LineString
from shapely.ops import unary_union
import shapely

ROOT = Path(r"D:\ai-workspace\project")
NE = ROOT / ".ne"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ALBERS = CRS.from_epsg(5070)
WGS = CRS.from_epsg(4326)
to_alb = Transformer.from_crs(WGS, ALBERS, always_xy=True)
to_wgs = Transformer.from_crs(ALBERS, WGS, always_xy=True)
# Required clearance between Ontario and every edge of the square.
MARGIN = 200_000.0


# Read a Natural Earth GeoJSON and return its feature list.
def load(p):
    return json.load(open(p, encoding="utf-8"))["features"]


# Project a lon/lat geometry to the Albers working CRS.
def prj(geom):
    return shapely.transform(geom, lambda a: np.column_stack(to_alb.transform(a[:, 0], a[:, 1])))


feats = load(NE / "admin1.geojson")
ont = shape([f for f in feats if f["properties"].get("name") == "Ontario"
             and f["properties"].get("admin") == "Canada"][0]["geometry"])
if isinstance(ont, MultiPolygon):
    ont = unary_union(list(ont.geoms))
# Work entirely in the projected CRS: distances and areas only mean something
# there, and the square has to be square in metres.
o = prj(ont)
minx, miny, maxx, maxy = o.bounds
W0, H0 = maxx - minx, maxy - miny
# Expand by the margin, then grow the shorter axis to make it square; the
# centre stays fixed, so two edges end up with more than the minimum clearance.
side = max(W0, H0) + 2 * MARGIN
cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
sx0, sx1 = cx - side / 2, cx + side / 2
sy0, sy1 = cy - side / 2, cy + side / 2

edges = {
    "west":  LineString([(sx0, sy0), (sx0, sy1)]),
    "east":  LineString([(sx1, sy0), (sx1, sy1)]),
    "south": LineString([(sx0, sy0), (sx1, sy0)]),
    "north": LineString([(sx0, sy1), (sx1, sy1)]),
}
# Measure the clearance instead of trusting the construction that produced it.
clear = {k: o.distance(v) for k, v in edges.items()}

# lon/lat bounding box of the square perimeter
per = []
for a, b in [((sx0, sy0), (sx1, sy0)), ((sx1, sy0), (sx1, sy1)),
             ((sx1, sy1), (sx0, sy1)), ((sx0, sy1), (sx0, sy0))]:
    t = np.linspace(0, 1, 400)
    per.append(np.column_stack([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]))
per = np.vstack(per)
lon, lat = to_wgs.transform(per[:, 0], per[:, 1])

stats = {
    "square_side_km": side / 1000,
    "square_area_km2": (side / 1000) ** 2,
    "ontario_bbox_km": [W0 / 1000, H0 / 1000],
    "clearance_km": {k: v / 1000 for k, v in clear.items()},
    "lon_min": float(lon.min()), "lon_max": float(lon.max()),
    "lat_min": float(lat.min()), "lat_max": float(lat.max()),
    "ontario_area_km2": o.area / 1e6,
    "square_vs_ontario": (side / 1000) ** 2 / (o.area / 1e6),
    "corner_lonlat": [
        list(to_wgs.transform(sx0, sy0)), list(to_wgs.transform(sx1, sy1)),
    ],
    "maiac_px": {
        "lon_px": int(round((lon.max() - lon.min()) / 0.008983152841195215)),
        "lat_px": int(round((lat.max() - lat.min()) / 0.008983152841195215)),
    },
}
(OUT / "ontario_square_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in stats.items()
                  if k != "clearance_km"}, indent=2))
print("clearance km:", {k: round(v / 1000, 1) for k, v in clear.items()})

# ---------- figure ----------
countries = [f for f in load(NE / "countries.geojson")
             if f["properties"].get("CONTINENT") == "North America"]
lakes = [f for f in load(NE / "lakes.geojson")
         if (lambda r: 99 if r is None else int(r))(f["properties"].get("scalerank")) <= 2]


# Convert a shapely polygon to a matplotlib Path, keeping interior rings.
def to_path(geom):
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    verts, codes = [], []
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            if len(xy) < 4:
                continue
            x, y = to_alb.transform(xy[:, 0], xy[:, 1])
            verts.append(np.column_stack([x, y]))
            codes.append(np.concatenate([[MplPath.MOVETO],
                                         np.full(len(xy) - 2, MplPath.LINETO),
                                         [MplPath.CLOSEPOLY]]))
    return MplPath(np.vstack(verts), np.concatenate(codes))


fig = plt.figure(figsize=(13, 12), facecolor="#f7f9fb")
ax = fig.add_axes([0.04, 0.05, 0.92, 0.84])
ax.set_facecolor("#dbeafe")

for f in countries:
    ax.add_patch(PathPatch(to_path(shape(f["geometry"])), facecolor="#e5e7eb",
                           edgecolor="#9ca3af", linewidth=0.5, zorder=1))
for f in lakes:
    ax.add_patch(PathPatch(to_path(shape(f["geometry"])), facecolor="#cfe2f5",
                           edgecolor="#93c5fd", linewidth=0.35, zorder=2))
ax.add_patch(PathPatch(to_path(ont), facecolor="#ea580c", alpha=0.9,
                       edgecolor="#7c2d12", linewidth=1.0, zorder=3))
ax.add_patch(Rectangle((sx0, sy0), side, side, fill=False, edgecolor="#0f172a",
                       linewidth=2.4, linestyle=(0, (8, 3)), zorder=5))

pad = 190_000
ax.set_xlim(sx0 - pad, sx1 + pad)
ax.set_ylim(sy0 - pad, sy1 + pad)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

# dimension annotation along the bottom edge
ax.annotate("", xy=(sx0, sy0 - 45_000), xytext=(sx1, sy0 - 45_000),
            arrowprops=dict(arrowstyle="<->", color="#0f172a", lw=1.3))
ax.text((sx0 + sx1) / 2, sy0 - 70_000, f"{side/1000:,.0f} km",
        ha="center", va="top", fontsize=13, fontweight="bold", color="#0f172a")

# clearance callouts
for name, (px, py, ha, va) in {
    "west":  (sx0 - 40_000, cy, "right", "center"),
    "east":  (sx1 + 40_000, cy, "left", "center"),
    "south": (sx0 + 90_000, sy0 + 90_000, "left", "center"),
    "north": (sx0 + 90_000, sy1 - 90_000, "left", "center"),
}.items():
    txt = f"{name}: {clear[name]/1000:.0f} km"
    ax.text(px, py, txt, ha=ha, va=va, fontsize=11, color="#334155",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                      edgecolor="#cbd5e1"), zorder=6)

x, y = to_alb.transform(-85.0, 50.2)
ax.text(x, y, "ONTARIO", ha="center", va="center", fontsize=17,
        fontweight="bold", color="white", zorder=7)

# scale bar
sb = 500_000
bx = sx1 - 0.06 * side - sb
by = sy0 + 0.05 * side
ax.plot([bx, bx + sb], [by, by], color="#111827", lw=5, solid_capstyle="butt", zorder=8)
ax.text(bx + sb / 2, by + 45_000, "500 km", ha="center", fontsize=10, zorder=8)

fig.suptitle("Proposed square domain: Ontario with 200 km clearance to every edge",
             fontsize=19, fontweight="bold", y=0.965, color="#0f172a")
box = (f"side            {side/1000:,.0f} km\n"
       f"area            {(side/1000)**2:,.0f} km²\n"
       f"Ontario bbox    {W0/1000:,.0f} × {H0/1000:,.0f} km\n"
       f"clearance       ≥ 200 km per edge\n"
       f"lon / lat       {abs(lon.min()):.1f}°W–{abs(lon.max()):.1f}°W\n"
       f"                {lat.min():.1f}°N–{lat.max():.1f}°N")
fig.text(0.055, 0.905, box, fontsize=10.5, family="DejaVu Sans Mono",
         va="top", ha="left", color="#0f172a",
         bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                   edgecolor="#cbd5e1", alpha=0.95))

fig.text(0.04, 0.012,
         "Projection: NAD83 / Conus Albers (EPSG:5070). Square is axis-aligned in that CRS; "
         "clearance measured as polygon-to-edge distance. Boundaries and lakes: Natural Earth 10m.",
         fontsize=8, color="#475569")

out = OUT / "ontario_square.png"
fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
print("saved", out, f"{out.stat().st_size/1e6:.1f} MB")



