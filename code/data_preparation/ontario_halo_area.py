#!/usr/bin/env python3
"""How large is Ontario plus a 200 km halo, and where does it sit in North America?

Method
------
  Ontario polygon   Natural Earth 10m admin-1 (land boundary; Great Lakes are
                    excluded, so the area is the land surface, not the official
                    total that includes inland water).
  200 km halo       buffer computed in an azimuthal-equidistant projection
                    centred on Ontario, where distances from the centre are
                    true. Scale error at the polygon edge is ~0.7%, i.e. about
                    +-1.4 km on a 200 km radius.
  area              polygons are reprojected to EPSG:5070 (NAD83 / Conus
                    Albers), an equal-area projection, so area is true.
  map               same Albers projection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shapely
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.ops import transform as shp_transform, unary_union

ROOT = Path(r"D:\ai-workspace\project")
NE = ROOT / ".ne"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Projections: Albers (equal-area) for measuring area, azimuthal-equidistant
# centred on Ontario for the 200 km buffer, where distance from the centre is true.
ALBERS = CRS.from_epsg(5070)
WGS = CRS.from_epsg(4326)


# Read a Natural Earth GeoJSON and return its feature list.
def load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["features"]


# ---------- Ontario ----------
admin1 = load(NE / "admin1.geojson")
ont_feats = [f for f in admin1
             if (f["properties"].get("name") or "").strip() == "Ontario"
             and (f["properties"].get("admin") or "").strip() == "Canada"]
assert len(ont_feats) == 1, f"expected 1 Ontario feature, got {len(ont_feats)}"
ontario = shape(ont_feats[0]["geometry"])
if isinstance(ontario, MultiPolygon):
    ontario = unary_union(list(ontario.geoms))

lat0, lon0 = ontario.centroid.y, ontario.centroid.x
AEQD = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")
to_aeqd = Transformer.from_crs(WGS, AEQD, always_xy=True)
to_albers = Transformer.from_crs(WGS, ALBERS, always_xy=True)
aeqd_to_alb = Transformer.from_crs(AEQD, ALBERS, always_xy=True)

ont_aeqd = shp_transform(to_aeqd.transform, ontario)
halo_aeqd = ont_aeqd.buffer(200_000, quad_segs=64)
union_aeqd = unary_union([ont_aeqd, halo_aeqd])

ont_alb = shp_transform(to_albers.transform, ontario)
halo_alb = shp_transform(aeqd_to_alb.transform, halo_aeqd)
union_alb = shp_transform(aeqd_to_alb.transform, union_aeqd)

A_ON = ont_alb.area / 1e6
A_UNION = union_alb.area / 1e6
A_HALO = A_UNION - A_ON
minx, miny, maxx, maxy = union_alb.bounds
W, H = (maxx - minx) / 1000, (maxy - miny) / 1000

# southern extent of the halo, in plain degrees
to_wgs = Transformer.from_crs(AEQD, WGS, always_xy=True)
south_aeqd = shapely.Point(minx * 0, 0)
hx, hy = np.array(halo_aeqd.exterior.coords).T
lon_s, lat_s = to_wgs.transform(hx, hy)
i = int(np.argmin(lat_s))

stats = {
    "ontario_area_km2": A_ON,
    "halo_area_km2": A_HALO,
    "union_area_km2": A_UNION,
    "bbox_w_km": W, "bbox_h_km": H, "bbox_area_km2": W * H,
    "equiv_square_side_km": float(np.sqrt(A_UNION)),
    "southmost_lat": float(lat_s[i]), "southmost_lon": float(lon_s[i]),
    "northmost_lat": float(np.max(lat_s)),
    "halo_pct_of_ontario": 100 * A_HALO / A_ON,
    "centroid": [lat0, lon0],
}
(OUT / "ontario_200km_halo_stats.json").write_text(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()}, indent=2), encoding="utf-8")

# ---------- drawing helpers ----------
# Convert a shapely polygon to a matplotlib Path, keeping interior rings
# (lakes) so the fill does not swallow them.
def rings_to_path(geom, tr):
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    verts, codes = [], []
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            if len(xy) < 4:
                continue
            if tr is None:
                x, y = xy[:, 0], xy[:, 1]
            else:
                x, y = tr.transform(xy[:, 0], xy[:, 1])
            pts = np.column_stack([x, y])
            verts.append(pts)
            codes.append(np.concatenate([[MplPath.MOVETO],
                                         np.full(len(pts) - 2, MplPath.LINETO),
                                         [MplPath.CLOSEPOLY]]))
    return MplPath(np.vstack(verts), np.concatenate(codes))


# Add one projected polygon to an axes.
def draw(ax, geom, tr, **kw):
    ax.add_patch(PathPatch(rings_to_path(geom, tr), **kw))


countries = [f for f in load(NE / "countries.geojson")
             if f["properties"].get("CONTINENT") == "North America"]
intr = [f for f in admin1 if f["properties"].get("admin") in
        ("Canada", "United States of America", "Mexico")]

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(19, 10.5), gridspec_kw={"width_ratios": [1.25, 1]},
    facecolor="#f7f9fb")
fig.subplots_adjust(left=0.03, right=0.985, top=0.90, bottom=0.06, wspace=0.06)

for ax in (ax1, ax2):
    ax.set_facecolor("#dbeafe")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#94a3b8"); s.set_linewidth(0.8)

# land
for f in countries:
    g = shape(f["geometry"])
    draw(ax1, g, to_albers, facecolor="#e5e7eb", edgecolor="#9ca3af", linewidth=0.4, zorder=1)
    draw(ax2, g, to_albers, facecolor="#e5e7eb", edgecolor="#9ca3af", linewidth=0.4, zorder=1)

# internal boundaries (light)
for f in intr:
    g = shape(f["geometry"]).boundary
    xy = np.asarray(g.geoms[0].coords) if hasattr(g, "geoms") else np.asarray(g.coords)
    for part in (g.geoms if hasattr(g, "geoms") else [g]):
        xy = np.asarray(part.coords)
        x, y = to_albers.transform(xy[:, 0], xy[:, 1])
        ax1.plot(x, y, color="#b6bcc6", lw=0.35, zorder=2)
        ax2.plot(x, y, color="#b6bcc6", lw=0.45, zorder=2)

# halo
for ax, lw in ((ax1, 1.0), (ax2, 1.4)):
    ax.add_patch(PathPatch(rings_to_path(halo_alb, None), facecolor="#fb923c", alpha=0.30,
                           edgecolor="#c2410c", linewidth=lw, linestyle=(0, (5, 2)), zorder=3))
    ax.add_patch(PathPatch(rings_to_path(ont_alb, None), facecolor="#ea580c", alpha=0.92,
                           edgecolor="#7c2d12", linewidth=lw, zorder=4))

# lakes on top, so the province fill does not swallow the Great Lakes
lakes = [f for f in load(NE / "lakes.geojson")
         if (lambda r: 99 if r is None else int(r))(f["properties"].get("scalerank")) <= 2]
for f in lakes:
    g = shape(f["geometry"])
    for ax in (ax1, ax2):
        draw(ax, g, to_albers, facecolor="#cfe2f5", edgecolor="#93c5fd",
             linewidth=0.35, zorder=5)

# graticule
for lon in range(-130, -50, 10):
    xs, ys = to_albers.transform(np.full(200, lon), np.linspace(20, 65, 200))
    ax1.plot(xs, ys, color="#cbd5e1", lw=0.5, zorder=0.5)
    ax2.plot(xs, ys, color="#cbd5e1", lw=0.6, zorder=0.5)
for lat in range(25, 65, 5):
    xs, ys = to_albers.transform(np.linspace(-135, -50, 200), np.full(200, lat))
    ax1.plot(xs, ys, color="#cbd5e1", lw=0.5, zorder=0.5)
    ax2.plot(xs, ys, color="#cbd5e1", lw=0.6, zorder=0.5)

# limits: sample the lon/lat box perimeter and project it
# Project a lon/lat window's perimeter and return Albers axis limits; a
# straight lon/lat box is not axis-aligned in Albers, so all four edges are
# sampled rather than just the corners.
def box_limits(lon0_, lon1_, lat0_, lat1_):
    lons = np.concatenate([np.linspace(lon0_, lon1_, 120)] * 2)
    lats = np.concatenate([np.full(120, lat0_), np.full(120, lat1_)])
    lons2 = np.concatenate([np.full(120, lon0_), np.full(120, lon1_)])
    lats2 = np.concatenate([np.linspace(lat0_, lat1_, 120)] * 2)
    x, y = to_albers.transform(np.concatenate([lons, lons2]), np.concatenate([lats, lats2]))
    return min(x), max(x), min(y), max(y)

x0, x1, y0, y1 = box_limits(-133, -57, 23.5, 60)
ax1.set_xlim(x0, x1); ax1.set_ylim(y0, y1)

pad = 120_000
ax2.set_xlim(minx - pad, maxx + pad); ax2.set_ylim(miny - pad, maxy + pad)

# labels
for lon, lat, name in [(-100, 45, "USA"), (-106, 55, "Canada"), (-84, 41.5, "Ohio"),
                       (-79, 47, "Ontario"), (-99, 24, "Mexico")]:
    x, y = to_albers.transform(lon, lat)
    ax1.text(x, y, name, ha="center", va="center", fontsize=11,
             color="#374151", style="italic", zorder=6)

x, y = to_albers.transform(-82.5, 51.5)
ax2.text(x, y, "ONTARIO", ha="center", va="center", fontsize=16, fontweight="bold",
         color="white", zorder=7)

# scale bar on the zoom panel
sb = 400_000
xr = ax2.get_xlim()[1] - ax2.get_xlim()[0]
sx = ax2.get_xlim()[0] + 0.06 * xr
sy = ax2.get_ylim()[0] + 0.07 * (ax2.get_ylim()[1] - ax2.get_ylim()[0])
ax2.plot([sx, sx + sb], [sy, sy], color="#111827", lw=4, solid_capstyle="butt", zorder=8)
ax2.text(sx + sb / 2, sy + 0.035 * (ax2.get_ylim()[1] - ax2.get_ylim()[0]), "400 km",
         ha="center", fontsize=10, color="#111827", zorder=8)

# southern-most marker
sx_m, sy_m = to_albers.transform(stats["southmost_lon"], stats["southmost_lat"])
ax2.plot(sx_m, sy_m, "v", color="#7c2d12", markersize=9, zorder=9)
ax2.annotate(f"southern limit of the halo\n{abs(stats['southmost_lat']):.2f}°N",
             xy=(sx_m, sy_m), xycoords="data", xytext=(0.52, 0.12),
             textcoords="axes fraction", ha="center", fontsize=9.5, color="#7c2d12",
             arrowprops=dict(arrowstyle="->", color="#7c2d12", lw=1,
                             connectionstyle="arc3,rad=0.15"), zorder=9)

# legend
from matplotlib.patches import Patch
handles = [Patch(facecolor="#ea580c", edgecolor="#7c2d12", label="Ontario"),
           Patch(facecolor="#fb923c", edgecolor="#c2410c", alpha=0.45,
                 label="200 km halo around Ontario")]
ax1.legend(handles=handles, loc="lower left", fontsize=10, frameon=True,
           framealpha=0.95, edgecolor="#cbd5e1")

fig.suptitle("Ontario and its 200 km halo — extent in North America",
             fontsize=20, fontweight="bold", y=0.965, color="#0f172a")

box = (f"Ontario area (polygon)  {A_ON:,.0f} km²\n"
       f"200 km halo (extra)      {A_HALO:,.0f} km²\n"
       f"Union                    {A_UNION:,.0f} km²\n"
       f"Bounding square          {W:,.0f} × {H:,.0f} km\n"
       f"                         = {W*H:,.0f} km²\n"
       f"Equivalent square side   {np.sqrt(A_UNION):,.0f} km")
ax1.text(0.015, 0.985, box, transform=ax1.transAxes, va="top", ha="left", fontsize=10,
         family="DejaVu Sans Mono", color="#0f172a", zorder=10,
         bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#cbd5e1", alpha=0.95))

fig.text(0.03, 0.012,
         "Projection: NAD83 / Conus Albers (EPSG:5070, equal-area). "
         "Boundaries: Natural Earth 10m admin-1. Halo: 200 km buffer in azimuthal-equidistant "
         "projection centred on Ontario (±1.4 km at the rim). The Ontario polygon includes inland water; lakes are drawn for context.",
         fontsize=8, color="#475569")

out = OUT / "ontario_200km_halo.png"
fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
print("saved", out, f"{out.stat().st_size/1e6:.1f} MB")







