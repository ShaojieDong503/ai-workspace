#!/usr/bin/env python3
"""Which non-Ontario NAPS stations fall inside the proposed square domain?

The square is the Ontario bounding box in EPSG:5070 expanded 200 km per side
and squared up (see ontario_square.py).  Stations from the raw annual files are
tested against the exact square, and against its lon/lat bounding box.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import shape, MultiPolygon
from shapely.ops import unary_union
import shapely

ROOT = Path(r"D:\ai-workspace\project")
RAW = ROOT / "data" / "Target_Data_2010_2024"
NE = ROOT / ".ne"
OUT = ROOT / "data" / "interim" / "domain"
OUT.mkdir(parents=True, exist_ok=True)

to_alb = Transformer.from_crs(4326, CRS.from_epsg(5070), always_xy=True) if False else None
from pyproj import Transformer as T
to_alb = T.from_crs(CRS.from_epsg(4326), CRS.from_epsg(5070), always_xy=True)

feats = json.load(open(NE / "admin1.geojson", encoding="utf-8"))["features"]
ont = shape([f for f in feats if f["properties"].get("name") == "Ontario"
             and f["properties"].get("admin") == "Canada"][0]["geometry"])
if isinstance(ont, MultiPolygon):
    ont = unary_union(list(ont.geoms))
o = shapely.transform(ont, lambda a: np.column_stack(to_alb.transform(a[:, 0], a[:, 1])))
minx, miny, maxx, maxy = o.bounds
side = max(maxx - minx, maxy - miny) + 400_000
cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
x0, x1, y0, y1 = cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2

# Collect the unique stations from every annual file (all provinces).
rows = []
for p in sorted(RAW.glob("PM25_*.csv")):
    d = pd.read_csv(p, skiprows=7, encoding="utf-8-sig", dtype=str,
                    usecols=[0, 2, 3, 4, 5, 6])
    d.columns = [c.split("//")[0].strip() for c in d.columns]
    rows.append(d)
allst = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["NAPS ID"])
allst["stn"] = allst["NAPS ID"].str.strip()
allst["lat"] = pd.to_numeric(allst["Latitude"], errors="coerce")
allst["lon"] = pd.to_numeric(allst["Longitude"], errors="coerce")
allst["prov"] = allst["Province/Territory"].str.strip().str.upper()
allst["city"] = allst["City"].str.strip()

# Project every station once, then the box test is four comparisons.
x, y = to_alb.transform(allst["lon"].to_numpy(), allst["lat"].to_numpy())
allst["x"], allst["y"] = x, y
allst["in_square"] = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
# Distance to the nearest edge; anything negative would mean outside the box.
allst["edge_km"] = np.minimum.reduce([x - x0, x1 - x, y - y0, y1 - y]) / 1000

# lon/lat bounding box for comparison
lonc = np.array([x0, x1, x0, x1]); latc = np.array([y0, y0, y1, y1])
to_wgs = T.from_crs(CRS.from_epsg(5070), CRS.from_epsg(4326), always_xy=True)
print("square bounds (Albers m):", round(x0), round(x1), round(y0), round(y1))
print("square side km:", round(side/1000, 1))

# Stations inside the exact square, grouped by province and sorted by depth
# from the edge.
inside = allst[allst["in_square"]].sort_values(["prov", "edge_km"], ascending=[True, False])
print("\ntotal stations in raw data:", len(allst))
print("stations inside square:", len(inside))
print("\nby province:")
print(inside["prov"].value_counts().to_string())
print("\noutside Ontario, inside the square:")
cols = ["stn", "city", "prov", "lat", "lon", "edge_km"]
print(inside[inside["prov"] != "ON"][cols].to_string(index=False))

inside.to_csv(OUT / "stations_inside_square.csv", index=False, encoding="utf-8-sig")
print("\nwrote", OUT / "stations_inside_square.csv")
