#!/usr/bin/env python3
"""AQS PM2.5 monitor coverage of the square domain.

Reads the EPA AQS hourly PM2.5 (parameter 88101) annual archives, takes the
unique monitoring sites per year, and reports which states they cover, both
nationally and inside the square defined by ontario_square.py.
"""
import argparse, json, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import shape, MultiPolygon
from shapely.ops import unary_union
import shapely

ROOT = Path(r"D:\ai-workspace\project")
RAW = ROOT / "data" / "raw" / "AQS"
NE = ROOT / ".ne"
OUT = ROOT / "data" / "interim" / "domain"
USE = ["State Code", "County Code", "Site Num", "Latitude", "Longitude", "State Name"]


# Rebuild the square exactly as ontario_square.py did, in EPSG:5070.
def square():
    to_alb = Transformer.from_crs(4326, CRS.from_epsg(5070), always_xy=True)
    feats = json.load(open(NE / "admin1.geojson", encoding="utf-8"))["features"]
    ont = shape([f for f in feats if f["properties"].get("name") == "Ontario"
                 and f["properties"].get("admin") == "Canada"][0]["geometry"])
    ont = unary_union(list(ont.geoms)) if isinstance(ont, MultiPolygon) else ont
    o = shapely.transform(ont, lambda a: np.column_stack(to_alb.transform(a[:, 0], a[:, 1])))
    minx, miny, maxx, maxy = o.bounds
    side = max(maxx - minx, maxy - miny) + 400_000
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    return (cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2), to_alb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=list(range(2010, 2025)))
    args = ap.parse_args()
    (x0, x1, y0, y1), to_alb = square()

    # One pass per annual archive: read the CSV straight out of the zip (no
    # extraction) and keep only the site identifiers and coordinates.
    per_year = []
    for y in args.years:
        z = RAW / f"hourly_88101_{y}.zip"
        if not z.exists():
            print(f"{y}: missing"); continue
        with zipfile.ZipFile(z) as zf:
            name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(name) as fh:
                d = pd.read_csv(fh, usecols=USE,
                                dtype={"State Code": str, "County Code": str, "Site Num": str})
        # The hourly table repeats each site thousands of times; one row per site is
        # all this analysis needs.
        s = d.drop_duplicates(subset=["State Code", "County Code", "Site Num"])
        s = s.assign(year=y)
        per_year.append(s)
        print(f"{y}: {len(d):,} rows -> {len(s):,} sites")

    alls = pd.concat(per_year, ignore_index=True)
    keys = ["State Code", "County Code", "Site Num"]
    sites = alls.groupby(keys).agg(
        state=("State Name", "first"),
        lat=("Latitude", "first"),
        lon=("Longitude", "first"),
        years=("year", lambda s: len(set(s))),
    ).reset_index()
    sites["site_id"] = sites["State Code"] + "-" + sites["County Code"] + "-" + sites["Site Num"]

    # Project site coordinates once, then test them against the square.
    x, y = to_alb.transform(sites["lon"].to_numpy(), sites["lat"].to_numpy())
    # Same projected box test used by the other domain scripts.
    sites["in_square"] = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)

    sites.to_csv(OUT / "aqs_pm25_sites_2010_2024.csv", index=False, encoding="utf-8-sig")

    nat = sites.groupby("state").size().sort_values(ascending=False)
    insq = sites[sites["in_square"]].groupby("state").size().sort_values(ascending=False)
    print(f"\nAQS PM2.5 sites in archive: {len(sites):,} across {sites['state'].nunique()} states")
    print(f"inside the square: {int(sites['in_square'].sum()):,} across {len(insq)} states")
    tab = pd.DataFrame({"sites_in_square": insq}).reindex(insq.index)
    print(tab.to_string())


if __name__ == "__main__":
    main()
