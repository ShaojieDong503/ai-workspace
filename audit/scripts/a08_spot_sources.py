"""Independent spot recomputation of MERRA, fire and HMS for months the
pipeline verifiers never sample (they use January and July only).

Different code paths from the extractors:
  MERRA  rasterio src.index(lon, lat) on the cell centre, band by description
  fire   brute-force haversine over that UTC-6 day's detections
  HMS    shapely point-in-polygon on the lon/lat cell centre (no reprojection,
         no rasterising); timestamps parsed with strptime
"""
import datetime as dt, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, rasterio, shapely, pyogrio
ROOT = Path(r"D:\ai-workspace\project")
cells = pd.read_parquet(ROOT / "data/interim/spine/cells.parquet").set_index("cell_id")
tgt_cells = pd.read_parquet(ROOT / "data/processed/target/target_table.parquet", columns=["cell_id"]).cell_id.unique()
MONTHS = [(2012, 4), (2016, 11), (2018, 9), (2023, 6), (2024, 12)]
VARS = ["PBLH", "TLML", "TOTEXTTAU", "BCSMASS", "U850", "SPEEDMAX", "QLML", "DUSMASS25"]
rng = np.random.default_rng(20261108)
R = 6371.0088
res = []

def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def hms_polys(day):
    z = ROOT / f"data/HMS_SMOKE/hms_smoke{day:%Y%m%d}.zip"
    if not z.exists():
        return None
    df = pyogrio.raw.read(f"/vsizip/{z.as_posix()}")
    meta, _, wkb, fields = df
    names = list(meta["fields"])
    col = dict(zip(names, fields))
    g = shapely.from_wkb(wkb, on_invalid="fix")
    out = []
    for i in range(len(g)):
        s = dt.datetime.strptime(str(col["Start"][i]).strip(), "%Y%j %H%M")
        e = dt.datetime.strptime(str(col["End"][i]).strip(), "%Y%j %H%M")
        out.append((g[i], s, e, str(col["Density"][i]).strip().lower() if "Density" in col else None))
    return out

for (y, m) in MONTHS:
    f = ROOT / f"data/processed/prediction/{y}/prediction_{y}_{m:02d}.parquet"
    ndays = pd.Period(f"{y}-{m:02d}").days_in_month
    picks = [(int(c), int(rng.integers(1, ndays + 1))) for c in
             np.concatenate([rng.choice(cells.index.to_numpy(), 30), rng.choice(tgt_cells, 10)])]
    # read only the picked cells
    tbl = pq.read_table(f, filters=[("cell_id", "in", sorted({c for c, _ in picks}))]).to_pandas()
    tbl["day"] = pd.to_datetime(tbl["date"]).dt.day
    # fire for the month (+ next year's file for December)
    fr = []
    for yy in ([y, y + 1] if m == 12 else [y]):
        for cn in ["Canada", "United_States"]:
            p = ROOT / f"data/MODIS_fire/modis_{yy}_{cn}.csv"
            if p.exists():
                fr.append(pd.read_csv(p, usecols=["latitude", "longitude", "acq_date", "acq_time", "frp", "type"]))
    fire = pd.concat(fr).drop_duplicates()
    fire = fire[fire["type"] == 0]
    t = pd.to_datetime(fire.acq_date) + pd.to_timedelta(fire.acq_time // 100, unit="h") + pd.to_timedelta(fire.acq_time % 100, unit="m")
    fire["d6"] = (t - pd.Timedelta(hours=6)).dt.date
    with rasterio.open(ROOT / f"data/raw/merra2_utc6_ee/source_region_merra_aer_daily_stack_{y}_{m:02d}.tif") as a, \
         rasterio.open(ROOT / f"data/raw/merra2_utc6_ee/source_region_merra_slv_daily_stack_{y}_{m:02d}.tif") as s, \
         rasterio.open(ROOT / f"data/raw/merra2_utc6_ee/source_region_merra_flx_daily_stack_{y}_{m:02d}.tif") as fx:
        srcs = [a, s, fx]
        for cid, dd in picks:
            row = tbl[(tbl.cell_id == cid) & (tbl.day == dd)].iloc[0]
            lat, lon = cells.loc[cid, "lat"], cells.loc[cid, "lon"]
            day = dt.date(y, m, dd)
            for v in VARS:
                key = f"{day:%Y%m%d}_{v}"
                for src in srcs:
                    if key in src.descriptions:
                        r_, c_ = src.index(lon, lat)
                        val = src.read(src.descriptions.index(key) + 1, window=((r_, r_ + 1), (c_, c_ + 1)))[0, 0]
                        res.append(("merra_" + v.lower(), y, m, cid, dd, float(val), float(row["merra_" + v.lower()])))
                        break
            f6 = fire[fire.d6 == day]
            dist = hav(lat, lon, f6.latitude.to_numpy(), f6.longitude.to_numpy())
            res.append(("fire_dist_nearest_km", y, m, cid, dd, float(dist.min()) if dist.size else np.nan, float(row.fire_dist_nearest_km)))
            for rr in (25, 50, 100, 250, 500):
                res.append((f"fire_count_{rr}km", y, m, cid, dd, float((dist <= rr).sum()), float(row[f"fire_count_{rr}km"])))
                res.append((f"fire_frp_sum_{rr}km", y, m, cid, dd, float(f6.frp.to_numpy()[dist <= rr].sum()), float(row[f"fire_frp_sum_{rr}km"])))
            # HMS
            start = dt.datetime.combine(day, dt.time(6)); end = start + dt.timedelta(days=1)
            if hms_polys(day) is None:
                res.append(("hms_n_polygons", y, m, cid, dd, np.nan, float(row.hms_n_polygons)))
                continue
            pt = shapely.Point(lon, lat); n = 0; dmax = np.nan
            for k in (-1, 0, 1):
                P = hms_polys(day + dt.timedelta(days=k)) or []
                for g, s_, e_, den in P:
                    if s_ < end and e_ >= start and g.contains(pt):
                        n += 1
                        val = {"light": 5.0, "medium": 16.0, "heavy": 27.0}.get(den, np.nan)
                        if not np.isnan(val):
                            dmax = val if np.isnan(dmax) else max(dmax, val)
            res.append(("hms_n_polygons", y, m, cid, dd, float(n), float(row.hms_n_polygons)))
            res.append(("hms_density_max", y, m, cid, dd, dmax, float(row.hms_density_max)))
    print(y, m, "done", flush=True)

R_ = pd.DataFrame(res, columns=["col", "y", "m", "cell_id", "day", "ref", "got"])
R_.to_csv(ROOT / "audit/evidence/a08_spot_sources.csv", index=False)
both_nan = R_.ref.isna() & R_.got.isna()
diff = (R_.ref - R_.got).abs()
tol = 1e-6 + 1e-5 * R_.ref.abs()
R_["ok"] = both_nan | (diff <= tol)
print(R_.groupby("col").ok.agg(["size", "sum"]).to_string())
print(R_[~R_.ok].to_string())
