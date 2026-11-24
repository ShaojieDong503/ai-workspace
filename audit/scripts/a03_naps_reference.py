"""Independent reference for NAPS station-day targets, from the raw annual CSVs.

From the specification (not from build_on_utc6.py / targets.py):
  * header line found by content; -999 = missing; every other value kept
  * duplicate station-local-days (two methods): keep most valid hours, tie -> 184
  * each hour h of local-standard date D covers [D+(h-1)h, D+h h) LST;
    UTC-6 day = floor(start_LST - offset - 6h); offset -5 east of 90W, -6 west
  * hours pooled across years; >= 18 valid hours; mean
  * station universe: EVERY NAPS station whose coordinates fall in a spine cell,
    whatever its province
"""
from pathlib import Path
import numpy as np, pandas as pd, pyproj

ROOT = Path(r"D:\ai-workspace\project")
OUT = ROOT / "audit/evidence"
X0, DX, Y0, DY, W = 3688478.20, 4999.827, 5239381.11, -4998.864, 1065
spine = pd.read_parquet(ROOT / "data/interim/spine/cells.parquet", columns=["cell_id", "province_state"])
spine_ids = set(spine.cell_id.tolist())
tr = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3347", always_xy=True)
H = [f"H{i:02d}" for i in range(1, 25)]

frames = []
for p in sorted((ROOT / "data/Target_Data_2010_2024").glob("PM25_*.csv")):
    lines = p.read_text(encoding="utf-8-sig").splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Pollutant//"))
    d = pd.read_csv(p, skiprows=hdr, encoding="utf-8-sig", dtype=str)
    d.columns = [c.split("//")[0].strip() for c in d.columns]
    d["file"] = p.name
    frames.append(d)
raw = pd.concat(frames, ignore_index=True)
raw["stn"] = raw["NAPS ID"].str.strip()
raw["lat"] = raw["Latitude"].astype(float); raw["lon"] = raw["Longitude"].astype(float)
raw["prov"] = raw["Province/Territory"].str.strip()
raw["method"] = raw["Method Code"].astype(int)
raw["date"] = pd.to_datetime(raw["Date"].str.strip(), format="%Y-%m-%d")
vals = raw[H].apply(pd.to_numeric, errors="coerce")
# sentinel census before replacing
neg = vals[(vals < 0) & (vals != -999)].stack()
print("raw rows", len(raw), "; negative non-sentinel hourly values:", len(neg),
      "; distinct:", sorted(neg.unique())[:20])
print("non-numeric hourly cells:", int(raw[H].notna().sum().sum() - vals.notna().sum().sum()))
vals = vals.where(vals != -999)
raw[H] = vals
raw["nv"] = vals.notna().sum(axis=1)

# station coordinates -> cell, per distinct coordinate (relocations would show here)
co = raw.groupby(["stn", "lat", "lon", "prov"], as_index=False).agg(rows=("date", "size"),
                                                                   first=("date", "min"), last=("date", "max"))
x, y = tr.transform(co.lon.to_numpy(), co.lat.to_numpy())
co["cell_id"] = np.rint((Y0 - y) / -DY).astype("int64") * W + np.rint((x - X0) / DX).astype("int64")
co["in_spine"] = co.cell_id.isin(spine_ids)
co.to_csv(OUT / "naps_ref_coords.csv", index=False)
multi = co.groupby("stn").cell_id.nunique()
print("stations with >1 cell across their coordinates:", list(multi[multi > 1].index))
keep_stn = set(co.loc[co.in_spine, "stn"])
print("NAPS stations with any coordinate in the spine:", len(keep_stn),
      co[co.in_spine].drop_duplicates("stn").prov.value_counts().to_dict())

r = raw[raw.stn.isin(keep_stn)].copy()
# duplicate local days: most valid hours, then method 184, then lowest method code
r["pri"] = (r["method"] != 184).astype(int)
r = r.sort_values(["stn", "date", "nv", "pri", "method"], ascending=[True, True, False, True, True])
ndup = int(r.duplicated(["stn", "date"]).sum())
r = r.drop_duplicates(["stn", "date"], keep="first")
print("duplicate station-local-days removed:", ndup)

long = r.melt(id_vars=["stn", "date", "lon", "prov"], value_vars=H, var_name="h", value_name="v").dropna(subset=["v"])
long["h"] = long["h"].str[1:].astype(int)
off = np.where(long["lon"] > -90.0, -5, -6)
start_lst = long["date"] + pd.to_timedelta(long["h"] - 1, unit="h")
long["day"] = (start_lst - pd.to_timedelta(off + 6, unit="h")).dt.floor("D")
g = long.groupby(["stn", "day"], as_index=False).agg(pm25=("v", "mean"), n=("v", "size"),
                                                    n_neg=("v", lambda s: int((s < 0).sum())),
                                                    prov=("prov", "first"))
nn = long[long.v >= 0].groupby(["stn", "day"]).agg(m_nn=("v", "mean"), n_nn=("v", "size"))
g = g.merge(nn, on=["stn", "day"], how="left")
g["pm25_negmissing"] = g["m_nn"].where(g["n_nn"] >= 18)
g = g[(g.n >= 18) & (g.day >= "2010-01-01") & (g.day <= "2024-12-31")]
g.drop(columns=["m_nn"]).to_parquet(OUT / "naps_ref_station_day.parquet", index=False)
print("reference NAPS station-days:", len(g), g.prov.value_counts().to_dict())
