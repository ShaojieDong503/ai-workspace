"""Independent reference for the AQS station-day targets.

Written from the specification, not from targets.py:
  * hour-start GMT timestamp -> UTC-6 day = floor(ts - 6 h)
  * hours are combined ACROSS annual files (a UTC-6 day can straddle two files)
  * per (site, POC, day): valid hours = distinct hours with a value; >= 18 to pass
  * site-day = mean over passing POCs
Also records per station-day: negative hourly values, qualifier-flagged hours,
duplicate hourly rows. Streams the zips with pyarrow's incremental reader.
"""
import zipfile, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.csv as pv, pyproj

ROOT = Path(r"D:\ai-workspace\project")
AQS = ROOT / "data/raw/AQS"
OUT = ROOT / "audit/evidence"
X0, DX, Y0, DY, W = 3688478.20, 4999.827, 5239381.11, -4998.864, 1065
spine = pd.read_parquet(ROOT / "data/interim/spine/cells.parquet", columns=["cell_id"])
spine_ids = set(spine.cell_id.tolist())
tr = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3347", always_xy=True)

cols = ["State Code", "County Code", "Site Num", "POC", "Latitude", "Longitude",
        "Date Local", "Time Local", "Date GMT", "Time GMT", "Sample Measurement", "Qualifier"]
types = {c: pa.string() for c in ["State Code", "County Code", "Site Num", "Qualifier",
                                   "Date Local", "Time Local", "Date GMT", "Time GMT"]}
partials, sites, dup_rows, offsets = [], [], 0, []
for z in sorted(AQS.glob("hourly_88101_*.zip")):
    with zipfile.ZipFile(z) as zf:
        name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(name) as fh:
            rdr = pv.open_csv(fh, read_options=pv.ReadOptions(block_size=128 << 20),
                              convert_options=pv.ConvertOptions(include_columns=cols, column_types=types))
            chunks = []
            for b in rdr:
                d = b.to_pandas()
                d["site"] = d["State Code"] + "-" + d["County Code"] + "-" + d["Site Num"]
                # keep only sites whose coordinates fall in a spine cell (own mapping)
                x, y = tr.transform(d["Longitude"].to_numpy(), d["Latitude"].to_numpy())
                cid = np.rint((Y0 - y) / -DY).astype("int64") * W + np.rint((x - X0) / DX).astype("int64")
                d["cell_id"] = cid
                d = d[d["cell_id"].isin(spine_ids)]
                chunks.append(d)
    d = pd.concat(chunks, ignore_index=True)
    sites.append(d.groupby("site").agg(lat=("Latitude", "first"), lon=("Longitude", "first"),
                                       cell_id=("cell_id", "first"), n_cells=("cell_id", "nunique")))
    gmt = pd.to_datetime(d["Date GMT"] + " " + d["Time GMT"], format="%Y-%m-%d %H:%M")
    loc = pd.to_datetime(d["Date Local"] + " " + d["Time Local"], format="%Y-%m-%d %H:%M")
    d["off_h"] = ((loc - gmt) / pd.Timedelta(hours=1)).astype("int8")
    offsets.append(d.groupby("site")["off_h"].agg(["min", "max"]))
    d["gmt"] = gmt
    d["v"] = pd.to_numeric(d["Sample Measurement"], errors="coerce")
    d["poc"] = d["POC"].astype(int)
    dup = d.duplicated(["site", "poc", "gmt"], keep=False)
    dup_rows += int(dup.sum())
    d["qual"] = d["Qualifier"].notna() & (d["Qualifier"].str.len() > 0)
    # one value per (site, poc, hour): mean of duplicate rows, if any
    h = d.groupby(["site", "poc", "gmt"], as_index=False).agg(v=("v", "mean"), qual=("qual", "max"))
    h = h[h["v"].notna()]
    h["day"] = (h["gmt"] - pd.Timedelta(hours=6)).dt.floor("D")
    h["neg"] = h["v"] < 0
    h["v_nn"] = h["v"].where(~h["neg"])
    p = h.groupby(["site", "poc", "day"], as_index=False).agg(
        s=("v", "sum"), n=("v", "size"), n_neg=("neg", "sum"), n_qual=("qual", "sum"),
        s_nn=("v_nn", "sum"), n_nn=("v_nn", "count"))
    partials.append(p)
    print(z.name, len(d), "rows in spine sites;", len(p), "partial site-poc-days; dup hour rows so far", dup_rows, flush=True)

P = pd.concat(partials).groupby(["site", "poc", "day"], as_index=False).sum()
P.to_parquet(OUT / "aqs_ref_site_poc_day.parquet", index=False)
pass_ = P[P["n"] >= 18].copy()
pass_["m"] = pass_["s"] / pass_["n"]
sd = pass_.groupby(["site", "day"], as_index=False).agg(pm25=("m", "mean"), n_poc=("poc", "size"),
                                                       n_min=("n", "min"), n_neg=("n_neg", "sum"),
                                                       n_qual=("n_qual", "sum"))
# variant: negative hours treated as missing (professor's rule)
v2 = P[P["n_nn"] >= 18].copy()
v2["m"] = v2["s_nn"] / v2["n_nn"]
v2 = v2.groupby(["site", "day"], as_index=False).agg(pm25_negmissing=("m", "mean"))
sd = sd.merge(v2, on=["site", "day"], how="outer")
sd = sd[(sd["day"] >= "2010-01-01") & (sd["day"] <= "2024-12-31")]
sd.to_parquet(OUT / "aqs_ref_station_day.parquet", index=False)
S = pd.concat(sites).groupby(level=0).agg(lat=("lat", "first"), lon=("lon", "first"),
                                          cell_id=("cell_id", "first"), n_cells=("n_cells", "max"))
S.to_csv(OUT / "aqs_ref_sites.csv")
O = pd.concat(offsets).groupby(level=0).agg(min=("min", "min"), max=("max", "max"))
O.to_csv(OUT / "aqs_ref_offsets.csv")
print("sites", len(S), "station-days", len(sd), "dup hour rows", dup_rows)
