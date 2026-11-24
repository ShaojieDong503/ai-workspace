"""Compare the independent AQS reference with station_day.parquet row by row."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(r"D:\ai-workspace\project"); EV = ROOT / "audit/evidence"
ref = pd.read_parquet(EV / "aqs_ref_station_day.parquet")
ref = ref[ref.pm25.notna()]
ref["station_uid"] = "AQS:" + ref.site
sd = pd.read_parquet(ROOT / "data/interim/stations/station_day.parquet")
sd = sd[sd.station_uid.str.startswith("AQS:")]
st = pd.read_parquet(ROOT / "data/interim/stations/stations.parquet")
sites = pd.read_csv(EV / "aqs_ref_sites.csv")
sites["station_uid"] = "AQS:" + sites.site
x = sites.merge(st[["station_uid", "cell_id"]], on="station_uid", how="outer", indicator=True)
print("site universe:", x._merge.value_counts().to_dict(), " cell mismatches:", int((x.cell_id_x != x.cell_id_y).sum()),
      " sites spanning >1 cell:", int((sites.n_cells > 1).sum()))
off = pd.read_csv(EV / "aqs_ref_offsets.csv"); off["station_uid"] = "AQS:" + off.site
m = ref.merge(sd, left_on=["station_uid", "day"], right_on=["station_uid", "date"], how="outer", indicator=True)
print(m["_merge"].value_counts().to_dict())
b = m[m._merge == "both"].copy()
b["d"] = (b.pm25_x - b.pm25_y).abs()
bad = b[b.d > 1e-4 + 1e-6 * b.pm25_x.abs()]
bad = bad.merge(off[["station_uid", "min"]], on="station_uid")
bad["md"] = bad.day.dt.strftime("%m-%d")
print("value mismatches:", len(bad), " by calendar day:", bad.md.value_counts().head(5).to_dict(),
      " by site UTC offset:", bad["min"].value_counts().to_dict())
print("n_valid_hours mismatch (ref n_min vs pipeline):", int((b.n_min != b.n_valid_hours).sum()))
nb = b[b.n_min != b.n_valid_hours]
print("  of which on 12-31:", int((nb.day.dt.strftime('%m-%d') == '12-31').sum()))
print(bad.sort_values("d", ascending=False).head(8)[["station_uid", "day", "pm25_x", "pm25_y", "n_min", "n_valid_hours"]].to_string())
print("n_poc mismatches:", int((b.n_poc_x != b.n_poc_y).sum()))
only = m[m._merge != "both"]
print("rows only on one side:\n", only[["station_uid", "day", "date", "pm25_x", "pm25_y", "n_min", "_merge"]].head(20).to_string())
# negative hours
print("AQS station-days with >=1 negative hour:", int((b.n_neg > 0).sum()), "of", len(b),
      "; daily mean still >=0:", int(((b.n_neg > 0) & (b.pm25_x >= 0)).sum()))
print("AQS days lost if negative hours were missing:", int(b.pm25_negmissing.isna().sum()))
dd = b.pm25_negmissing - b.pm25_x
print("shift when negative hours missing: mean", round(float(dd.mean()), 4), " p99", round(float(dd.quantile(.99)), 3),
      " max", round(float(dd.max()), 3), " days changed >0.5:", int((dd > 0.5).sum()))
print("AQS station-days containing qualifier-flagged hours:", int((b.n_qual > 0).sum()))
# multi-POC disagreement
P = pd.read_parquet(EV / "aqs_ref_site_poc_day.parquet")
P = P[P.n >= 18]; P["m"] = P.s / P.n
g = P.groupby(["site", "day"]).m.agg(["size", "min", "max"])
g = g[g["size"] > 1]
print("multi-POC site-days:", len(g), " max-min > 10:", int(((g["max"] - g["min"]) > 10).sum()),
      " > 25:", int(((g["max"] - g["min"]) > 25).sum()))
print(g.assign(spread=g["max"] - g["min"]).sort_values("spread", ascending=False).head(5).to_string())
