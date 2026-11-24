"""Training table after the negative-hour rule: targets recomputed from the audit
references (not from station_day), features compared with the previous table."""
import pandas as pd, numpy as np
EV = r"D:\ai-workspace\project\audit\evidence"
T = r"D:\ai-workspace\project\data\processed\target\target_table.parquet"
OLD = r"D:\ai-workspace\project\data\processed\target\_before_negmissing_20261108\target_table.parquet"
ST = r"D:\ai-workspace\project\data\interim\stations\stations.parquet"

t = pd.read_parquet(T); old = pd.read_parquet(OLD)
st = pd.read_parquet(ST)[["station_uid", "cell_id"]]
p = pd.read_parquet(f"{EV}/aqs_ref_site_poc_day.parquet"); p = p[p.n_nn >= 18]; p["v"] = p.s_nn / p.n_nn
a = p.groupby(["site", "day"]).agg(pm25=("v", "mean"), nh=("n_nn", "min")).reset_index()
a["station_uid"] = "AQS:" + a.site.astype(str)
n = pd.read_parquet(f"{EV}/naps_ref_station_day.parquet"); n = n[n.n_nn >= 18]
n = pd.DataFrame({"station_uid": "NAPS:" + n.stn.astype(str), "day": n.day, "pm25": n.pm25_negmissing, "nh": n.n_nn})
r = pd.concat([a[["station_uid", "day", "pm25", "nh"]], n]).merge(st, on="station_uid")
r = r[pd.to_datetime(r.day).between("2010-01-01", "2024-12-31")]
e = r.groupby(["cell_id", "day"]).agg(pm25=("pm25", "mean"), ns=("pm25", "size"), nh=("nh", "min")).reset_index()
e["date"] = pd.to_datetime(e.day)
m = t[["cell_id", "date", "pm25", "n_stations", "n_hours_min"]].merge(e, on=["cell_id", "date"], how="outer", indicator=True, suffixes=("", "_ref"))
print("training rows", len(t), " reference cell-days", len(e), m._merge.value_counts().to_dict())
b = m[m._merge == "both"]
print("pm25 |diff| max %.2e" % (b.pm25 - b.pm25_ref).abs().max(),
      " n_stations mism", int((b.n_stations != b.ns).sum()), " n_hours_min mism", int((b.n_hours_min != b.nh).sum()))
print("pm25 min", float(t.pm25.min()), " max", float(t.pm25.max()), " zeros", int((t.pm25 == 0).sum()), " median", float(t.pm25.median()))

# features: every non-target column identical to the previous table on common rows
feat = [c for c in t.columns if c not in ("pm25", "n_stations", "n_hours_min")]
j = old.merge(t, on=["cell_id", "date"], suffixes=("_o", "_n"))
bad = [c for c in feat if c not in ("cell_id", "date") and not j[c + "_o"].equals(j[c + "_n"])]
print("common rows", len(j), " feature columns differing:", bad, " rows only in new:", len(t) - len(j), " only in old:", len(old) - len(j))
print("columns same:", list(old.columns) == list(t.columns))

# README numbers
print("per year:", " · ".join(f"{y} {c:,}" for y, c in t.groupby(t.date.dt.year).size().items()))
print("n_stations:", t.n_stations.value_counts().sort_index().to_dict())
g = t.groupby("on_region").agg(cells=("cell_id", "nunique"), rows=("cell_id", "size"))
print(g.to_string()); print("ON total rows", int(g.loc[g.index > 0, "rows"].sum()))
for km in (100, 200, 300):
    s = t[t.dist_ontario_km <= km]; print(km, f"{len(s):,}", s.cell_id.nunique())
