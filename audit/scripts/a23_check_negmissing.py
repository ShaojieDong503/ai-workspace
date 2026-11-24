"""Check the rebuilt station_day (negative hour = missing) against the audit's
independent raw recomputation (a02 AQS, a03 NAPS), row by row.

Expected values come only from the audit references:
  AQS : per (site, POC, UTC-6 day) the non-negative sum s_nn and count n_nn,
        pooled across annual files; POC kept if n_nn >= 18; site = mean of POCs.
  NAPS: n_nn >= 18, value pm25_negmissing.
"""
import pandas as pd
import numpy as np

EV = r"D:\ai-workspace\project\audit\evidence"
SD = r"D:\ai-workspace\project\data\interim\stations\station_day.parquet"
OLD = r"D:\ai-workspace\project\data\interim\stations\_before_negmissing_20261108\station_day.parquet"

sd = pd.read_parquet(SD)
sd["date"] = pd.to_datetime(sd["date"])
print("rows", len(sd), " negative pm25:", int((sd.pm25 < 0).sum()),
      " min hours:", int(sd.n_valid_hours.min()), " dup keys:", int(sd.duplicated(["station_uid", "date"]).sum()))

# ---- AQS -------------------------------------------------------------------
p = pd.read_parquet(f"{EV}/aqs_ref_site_poc_day.parquet")
p = p[p.n_nn >= 18].copy()
p["v"] = p.s_nn / p.n_nn
ea = (p.groupby(["site", "day"]).agg(pm25=("v", "mean"), n_poc=("poc", "size"), nh=("n_nn", "min"))
        .reset_index())
ea["station_uid"] = "AQS:" + ea.site.astype(str)
ea = ea.rename(columns={"day": "date"})
ea["date"] = pd.to_datetime(ea["date"])
ea = ea[ea.date.between("2010-01-01", "2024-12-31")]

na = pd.read_parquet(f"{EV}/naps_ref_station_day.parquet")
na = na[na.n_nn >= 18]
en = pd.DataFrame({"station_uid": "NAPS:" + na.stn.astype(str), "date": pd.to_datetime(na.day),
                   "pm25": na.pm25_negmissing, "n_poc": 1, "nh": na.n_nn, "prov": na.prov})

for name, exp, pre in [("AQS", ea, "AQS:"), ("NAPS", en, "NAPS:")]:
    got = sd[sd.station_uid.str.startswith(pre)]
    exp = exp[exp.station_uid.isin(set(got.station_uid))]
    m = got.merge(exp, on=["station_uid", "date"], how="outer", suffixes=("", "_ref"), indicator=True)
    only = m._merge.value_counts().to_dict()
    b = m[m._merge == "both"]
    d = (b.pm25 - b.pm25_ref).abs()
    print(f"\n{name}: pipeline {len(got):,}  reference {len(exp):,}  merge {only}")
    print(f"  |diff| max {d.max():.2e}  >1e-3: {int((d > 1e-3).sum())}")
    print(f"  n_valid_hours mismatches: {int((b.n_valid_hours != b.nh).sum())}  "
          f"n_poc mismatches: {int((b.n_poc != b.n_poc_ref).sum())}")
    if (d > 1e-3).any():
        print(b.loc[d > 1e-3, ["station_uid", "date", "pm25", "pm25_ref", "n_valid_hours", "nh"]].head(10).to_string())
    dec31 = b[(b.date.dt.month == 12) & (b.date.dt.day == 31)]
    print(f"  31-Dec rows: {len(dec31):,}  max |diff| {(dec31.pm25 - dec31.pm25_ref).abs().max():.2e}")

# ---- before / after ----------------------------------------------------------
old = pd.read_parquet(OLD); old["date"] = pd.to_datetime(old["date"])
net = lambda s: s.str.split(":").str[0]
print("\nbefore -> after station-days by network")
print(pd.DataFrame({"before": old.groupby(net(old.station_uid)).size(),
                    "before_nonneg": old[old.pm25 >= 0].groupby(net(old.station_uid)).size(),
                    "after": sd.groupby(net(sd.station_uid)).size()}).to_string())
j = old.merge(sd, on=["station_uid", "date"], suffixes=("_old", "_new"))
sh = j.pm25_new - j.pm25_old
print(f"common rows {len(j):,}; mean shift {sh.mean():+.4f}; >0.5 shift {int((sh > 0.5).sum()):,}; "
      f"new rows not in old: {len(sd) - len(j)}")
print("mean pm25 before (nonneg) %.3f  after %.3f" % (old[old.pm25 >= 0].pm25.mean(), sd.pm25.mean()))
