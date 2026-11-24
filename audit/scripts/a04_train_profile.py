"""Full profile of the training table and its relation to station_day / spine."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(r"D:\ai-workspace\project")
t = pd.read_parquet(ROOT / "data/processed/target/target_table.parquet")
sd = pd.read_parquet(ROOT / "data/interim/stations/station_day.parquet")
st = pd.read_parquet(ROOT / "data/interim/stations/stations.parquet")
cells = pd.read_parquet(ROOT / "data/interim/spine/cells.parquet")
co = pd.read_parquet(ROOT / "data/processed/grid/cell_ontario.parquet")
P = print
P("rows", len(t), "cols", t.shape[1], "cells", t.cell_id.nunique())
P("dup keys", int(t.duplicated(["cell_id", "date"]).sum()),
  "sorted", bool((t[["date", "cell_id"]].apply(tuple, axis=1).is_monotonic_increasing)))
P("NaN per non-allowed col:", {c: int(t[c].isna().sum()) for c in
   ["pm25", "n_stations", "n_hours_min", "province_code", "province_state", "dist_ontario_km", "on_region",
    "merra_pblh", "fire_count_500km"] if t[c].isna().any()})
P("pm25 describe:\n", t.pm25.describe(percentiles=[.001, .01, .5, .99, .999, .9999]).round(3).to_string())
P("pm25 == 0 rows:", int((t.pm25 == 0).sum()), " <0:", int((t.pm25 < 0).sum()))
P("n_stations:", t.n_stations.value_counts().sort_index().to_dict())
P("n_hours_min:", t.n_hours_min.value_counts().sort_index().to_dict())
# reconciliation: station-day -> cell-day
s2 = sd[sd.pm25 >= 0]
cd = s2.groupby(["cell_id", "date"]).size()
P("cell-days from station_day (pm25>=0):", len(cd), " training rows:", len(t),
  " diff:", len(cd) - len(t))
# per-network station-days dropped as negative
sd["net"] = sd.station_uid.str.split(":").str[0]
P("negative station-days by network:", sd[sd.pm25 < 0].groupby("net").size().to_dict(),
  " min:", float(sd.pm25.min()))
# largest values: which station-days
top = sd.sort_values("pm25", ascending=False).head(15).merge(st[["station_uid", "name"]], on="station_uid")
P("top station-days:\n", top[["station_uid", "name", "date", "pm25", "n_valid_hours"]].to_string(index=False))
# outliers relative to cell's other stations on the same day
multi = s2.groupby(["cell_id", "date"]).pm25.agg(["size", "min", "max"])
multi = multi[multi["size"] > 1]
multi["spread"] = multi["max"] - multi["min"]
P("multi-station cell-days:", len(multi), " spread>50:", int((multi.spread > 50).sum()),
  " spread>20:", int((multi.spread > 20).sum()))
P(multi.sort_values("spread", ascending=False).head(8).to_string())
# region / province consistency
t2 = t.drop_duplicates("cell_id")[["cell_id", "province_state", "on_region", "dist_ontario_km"]]
x = t2[((t2.province_state == "Ontario") & (t2.on_region == 0)) | ((t2.province_state != "Ontario") & (t2.on_region > 0))]
P("cells where province_state and on_region disagree:\n", x.to_string(index=False))
# AOD coverage among training rows by year
t["aod_obs"] = t.aod055.notna()
P("training aod055 observed share by year:", t.groupby("year").aod_obs.mean().round(3).to_dict())
P("training rows south of 41N (maiac_covered=0):", int((t.maiac_covered == 0).sum()))
# per-year, per-network station counts (stations with >=1 day)
P("stations per year by network:\n", sd.groupby([sd.date.dt.year, "net"]).station_uid.nunique().unstack().to_string())
# annual mean pm25 per network
P("annual mean pm25 by network:\n", sd[sd.pm25 >= 0].groupby([sd.date.dt.year, "net"]).pm25.mean().unstack().round(2).to_string())
# monthly row counts: gaps (months with unusually few rows)
mc = t.groupby([t.year, t.month]).size()
P("min/max monthly training rows:", int(mc.min()), int(mc.max()), " months:", len(mc))
# calendar consistency
d = t.date
P("calendar mismatches:", int(((d.dt.year != t.year) | (d.dt.month != t.month) | (d.dt.dayofyear != t.doy) | (d.dt.dayofweek + 1 != t.dow)).sum()))
