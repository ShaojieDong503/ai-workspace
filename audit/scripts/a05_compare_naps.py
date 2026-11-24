"""Compare the independent NAPS reference with station_day.parquet row by row."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(r"D:\ai-workspace\project"); EV = ROOT / "audit/evidence"
ref = pd.read_parquet(EV / "naps_ref_station_day.parquet")
ref["station_uid"] = "NAPS:" + ref.stn
sd = pd.read_parquet(ROOT / "data/interim/stations/station_day.parquet")
sd = sd[sd.station_uid.str.startswith("NAPS:")]
m = ref.merge(sd, left_on=["station_uid", "day"], right_on=["station_uid", "date"], how="outer", indicator=True)
print(m["_merge"].value_counts().to_dict())
b = m[m._merge == "both"].copy()
b["d"] = (b.pm25_x - b.pm25_y).abs()
b["tol"] = np.where(b.prov == "ON", 0.005 + 1e-6, 1e-4 + 1e-6 * b.pm25_x.abs())
bad = b[b.d > b.tol]
print("value mismatches:", len(bad), " by prov:", bad.prov.value_counts().to_dict())
print(bad.sort_values("d", ascending=False).head(10)[["station_uid", "day", "pm25_x", "pm25_y", "n", "n_valid_hours"]].to_string())
print("n_valid_hours mismatches:", int((b.n != b.n_valid_hours).sum()))
print("max |diff| ON:", float(b[b.prov == "ON"].d.max()), " QC/MB:", float(b[b.prov != "ON"].d.max()))
# negative hours inside kept days
print("station-days with >=1 negative hour:", int((b.n_neg > 0).sum()),
      " of which daily mean still >= 0:", int(((b.n_neg > 0) & (b.pm25_x >= 0)).sum()))
nm = b[b.pm25_negmissing.notna()]
print("days lost if negative hours were missing (<18 left):", int(b.pm25_negmissing.isna().sum()))
dd = (nm.pm25_negmissing - nm.pm25_x)
print("shift in daily mean when negative hours are missing: mean", round(float(dd.mean()), 4),
      " max", round(float(dd.max()), 3), " days changed by >0.5:", int((dd > 0.5).sum()))
only = m[m._merge != "both"]
print(only.head(10).to_string())
