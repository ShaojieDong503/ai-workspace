"""Summarise the full-scan profile of the prediction table."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(r"D:\ai-workspace\project")
p = pd.read_csv(ROOT / "audit/evidence/pred_profile.csv")
p["y"] = p.ym.str[:4].astype(int); p["m"] = p.ym.str[5:].astype(int)
p["nd"] = [pd.Period(f"{y}-{m:02d}").days_in_month for y, m in zip(p.y, p.m)]
print("months", len(p), " total rows", int(p.n_rows.sum()))
print("rows != days*163787:", p[p.n_rows != p.nd * 163787].ym.tolist(),
      " n_cells != 163787:", p[p.n_cells != 163787].ym.tolist(), " n_days != nd:", p[p.n_days != p.nd].ym.tolist())
cols = sorted({c.rsplit("__", 1)[0] for c in p.columns if "__" in c})
rows = []
for c in cols:
    rows.append(dict(col=c, nan=int(p[c + "__nan"].sum()), nan_share=p[c + "__nan"].sum() / p.n_rows.sum(),
                     zero=int(p[c + "__zero"].sum()), neg=int(p[c + "__neg"].sum()),
                     min=p[c + "__min"].min(), max=p[c + "__max"].max(),
                     months_all_nan=int((p[c + "__nan"] == p.n_rows).sum())))
S = pd.DataFrame(rows).set_index("col")
pd.set_option("display.width", 200)
print(S.to_string(float_format=lambda v: f"{v:.4g}"))
S.to_csv(ROOT / "audit/evidence/pred_column_summary.csv")
# month-level missing share of aod055 among covered cells is not separable here; show NaN share per month for key cols
for c in ["aod055", "hms_any", "fire_dist_nearest_km", "aod_last_obs"]:
    s = (p[c + "__nan"] / p.n_rows).round(3)
    print(c, "NaN share by month: min", s.min(), "max", s.max(), " top:", dict(zip(p.ym[s.nlargest(4).index], s.nlargest(4))))
print("months with any hms NaN:", p[p.hms_any__nan > 0].ym.tolist())
print("months with injection_height < 0:", dict(zip(p[p.injection_height__neg > 0].ym, p[p.injection_height__neg > 0].injection_height__neg)))
print("fire_count_500km max by month top:", p.nlargest(5, "fire_count_500km__max")[["ym", "fire_count_500km__max"]].values.tolist())
