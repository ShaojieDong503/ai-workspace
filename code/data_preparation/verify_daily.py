import pandas as pd, numpy as np
from pathlib import Path
H=Path(r"D:\ai-workspace\project\data\processed\on_utc6")
D=Path(r"D:\ai-workspace\project\data\processed\on_utc6_daily")
HOURS=[f"H{h:02d}" for h in range(1,25)]
problems=[]
# Assertion helper: print PASS/FAIL and remember failures.
def check(c,m):
    print(("PASS  " if c else "FAIL  ")+m)
    if not c: problems.append(m)

# Accumulate source and kept row counts across all years.
tot_in=tot_kept=0
# Walk the hourly product one year at a time and recompute from scratch.
for p in sorted(H.glob("ON_PM25_UTC6_*.csv")):
    y=int(p.stem.split("_")[-1])
    h=pd.read_csv(p,skiprows=7,encoding="utf-8-sig",dtype=str)
    h.columns=[c.split("//")[0].strip() for c in h.columns]
    hv=h[HOURS].apply(pd.to_numeric,errors="raise").replace(-999,np.nan)
    nv=hv.notna().sum(axis=1); mean=hv.mean(axis=1)
    d=pd.read_csv(D/f"ON_PM25_UTC6_daily_{y}.csv",skiprows=7,encoding="utf-8-sig",dtype=str)
    d.columns=[c.split("//")[0].strip() for c in d.columns]
    d["nv"]=d["n_valid_hours"].astype(int); d["m"]=d["PM25_daily"].astype(float)
    tot_in+=len(h); tot_kept+=len(d)
    # The rule must hold exactly: rows kept == rows with at least 18 valid hours.
    check(len(d)==int((nv>=18).sum()), f"{y}: kept rows == count(n_valid>=18) ({len(d)} vs {int((nv>=18).sum())})")
    check(bool((d["nv"]>=18).all()), f"{y}: every kept row has >=18 valid hours (min {d['nv'].min()})")
    # join on station+date and compare means
    # Join back on station+date and recompute each mean independently of the builder.
    j=h[["NAPS ID","Date"]].assign(mean=mean.round(2), nv=nv).merge(
        d[["NAPS ID","Date","m","nv"]], on=["NAPS ID","Date"], suffixes=("_src","_out"), how="left")
    kept=j[j["nv_out"].notna()]
    check(len(kept)==len(d), f"{y}: every daily row matched back to an hourly row")
    check(bool(np.allclose(kept["mean"], kept["m"], atol=0.006)), f"{y}: recomputed means match ({len(kept)} rows)")
    check(bool((kept["nv_src"]==kept["nv_out"]).all()), f"{y}: n_valid preserved")

print()
print("hourly rows:", f"{tot_in:,}", " daily rows:", f"{tot_kept:,}")
# threshold sanity
import re
daily_files=[p for p in sorted(D.glob("ON_PM25_UTC6_daily_*.csv")) if re.fullmatch(r"ON_PM25_UTC6_daily_\d{4}\.csv", p.name)]
# Combine the per-year files for the cross-year summary below.
allkept=pd.concat([pd.read_csv(p,skiprows=7,encoding="utf-8-sig",dtype=str).rename(columns=lambda c: c.split("//")[0].strip()) for p in daily_files])
print("min n_valid across all kept rows:", allkept["n_valid_hours"].astype(int).min())
print("max n_valid:", allkept["n_valid_hours"].astype(int).max())
print("rows with exactly 18:", int((allkept["n_valid_hours"].astype(int)==18).sum()))
print("total kept:", len(allkept), "stations:", allkept["NAPS ID"].nunique())
print("daily rows with negative mean:", int((allkept["PM25_daily"].astype(float)<0).sum()))
print("date range:", allkept["Date"].min(), "..", allkept["Date"].max())
print()
print("Toronto 060430 2024-01-01 ->", allkept[(allkept["NAPS ID"]=="060430")&(allkept["Date"]=="2024-01-01")][["Date","PM25_daily","n_valid_hours"]].to_string(index=False))
if problems: print("\nFAILURES:", problems); raise SystemExit(1)
print("\nDAILY VERIFY: PASS")

