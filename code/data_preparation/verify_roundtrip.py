#!/usr/bin/env python3
"""Independent verification of the Ontario UTC-6 outputs.

Rebuilds the source station-day grid, inverts the shift on every output
cell, and proves the round trip is exact.  Nothing here reuses the build's
intermediate state: it re-reads the raw CSVs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_on_utc6 as b  # noqa: E402

DATA = Path(r"D:\ai-workspace\project\data\Target_Data_2010_2024")
OUT = Path(r"D:\ai-workspace\project\data\processed\on_utc6")
HOURS = b.HOURS
EXPECTED_HEADER = (
    ["Pollutant//Polluant", "Method Code//Code Méthode", "NAPS ID//Identifiant SNPA",
     "City//Ville", "Province/Territory//Province/Territoire", "Latitude//Latitude",
     "Longitude//Longitude", "Date//Date"] + [f"{h}//{h}" for h in HOURS]
)
problems: list[str] = []


# Tiny assertion helper: print PASS/FAIL and collect failures.
def check(cond: bool, msg: str) -> None:
    if not cond:
        problems.append(msg)
    print(("PASS  " if cond else "FAIL  ") + msg)


# ---------- source side ----------
# Rebuild the source grid from the raw CSVs, without touching the build's
# intermediate state.
frames = []
for p in sorted(DATA.glob("PM25_*.csv")):
    d = b.read_year(p)
    frames.append(d[d["Province/Territory"].str.strip().str.upper() == "ON"])
src = b.clean(pd.concat(frames, ignore_index=True))
src, _ = b.dedupe(src, OUT)

src_long = src.melt(id_vars=["stn", "Date"], value_vars=HOURS,
                    var_name="slot", value_name="pm25")
src_long["h"] = src_long["slot"].str[1:].astype(int)
src_long = src_long[["stn", "Date", "h", "pm25"]].rename(columns={"Date": "local_date"})
src_long["local_date"] = pd.to_datetime(src_long["local_date"])

# ---------- output side ----------
offsets = pd.read_csv(OUT / "station_offsets.csv", encoding="utf-8-sig", dtype={"stn": str})
offset_map = dict(zip(offsets["stn"].astype(str), offsets["offset_hours"].astype(int)))
shift_map = {k: -(v + 6) for k, v in offset_map.items()}

# Read each output file; the bilingual header is checked literally so a
# renamed column cannot slip through.
out_rows, header_ok, years_seen = [], True, []
# Read every year file back. The header is compared literally, not by column
# position, so a renamed or reordered column is caught.
for path in sorted(OUT.glob("ON_PM25_UTC6_*.csv")):
    year = int(path.stem.split("_")[-1])
    years_seen.append(year)
    raw = path.read_text(encoding="utf-8-sig")
    lines = raw.split("\n")
    header = lines[7].split(",")
    if header != EXPECTED_HEADER:
        header_ok = False
        print(f"  header mismatch in {path.name}: {header[:3]}...")
    d = pd.read_csv(path, skiprows=7, encoding="utf-8-sig", dtype=str)
    d.columns = [c.split("//")[0].strip() for c in d.columns]
    d["year_file"] = year
    out_rows.append(d)

check(header_ok, "bilingual header identical in every output file")
check(years_seen == list(range(2010, 2025)), f"15 output files, years {years_seen[0]}-{years_seen[-1]}")

out = pd.concat(out_rows, ignore_index=True)
check(out["Province/Territory"].eq("ON").all(), "every output row is Province/Territory = ON")
n_stations = out["NAPS ID"].nunique()
check(n_stations == 59, f"59 distinct stations in output (found {n_stations})")
check(out["Pollutant"].eq("PM2.5").all(), "pollutant column is PM2.5 for every row")
check(pd.to_numeric(out["Method Code"], errors="coerce").isin([184, 236, 703, 636, 170]).all(),
      "Method Code values are sane")

out["Date"] = pd.to_datetime(out["Date"], format="%Y-%m-%d")
check((out["Date"].dt.year == out["year_file"]).all(), "each row lands in its own year file")

vals = out[HOURS].apply(pd.to_numeric, errors="raise").to_numpy()
non_sentinel = vals[vals != b.NO_DATA]
check(bool(np.all(non_sentinel % 1 == 0)), "all non-sentinel values are integers")
check(bool((non_sentinel >= -50).all() and (non_sentinel <= 2000).all()),
      f"values in plausible range ({non_sentinel.min()} .. {non_sentinel.max()})")
check(not out.duplicated(subset=["NAPS ID", "Date"]).any(), "no duplicate station-day rows")

# ---------- invert the shift ----------
# Back to long rows so every single cell can be matched one by one.
long = out.melt(id_vars=["NAPS ID", "Date"], value_vars=HOURS, var_name="slot", value_name="pm25")
long["h"] = long["slot"].str[1:].astype(int)
long["shift"] = long["NAPS ID"].map(shift_map)
check(not long["shift"].isna().any(), "every station has an offset")
check(set(long["shift"].unique()) == {-1}, "all stations shift by exactly -1 h")

# Invert the build's shift: target clock back to local standard time.
ts_target = long["Date"] + pd.to_timedelta(long["h"], unit="h")
# Undo the shift: target = local + shift, therefore local = target - shift.
ts_local = ts_target - pd.to_timedelta(long["shift"], unit="h")   # undoing target = LST + shift
local_date = ts_local.dt.normalize().astype("datetime64[ns]")
hour = ts_local.dt.hour
# 00:00 belongs to the previous day as H24, mirroring the build.
roll = hour == 0
local_date = local_date.where(~roll, local_date - pd.Timedelta(days=1))
hour = hour.where(~roll, 24)

long["local_date"] = pd.to_datetime(local_date)
long["h_local"] = hour.astype(int)
long["pm25_out"] = pd.to_numeric(long["pm25"], errors="raise").replace(b.NO_DATA, np.nan)

# Outer join so both missing and surplus cells are visible, not just matches.
merged = src_long.merge(long[["NAPS ID", "local_date", "h_local", "pm25_out"]],
                        left_on=["stn", "local_date", "h"],
                        right_on=["NAPS ID", "local_date", "h_local"],
                        how="outer", indicator=True)

right_only = merged[merged["_merge"] == "right_only"]
check(int(right_only["pm25_out"].notna().sum()) == 0,
      f"no output value lacks a source origin ({len(right_only):,} NaN-only cells from "
      "trailing H24 and one borrowed hour)")

missing = merged[merged["_merge"] == "left_only"]
first_day = src.groupby("stn")["Date"].transform("min") if False else None
first_map = src.groupby("stn")["Date"].min()
orphan_expected = {(s, d, 1) for s, d in first_map.items()}
orphan_found = {(r["stn"], r["local_date"], int(r["h"])) for _, r in missing.iterrows()}
check(orphan_found == orphan_expected,
      f"exactly the {len(orphan_expected)} orphan H01 cells are absent "
      f"(found {len(orphan_found)}, matching={orphan_found == orphan_expected})")

both = merged[merged["_merge"] == "both"]
a = both["pm25"].to_numpy(dtype=float)
c = both["pm25_out"].to_numpy(dtype=float)
same = np.isclose(a, c, equal_nan=True).all()
check(bool(same), f"all {len(both):,} reconstructed cells match the source exactly")

print()
print(f"source cells        : {len(src_long):,}")
print(f"output cells        : {len(long):,}")
print(f"matched cells       : {len(both):,}")
print(f"orphan cells (drop) : {len(missing):,}")
print()
if problems:
    print("FAILURES:")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("ROUND TRIP: PASS")



