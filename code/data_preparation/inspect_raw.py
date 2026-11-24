"""Inventory the raw NAPS PM2.5 CSV exports before any transformation."""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

# Raw exports sit two levels above this script (project/data/Target_Data_2010_2024).
DATA = Path(__file__).resolve().parents[2] / "data" / "Target_Data_2010_2024"
HOURS = [f"H{h:02d}" for h in range(1, 25)]

# Read every annual file, skipping the 6-line preamble each one carries.
frames = []
# One file per year; the year is taken from the filename, not from the rows.
for path in sorted(glob.glob(str(DATA / "PM25_*.csv"))):
    year = int(Path(path).stem.split("_")[1])
    df = pd.read_csv(path, skiprows=7, encoding="utf-8-sig", dtype=str)
    df.columns = [c.split("//")[0].strip() for c in df.columns]
    df["source_file"] = Path(path).name
    df["year"] = year
    print(f"{Path(path).name}: {len(df):>7,} rows, cols={len(df.columns)}")
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)
print(f"\nTOTAL raw rows: {len(raw):,}")
# Which provinces actually exist, before filtering for Ontario.
print("\nProvince values:")
print(raw["Province/Territory"].value_counts(dropna=False).to_string())

# Filter to Ontario. The province column is whitespace-stripped first because
# the exports pad it.
on = raw[raw["Province/Territory"].str.strip().str.upper() == "ON"].copy()
print(f"\nOntario rows: {len(on):,}")
print(f"Ontario files: {on['source_file'].nunique()}, years: {sorted(on['year'].unique())}")

on["stn"] = on["NAPS ID"].str.strip()
on["lat"] = pd.to_numeric(on["Latitude"], errors="coerce")
on["lon"] = pd.to_numeric(on["Longitude"], errors="coerce")

stn = (
    on.groupby("stn")
    .agg(
        city=("City", lambda s: " / ".join(sorted(set(s.dropna().str.strip())))),
        lat=("lat", "first"),
        lon=("lon", "first"),
        methods=("Method Code", lambda s: ",".join(sorted(set(s.dropna().str.strip())))),
        rows=("stn", "size"),
        first_date=("Date", "min"),
        last_date=("Date", "max"),
        files=("source_file", "nunique"),
    )
    .sort_values("lon")
)
print(f"\nOntario stations: {len(stn)}")
print(stn.to_string())
print("\nLongitude range:", f"{stn['lon'].min():.4f} .. {stn['lon'].max():.4f}")
print("Stations west of 90W:", list(stn.loc[stn["lon"] < -90, "city"]))

# Audit the 24 hourly columns: how many are blank versus the -999 sentinel.
h = on[HOURS]
print("\nHourly cell audit (Ontario):")
print(f"  total cells      : {h.size:,}")
print(f"  NaN/blank        : {int(h.isna().sum().sum()):,}")
print(f"  '-999' sentinel  : {int((h == '-999').sum().sum()):,}")

# Duplicate station-days: same station, same date, more than one row.
dup = on.duplicated(subset=["stn", "Date"], keep=False)
print(f"\nDuplicate station-day rows: {int(dup.sum()):,} ({on.loc[dup, 'stn'].nunique()} stations affected)")
if dup.any():
    print(on.loc[dup].groupby("stn").size().to_string())

# Coverage check: 365 or 366 rows means a complete year, fewer means gaps.
print("\nRows per station-year (Ontario):")
print(on.pivot_table(index="stn", columns="year", values="Date", aggfunc="size").to_string())
