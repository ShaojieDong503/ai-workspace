#!/usr/bin/env python3
"""Station-day PM2.5 targets on the UTC-6 day, all three networks.

Rule
----
The target is the daily mean over the UTC-6 day (D 06:00 UTC .. D+1 06:00 UTC),
computed only when at least 18 of the 24 hours are valid. Missing hours are
never averaged over, and the mean is never taken over fewer than 18 hours.

A negative hourly value is invalid and treated exactly like a missing hour
(supervisor's rule): it neither enters the mean nor counts toward the 18.

Ontario is read from the hourly UTC-6 product data/processed/on_utc6 (same
clock; the daily on_utc6_daily product keeps negatives, so it is not used).
Quebec and Manitoba are built here by importing the verified functions from
code/data_preparation/build_on_utc6.py rather than copying them; Manitoba is
CST, so its standard offset is -6 and the shift is zero. AQS is read from the
annual zips in UTC and shifted back six hours, which is exact because AQS
timestamps are hour-start, non-DST. A UTC-6 day straddles two annual files
(31 Dec 18:00 UTC onward lies in the next year's file), so AQS hours are pooled
across files as sums and counts before the 18-hour rule.

Output: data/interim/stations/station_day.parquet
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv

ROOT = Path(r"D:\ai-workspace\project")
sys.path.insert(0, str(ROOT / "code" / "data_preparation"))
import build_on_utc6 as b  # noqa: E402  (verified UTC-6 builder)

RAW_NAPS = ROOT / "data" / "Target_Data_2010_2024"
ON_HOURLY = ROOT / "data" / "processed" / "on_utc6"
AQS_DIR = ROOT / "data" / "raw" / "AQS"
STATIONS = ROOT / "data" / "interim" / "stations" / "stations.parquet"
OUT = ROOT / "data" / "interim" / "stations"

MIN_VALID = 18


def from_wide(wide: pd.DataFrame, network: str) -> pd.DataFrame:
    """Apply the >=18-hour rule to a UTC-6 wide table and return long station-days."""
    hourly = wide[b.HOURS].astype("float64")
    hourly = hourly.where(hourly >= 0)  # negative hour = missing hour
    n_valid = hourly.notna().sum(axis=1)
    keep = n_valid >= MIN_VALID
    out = pd.DataFrame({
        "station_uid": (network + ":" + wide["stn"].astype(str)),
        "date": wide["date_utc6"].dt.date,
        "pm25": hourly.mean(axis=1, skipna=True).astype("float32"),
        "n_valid_hours": n_valid.astype("int8"),
        "n_poc": np.int8(1),
    })[keep]
    return out


def naps_on() -> pd.DataFrame:
    frames = []
    for p in sorted(ON_HOURLY.glob("ON_PM25_UTC6_*.csv")):
        d = pd.read_csv(p, skiprows=7, encoding="utf-8-sig", dtype=str)
        d.columns = [c.split("//")[0].strip() for c in d.columns]
        hours = d[b.HOURS].apply(pd.to_numeric, errors="raise").replace(b.NO_DATA, np.nan)
        hours["stn"] = d["NAPS ID"].str.strip()
        hours["date_utc6"] = pd.to_datetime(d["Date"])
        frames.append(hours)
    return from_wide(pd.concat(frames, ignore_index=True), "NAPS")


def naps_qc_mb(tmp_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(RAW_NAPS.glob("PM25_*.csv")):
        d = b.read_year(p)
        d = d[d["Province/Territory"].str.strip().str.upper().isin(["QC", "MB"])]
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    cleaned = b.clean(raw)
    deduped, _ = b.dedupe(cleaned, tmp_dir)

    # standard offset per station: QC is EST (-5), MB is CST (-6)
    stations = deduped.groupby("stn").agg(lat=("lat", "median"), lon=("lon", "median"))
    offsets = {s: b.standard_offset_hours(r.lat, r.lon) for s, r in stations.iterrows()}
    wide, _ = b.shift_to_utc6(deduped, offsets)
    return from_wide(wide, "NAPS")


def aqs(sites: set[str], chunk_rows: int = 2_000_000) -> pd.DataFrame:
    parts = []
    for z in sorted(AQS_DIR.glob("hourly_88101_*.zip")):
        with zipfile.ZipFile(z) as zf:
            member = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(member) as fh:
                tbl = pv.read_csv(
                    fh,
                    convert_options=pv.ConvertOptions(
                        include_columns=["State Code", "County Code", "Site Num", "POC",
                                         "Date GMT", "Time GMT", "Sample Measurement"],
                        column_types={"State Code": pa.string(), "County Code": pa.string(),
                                      "Site Num": pa.string()},
                    ),
                    read_options=pv.ReadOptions(block_size=64 << 20),
                )
        d = tbl.to_pandas()
        d["site_id"] = d["State Code"] + "-" + d["County Code"] + "-" + d["Site Num"]
        d = d[d["site_id"].isin(sites)]
        if d.empty:
            continue
        ts = pd.to_datetime(d["Date GMT"].astype(str) + " " + d["Time GMT"].astype(str),
                            errors="coerce", format="mixed")
        # hour-start UTC -> UTC-6 day
        d["day"] = (ts - pd.Timedelta(hours=6)).dt.date
        d["value"] = pd.to_numeric(d["Sample Measurement"], errors="coerce")
        d["value"] = d["value"].where(d["value"] >= 0)  # negative hour = missing hour
        d["poc"] = d["POC"].astype(str)
        g = (d.dropna(subset=["day"])
               .groupby(["site_id", "poc", "day"])
               .agg(s=("value", "sum"), n=("value", "count")))
        parts.append(g.reset_index())
    # pool the days that straddle two annual files before applying the rule
    g = (pd.concat(parts, ignore_index=True)
           .groupby(["site_id", "poc", "day"], as_index=False)[["s", "n"]].sum())
    g = g[g["n"] >= MIN_VALID]
    g["value"] = g["s"] / g["n"]
    site_day = (g.groupby(["site_id", "day"])
                  .agg(pm25=("value", "mean"), n_poc=("poc", "size"),
                       n_valid_hours=("n", "min"))
                  .reset_index())
    return pd.DataFrame({
        "station_uid": "AQS:" + site_day["site_id"].astype(str),
        "date": site_day["day"],
        "pm25": site_day["pm25"].astype("float32"),
        "n_valid_hours": site_day["n_valid_hours"].astype("int8"),
        "n_poc": site_day["n_poc"].astype("int8"),
    })


def main() -> int:
    ap = argparse.ArgumentParser(description="Station-day targets on the UTC-6 day")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    stations = pd.read_parquet(STATIONS)
    sites = set(stations.loc[stations.network == "AQS", "station_uid"]
                .str.replace("AQS:", "", regex=False))

    parts = [naps_on(), naps_qc_mb(out), aqs(sites)]
    sd = pd.concat(parts, ignore_index=True)
    sd = sd.dropna(subset=["pm25"])
    sd["date"] = pd.to_datetime(sd["date"])
    sd = sd.drop_duplicates(subset=["station_uid", "date"])

    # Keep only stations that resolved to a cell in the spine. NAPS decades
    # contain many sites outside the square, so the target stage must filter.
    cell = dict(zip(stations["station_uid"], stations["cell_id"]))
    keep = sd["station_uid"].isin(cell)
    print(f"station-days outside the spine: {int((~keep).sum()):,} "
          f"({sd.loc[~keep, 'station_uid'].nunique()} stations) -> dropped")
    sd = sd[keep].copy()
    sd["cell_id"] = sd["station_uid"].map(cell).astype("int32")
    sd = sd[["station_uid", "date", "cell_id", "pm25", "n_valid_hours", "n_poc"]]
    sd = sd.sort_values(["date", "station_uid"]).reset_index(drop=True)
    sd.to_parquet(out / "station_day.parquet", index=False, compression="zstd")

    print(f"station-days: {len(sd):,}")
    print(sd.groupby(sd.station_uid.str.split(':').str[0]).size().to_string())
    print("\nper year:")
    print(sd.groupby(sd.date.dt.year).size().to_string())
    print("\nstations with data:", sd.station_uid.nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
