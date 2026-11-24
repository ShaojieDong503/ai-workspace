#!/usr/bin/env python3
"""Ontario PM2.5 hourly series standardized to UTC-6 (fixed CST, no DST).

Source files are NAPS/ECCC annual exports in hour-ending *local standard
time*.  This script keeps Ontario stations only and re-indexes every hour
onto a fixed UTC-6 clock, writing one wide CSV per year that mirrors the
source layout (6-line preamble, bilingual header, one row per station-day,
H01-H24).

Time maths
----------
    UTC    = LST - offset(zone)          offset is negative west of UTC
    target = UTC - 6 = LST - (offset + 6)

Ontario stations are all Eastern Standard Time (offset -5), so every hour
moves one hour earlier.  That is a whole-hour shift, so values are only
relabelled -- never interpolated or averaged.

    local H01 (ends 01:00 EST) -> 06:00 UTC -> 00:00 UTC-6  = previous day H24
    local H02 (ends 02:00 EST) -> 07:00 UTC -> 01:00 UTC-6  = same day H01
    ...
    local H24 (ends 24:00 EST) -> 05:00 UTC -> 23:00 UTC-6  = same day H23

So a UTC-6 day borrows H24 from the next local day, its other 23 hours come
from the same local day, and each station loses exactly one orphan hour at
the very start of its series.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Column names and constants shared by every stage below.
HOURS = [f"H{h:02d}" for h in range(1, 25)]
META = ["Pollutant", "Method Code", "NAPS ID", "City", "Province/Territory",
        "Latitude", "Longitude", "Date"]
PROVINCE = "ON"
NO_DATA = -999


# ---------------------------------------------------------------------------
# Time-zone resolution
#
# The source files are in local *standard* time, so the offset is the winter
# offset of the zone, never the daylight-saving one.  Ontario has two candidates
# split at the 90th meridian, and every station in this dataset sits east of it.
# ---------------------------------------------------------------------------
def standard_offset_hours(latitude: float, longitude: float) -> int:
    """Standard (winter) UTC offset for an Ontario station.

    Eastern (UTC-5) east of the 90th meridian, Central (UTC-6) west of it.
    Every Ontario station in 2010-2024 lies east of 90W (westernmost is
    Thunder Bay, -89.29), so this resolves to UTC-5 for all of them.  The
    rule is kept for future data and every assignment is written out to
    station_offsets.csv for review -- stations near the boundary (Atikokan,
    Pickle Lake, Sioux Lookout) would need a manual check.
    """
    return -5 if longitude > -90.0 else -6


# ---------------------------------------------------------------------------
# I/O
#
# The exports carry a 6-line bilingual preamble, so the real header is line 8 and
# read_csv must skip 7 rows.  Column names are bilingual ("Pollutant//Polluant"),
# so only the part before "//" is kept.
# ---------------------------------------------------------------------------
def read_year(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=7, encoding="utf-8-sig", dtype=str)
    df.columns = [c.split("//")[0].strip() for c in df.columns]
    return df[META + HOURS].copy()


# ---------------------------------------------------------------------------
# Cleaning
#
# -999 is the source's missing-data sentinel and becomes NaN *before* any
# arithmetic; leaving it in would poison every mean.  Zeros stay zeros.
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"].str.strip(), format="%Y-%m-%d", errors="raise")
    df["stn"] = df["NAPS ID"].str.strip()
    df["Method Code"] = pd.to_numeric(df["Method Code"], errors="raise").astype(int)
    df["lat"] = pd.to_numeric(df["Latitude"], errors="raise")
    df["lon"] = pd.to_numeric(df["Longitude"], errors="raise")

    # Everything arrives as text: coerce to numbers, then swap the sentinel for NaN.
    hourly = df[HOURS].apply(pd.to_numeric, errors="coerce").replace(NO_DATA, np.nan)
    df[HOURS] = hourly.astype("float64")
    df["n_valid"] = hourly.notna().sum(axis=1)
    return df


# ---------------------------------------------------------------------------
# De-duplication
#
# Some station-days appear twice, once per instrument method.  Keep the row with
# the most valid hours, breaking ties towards method 184 (the long-running
# continuous monitor).  Conflicts are written out, never silently resolved.
# ---------------------------------------------------------------------------
def dedupe(df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per station-day: most valid hours, tie-break to method 184."""
    df = df.copy()
    # Method 184 sorts first; every other method second.
    df["priority"] = np.where(df["Method Code"] == 184, 0, 1)
    # Two-pass sort: priority first, then a *stable* sort on valid hours, so rows
    # with equal hours keep the 184 preference established above.
    order = df.sort_values(["stn", "Date", "priority", "Method Code"])
    order = order.sort_values(["stn", "Date", "n_valid"], ascending=[True, True, False], kind="stable")
    keep = order.drop_duplicates(subset=["stn", "Date"], keep="first")

    dup_mask = df.duplicated(subset=["stn", "Date"], keep=False)
    conflicts = []
    for (stn, day), grp in df[dup_mask].groupby(["stn", "Date"]):
        grp = grp.sort_values("n_valid", ascending=False)
        a, b = grp.iloc[0], grp.iloc[1]
        if a["n_valid"] == 0 and b["n_valid"] == 0:
            continue
        diff_hours = int((~np.isclose(a[HOURS].astype(float), b[HOURS].astype(float),
                                      equal_nan=True)).sum())
        conflicts.append({
            "stn": stn, "date": day.date(),
            "kept_method": a["Method Code"], "kept_valid": int(a["n_valid"]),
            "dropped_method": b["Method Code"], "dropped_valid": int(b["n_valid"]),
            "hours_differing": diff_hours,
        })
    conflicts_df = pd.DataFrame(conflicts)
    if not conflicts_df.empty:
        conflicts_df.to_csv(out_dir / "duplicate_conflicts.csv", index=False, encoding="utf-8-sig")
    return keep.sort_values(["stn", "Date"]).reset_index(drop=True), conflicts_df


# ---------------------------------------------------------------------------
# The UTC-6 re-indexing
#
# Each value is an hour-ending label: H01 covers 00:00-01:00 local, H24 covers
# 23:00-24:00.  For standard offset o, UTC = LST - o, so the target clock is
# LST - (o + 6); for EST (o = -5) that is a one-hour shift.
#
# A whole-hour shift is a pure relabelling: no interpolation, no averaging.  It
# does move values across day boundaries, so the work happens on long
# (station, date, hour) rows and only pivots back to wide at the end.
# ---------------------------------------------------------------------------
def shift_to_utc6(df: pd.DataFrame, offsets: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    shift_h = {stn: -(off + 6) for stn, off in offsets.items()}

    long = df.melt(
        id_vars=["stn", "Date", "Method Code"],
        value_vars=HOURS, var_name="slot", value_name="pm25",
    )
    long["h"] = long["slot"].str[1:].astype(int)
    long["shift"] = long["stn"].map(shift_h)
    if long["shift"].isna().any():
        missing = sorted(long.loc[long["shift"].isna(), "stn"].unique())
        raise SystemExit(f"no timezone offset for stations: {missing}")

    # Hour-ending timestamp: H01 -> 01:00, H24 -> 24:00 (next day 00:00).
    ts_end = long["Date"] + pd.to_timedelta(long["h"], unit="h")
    ts_target = ts_end + pd.to_timedelta(long["shift"], unit="h")

    date_out = ts_target.dt.normalize()
    hour_out = ts_target.dt.hour
    # A target clock of exactly 00:00 is hour-ending 24 of the *previous* day,
    # not hour 0 of this one.  Rolling it back preserves the H01..H24 convention.
    roll = hour_out == 0                      # 00:00 belongs to previous day H24
    date_out = date_out.where(~roll, date_out - pd.Timedelta(days=1))
    hour_out = hour_out.where(~roll, 24)

    long["date_utc6"] = date_out
    long["h_out"] = hour_out

    wide = long.pivot(index=["stn", "date_utc6"], columns="h_out", values="pm25")
    wide = wide.reindex(columns=range(1, 25))
    wide.columns = HOURS

    grp = long.groupby(["stn", "date_utc6"])
    method = grp["Method Code"].agg(lambda s: s.iloc[0])
    n_methods = grp["Method Code"].nunique()

    wide = wide.reset_index()
    key = wide.set_index(["stn", "date_utc6"]).index
    wide["Method Code"] = key.map(method)
    wide["n_methods"] = key.map(n_methods)

    first_day = df.groupby("stn")["Date"].min()
    wide["first_local"] = wide["stn"].map(first_day)
    orphan = wide["date_utc6"] < wide["first_local"]
    orphan_df = wide.loc[orphan, ["stn", "date_utc6", "H24"]].copy()
    wide = wide.loc[~orphan].drop(columns=["first_local"])
    return wide, orphan_df


# ---------------------------------------------------------------------------
# Output
#
# Mirrors the source exactly: same preamble shape, same bilingual header, one row
# per station-day, same -999 sentinel.  Only the clock and the station set differ.
# ---------------------------------------------------------------------------
def write_year(out_dir: Path, year: int, frame: pd.DataFrame, stations: pd.DataFrame,
               gen_date: str) -> Path:
    frame = frame.sort_values(["stn", "date_utc6"]).reset_index(drop=True)
    out = pd.DataFrame({
        "Pollutant": "PM2.5",
        "Method Code": frame["Method Code"].astype(int),
        "NAPS ID": frame["stn"],
        "City": frame["stn"].map(stations["city"]),
        "Province/Territory": PROVINCE,
        "Latitude": frame["stn"].map(stations["lat_str"]),
        "Longitude": frame["stn"].map(stations["lon_str"]),
        "Date": frame["date_utc6"].dt.strftime("%Y-%m-%d"),
    })
    for h in HOURS:
        col = frame[h]
        if col.notna().any() and not np.allclose(col.dropna() % 1, 0):
            raise SystemExit(f"non-integer PM2.5 value in {year} {h}")
        out[h] = col.fillna(NO_DATA).round().astype("int64")

    preamble = (
        f"File generated on // Fichier créé sur, {gen_date}\n"
        "\n"
        "Pollutant // Polluant:, PM2.5\n"
        "Units // Unités, µg/m3\n"
        "Note // Note,Hour ending UTC-6 (fixed offset, no daylight saving time) // "
        "Heure se terminant à UTC-6 (décalage fixe, sans heure d'été)\n"
        f"Note // Note,Zeros are valid. {NO_DATA} denotes no data available // "
        f"Zéros sont valides .{NO_DATA} Données non disponibles\n"
        "\n"
    )
    header = ",".join(["Pollutant//Polluant", "Method Code//Code Méthode",
                       "NAPS ID//Identifiant SNPA", "City//Ville",
                       "Province/Territory//Province/Territoire", "Latitude//Latitude",
                       "Longitude//Longitude", "Date//Date"]
                      + [f"{h}//{h}" for h in HOURS])

    path = out_dir / f"ON_PM25_UTC6_{year}.csv"
    body = out.to_csv(index=False, header=False, lineterminator="\n")
    path.write_text(preamble + header + "\n" + body, encoding="utf-8-sig", newline="")
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Ontario PM2.5 -> UTC-6 builder")
    ap.add_argument("--data-dir", default=r"D:\ai-workspace\project\data\Target_Data_2010_2024")
    ap.add_argument("--out-dir", default=r"D:\ai-workspace\project\data\processed\on_utc6")
    ap.add_argument("--target-year", type=int, nargs="*", help="only write these years")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("PM25_*.csv"))
    if not files:
        raise SystemExit(f"no PM25_*.csv under {data_dir}")

    frames, per_file = [], []
    for path in files:
        df = read_year(path)
        n_all = len(df)
        df = df[df["Province/Territory"].str.strip().str.upper() == PROVINCE]
        per_file.append({"file": path.name, "rows_read": n_all, "on_rows": len(df)})
        frames.append(df)
        print(f"{path.name}: {n_all:>7,} rows -> {len(df):>6,} ON")

    raw = pd.concat(frames, ignore_index=True)
    cleaned = clean(raw)
    deduped, conflicts = dedupe(cleaned, out_dir)

    stations = (
        deduped.groupby("stn")
        .agg(city=("City", lambda s: s.dropna().str.strip().mode().iat[0]),
             lat=("lat", "median"), lon=("lon", "median"),
             first_local=("Date", "min"), last_local=("Date", "max"),
             n_source_days=("Date", "size"))
        .sort_values("lon")
    )
    stations["offset_hours"] = [standard_offset_hours(a, b)
                                for a, b in zip(stations["lat"], stations["lon"])]
    stations["shift_hours"] = -(stations["offset_hours"] + 6)
    stations["zone"] = np.where(stations["offset_hours"] == -5,
                                "America/Toronto (EST)", "America/Winnipeg (CST)")
    stations["lat_str"] = stations["lat"].map(lambda v: f"{v:.5f}")
    stations["lon_str"] = stations["lon"].map(lambda v: f"{v:.5f}")

    station_table = stations.reset_index()[["stn", "city", "lat", "lon", "offset_hours",
                                            "shift_hours", "zone", "first_local",
                                            "last_local", "n_source_days"]]
    station_table.to_csv(out_dir / "station_offsets.csv", index=False, encoding="utf-8-sig")

    offsets = dict(zip(stations.index, stations["offset_hours"]))
    wide, orphan = shift_to_utc6(deduped, offsets)

    gen_date = date.today().isoformat()
    written, file_rows = [], {}
    years = sorted(int(y) for y in wide["date_utc6"].dt.year.unique())
    for year in years:
        if args.target_year and year not in args.target_year:
            continue
        chunk = wide[wide["date_utc6"].dt.year == year]
        written.append(write_year(out_dir, year, chunk, stations, gen_date))
        file_rows[year] = len(chunk)

    method_mix = wide.loc[wide["Method Code"] != 184, ["stn", "date_utc6", "Method Code"]]
    method_mix.to_csv(out_dir / "method_mix.csv", index=False, encoding="utf-8-sig")

    runs = (wide.sort_values(["stn", "date_utc6"])
            .groupby("stn")["Method Code"]
            .apply(lambda s: int((s != s.shift()).sum())))
    runs.rename("method_runs").to_csv(out_dir / "method_runs.csv", encoding="utf-8-sig")

    # ---------- QA ----------
    # The orphan hour: a station's first local H01 has no earlier UTC-6 day to
    # land in, so it is dropped rather than invented.
    first_day = deduped.groupby("stn")["Date"].transform("min")
    orphan_mask = (deduped["Date"] == first_day).to_numpy()
    src = deduped[HOURS].to_numpy(dtype=float).copy()
    dropped_values = int(np.isfinite(src[orphan_mask, 0]).sum())
    # Blank those cells so the value-conservation check compares like with like.
    src[orphan_mask, 0] = np.nan
    src_vals = np.sort(src[~np.isnan(src)])

    out_arr = wide[HOURS].to_numpy(dtype=float)
    out_vals = np.sort(out_arr[~np.isnan(out_arr)])
    conserved = src_vals.size == out_vals.size and bool(np.array_equal(src_vals, out_vals))

    # Hours actually observed per station-day, for the completeness table.
    comp = wide[HOURS].notna().sum(axis=1)
    last_day = wide.groupby("stn")["date_utc6"].transform("max")
    trailing = (wide["date_utc6"] == last_day) & wide["H24"].isna()

    if conflicts.empty:
        both = blank = conflicts
    else:
        both = conflicts[conflicts["dropped_valid"] > 0]
        blank = conflicts[conflicts["dropped_valid"] == 0]

    per_station_methods = wide.groupby("stn")["Method Code"].agg(
        lambda s: ",".join(str(int(v)) for v in sorted(set(s))))
    mixed_method_stations = per_station_methods[per_station_methods.str.contains(",")]

    # Assemble the markdown report line by line, then write it with a BOM so
    # Windows tools render the accents and the en dashes correctly.
    lines = [
        "# Ontario PM2.5 → UTC-6 build report", "",
        f"Generated: {gen_date}",
        f"Source: `{data_dir}`",
        f"Output: `{out_dir}` ({len(written)} files)", "",
        "## Mapping", "",
        "`target = LST - (offset + 6)`; every ON station is EST (offset −5), so **shift = −1 h**.",
        "A whole-hour shift relabels values and never interpolates them.",
        "Per-station offsets: `station_offsets.csv`.", "",
        "## Row accounting", "",
        f"- source rows read: {sum(p['rows_read'] for p in per_file):,}",
        f"- Ontario rows kept: {len(cleaned):,}",
        f"- duplicate station-day rows: {int(cleaned.duplicated(subset=['stn', 'Date'], keep=False).sum()):,}",
        f"  - dropped twin was entirely {NO_DATA}: {len(blank):,}",
        f"  - dropped twin also carried data: {len(both):,} (`duplicate_conflicts.csv`)",
        f"- source station-days after dedupe: {len(deduped):,}",
        f"- UTC-6 station-days written: {len(wide):,}",
        f"- orphan hours dropped at series start: {len(orphan):,} ({dropped_values:,} carried a value)",
        "", "## Completeness (UTC-6 days)", "",
        f"- 24/24 valid hours: {int((comp == 24).sum()):,}",
        f"- 1–23 valid hours: {int(((comp > 0) & (comp < 24)).sum()):,}",
        f"  - of which the series-end day missing H24: {int(trailing.sum()):,}",
        f"- 0 valid hours: {int((comp == 0).sum()):,}",
        "", "## Value conservation", "",
        f"- valid values after dedupe: {int(src_vals.size):,}",
        f"- valid values after dropping orphans: {int(src_vals.size):,}",
        f"- valid values written: {int(out_vals.size):,}",
        f"- multiset identical: **{'PASS' if conserved else 'FAIL'}**", "",
        "## Stations", "",
        f"- {len(stations)} Ontario stations, longitude {stations['lon'].min():.4f} .. {stations['lon'].max():.4f}",
        "- offset −5 h (EST): {n} stations; offset −6 h (CST): {m} stations".format(
            n=int((stations["offset_hours"] == -5).sum()),
            m=int((stations["offset_hours"] == -6).sum())),
        f"- stations whose kept Method Code changes over time: {len(mixed_method_stations)} "
        f"(`method_runs.csv`; >4 runs means day-level flip-flopping: "
        f"{', '.join(runs[runs > 4].index) or 'none'})",
        "", "## Rows per file", "",
        "| year | station-days |", "|---|---:|",
    ] + [f"| {y} | {n:,} |" for y, n in file_rows.items()]

    (out_dir / "qa_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    print(f"\nwrote {len(written)} year files to {out_dir}")
    print(f"station-days: {len(deduped):,} source -> {len(wide):,} UTC-6")
    print(f"orphan hours dropped: {len(orphan)} ({dropped_values} with a value)")
    print(f"value conservation: {'PASS' if conserved else 'FAIL'} ({src_vals.size:,} vs {out_vals.size:,})")
    print(f"complete days {int((comp == 24).sum()):,}; partial {int(((comp > 0) & (comp < 24)).sum()):,}; "
          f"empty {int((comp == 0).sum()):,}")
    return 0 if conserved else 1


if __name__ == "__main__":
    sys.exit(main())

