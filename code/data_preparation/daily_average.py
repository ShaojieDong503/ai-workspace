#!/usr/bin/env python3
"""Daily means from the Ontario UTC-6 hourly product.

A UTC-6 station-day is kept only if at least 18 of its 24 hours are valid
(-999 counts as missing).  The mean is taken over the valid hours only, so
a kept day with 19 hours averages those 19 values.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Column names, the missing-data sentinel and the minimum-hours rule.
HOURS = [f"H{h:02d}" for h in range(1, 25)]
NO_DATA = -999
MIN_VALID = 18
# Output header, bilingual so the daily product matches the hourly one.
BILINGUAL = {
    "Pollutant": "Pollutant//Polluant",
    "NAPS ID": "NAPS ID//Identifiant SNPA",
    "City": "City//Ville",
    "Province/Territory": "Province/Territory//Province/Territoire",
    "Latitude": "Latitude//Latitude",
    "Longitude": "Longitude//Longitude",
    "Date": "Date//Date",
    "PM25_daily": "PM25_daily//PM2.5_journalier",
    "n_valid_hours": "n_valid_hours//Nb_heures_valides",
}


# Every file repeats the source's 6-line preamble so the products look alike.
def preamble(gen: str) -> str:
    return (
        f"File generated on // Fichier créé sur, {gen}\n"
        "\n"
        "Pollutant // Polluant:, PM2.5\n"
        "Units // Unités, µg/m3\n"
        "Note // Note,Daily mean of hour ending UTC-6 values; days with fewer than "
        f"{MIN_VALID} valid hours are omitted // Moyenne journalière des heures se "
        f"terminant à UTC-6; les jours avec moins de {MIN_VALID} heures valides sont omis\n"
        f"Note // Note,Zeros are valid. {NO_DATA} denotes no data available // "
        f"Zéros sont valides .{NO_DATA} Données non disponibles\n"
        "\n"
    )


# Write one CSV: preamble + bilingual header + body, UTF-8 with BOM so Excel
# opens it without mangling the French accents.
def write(frame: pd.DataFrame, path: Path, gen: str) -> None:
    body = frame.rename(columns=BILINGUAL).to_csv(index=False, header=False, lineterminator="\n")
    path.write_text(preamble(gen) + ",".join(BILINGUAL.values()) + "\n" + body,
                    encoding="utf-8-sig", newline="")


# Read the hourly product, apply the 18-hour rule, write one file per year.
def build(hourly_dir: Path, out_dir: Path) -> int:
    files = [p for p in sorted(hourly_dir.glob("ON_PM25_UTC6_*.csv")) if "daily" not in p.name]
    if not files:
        raise SystemExit(f"no hourly files under {hourly_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = date.today().isoformat()
    stats, written, all_rows = [], [], []

    for path in files:
        year = int(path.stem.split("_")[-1])
        df = pd.read_csv(path, skiprows=7, encoding="utf-8-sig", dtype=str)
        df.columns = [c.split("//")[0].strip() for c in df.columns]
        # Sentinel to NaN first: the mean below must never see -999.
        hourly = df[HOURS].apply(pd.to_numeric, errors="raise").replace(NO_DATA, np.nan)
        # How many of the 24 hours are real observations.
        n_valid = hourly.notna().sum(axis=1)
        # Mean over the valid hours only, exactly as the rule states.
        mean = hourly.mean(axis=1, skipna=True)

        # The rule: drop a day with fewer than 18 valid hours; average the rest.
        keep = n_valid >= MIN_VALID
        stats.append({
            "year": year,
            "rows_in": len(df),
            "rows_kept": int(keep.sum()),
            "rows_dropped": int((~keep).sum()),
            "dropped_lt18": int(((n_valid > 0) & (~keep)).sum()),
            "dropped_empty": int(((n_valid == 0) & (~keep)).sum()),
            "kept_min_hours": int(n_valid[keep].min()) if keep.any() else None,
            "mean_min": round(float(mean[keep].min()), 3) if keep.any() else None,
            "mean_max": round(float(mean[keep].max()), 3) if keep.any() else None,
        })

        out = pd.DataFrame({
            "Pollutant": "PM2.5",
            "NAPS ID": df["NAPS ID"],
            "City": df["City"],
            "Province/Territory": df["Province/Territory"],
            "Latitude": df["Latitude"],
            "Longitude": df["Longitude"],
            "Date": df["Date"],
            "PM25_daily": mean.round(2),
            "n_valid_hours": n_valid,
        })[keep].reset_index(drop=True)
        all_rows.append(out)

        target = out_dir / f"ON_PM25_UTC6_daily_{year}.csv"
        write(out, target, gen)
        written.append(target)

    st = pd.DataFrame(stats)
    combined = pd.concat(all_rows, ignore_index=True)
    write(combined, out_dir / "ON_PM25_UTC6_daily_2010_2024.csv", gen)

    lines = [
        "# Ontario PM2.5 — daily means (UTC-6)", "",
        f"Generated: {gen}",
        f"Source: `{hourly_dir}`",
        f"Output: `{out_dir}`", "",
        f"Rule: keep a station-day when **n_valid_hours >= {MIN_VALID}**; "
        f"the mean is over valid hours only. `{NO_DATA}` never enters the mean.", "",
        f"- hourly station-days in: {int(st['rows_in'].sum()):,}",
        f"- daily rows kept: {int(st['rows_kept'].sum()):,}",
        f"- rows dropped, 1–17 valid hours: {int(st['dropped_lt18'].sum()):,}",
        f"- rows dropped, 0 valid hours: {int(st['dropped_empty'].sum()):,}",
        f"- mean range: {combined['PM25_daily'].min():.2f} .. {combined['PM25_daily'].max():.2f} µg/m³",
        f"- stations: {combined['NAPS ID'].nunique()}",
        "", "| year | kept | dropped <18 h | dropped empty | min hours kept | max daily mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in st.iterrows():
        lines.append(f"| {r['year']} | {r['rows_kept']:,} | {r['dropped_lt18']:,} | "
                     f"{r['dropped_empty']:,} | {r['kept_min_hours']} | {r['mean_max']:.2f} |")
    (out_dir / "qa_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    print(st.to_string(index=False))
    print(f"\nkept {len(combined):,} station-days, wrote {len(written)} year files + combined")
    return 0


# CLI entry point.
def main() -> int:
    ap = argparse.ArgumentParser(description="Daily means from ON UTC-6 hourly data")
    ap.add_argument("--hourly-dir", default=r"D:\ai-workspace\project\data\processed\on_utc6")
    ap.add_argument("--out-dir", default=r"D:\ai-workspace\project\data\processed\on_utc6_daily")
    args = ap.parse_args()
    return build(Path(args.hourly_dir), Path(args.out_dir))


if __name__ == "__main__":
    sys.exit(main())
