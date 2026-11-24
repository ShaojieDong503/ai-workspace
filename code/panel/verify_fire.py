#!/usr/bin/env python3
"""Independent check of the fire extractor, by brute force.

Loads the year's detections, applies type == 0, converts to the UTC-6 day, and
for each sampled cell-day computes distances with a plain haversine formula over
that day's fires - no BallTree, no radius query, no shared code with the builder.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from utils import ROOT
from verify_common import cells, check, compare, sample_pairs, summary

SRC = ROOT / "data" / "MODIS_fire"
RADII = (25, 50, 100, 250, 500)
R_EARTH = 6371.0088


def load_year(year: int) -> pd.DataFrame:
    frames = []
    for name in (f"modis_{year}_Canada.csv", f"modis_{year}_United_States.csv"):
        p = SRC / name
        if p.exists():
            frames.append(pd.read_csv(p, low_memory=False))
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True).drop_duplicates()
    d = d[d["type"] == 0].copy()
    ts = pd.to_datetime(d["acq_date"].astype(str) + " " + d["acq_time"].astype(str).str.zfill(4),
                        format="%Y-%m-%d %H%M", errors="coerce")
    d["day"] = (ts - pd.Timedelta(hours=6)).dt.date
    d["frp"] = pd.to_numeric(d["frp"], errors="coerce").fillna(0.0)
    return d.dropna(subset=["day", "latitude", "longitude"])


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def rule_cases(fires: pd.DataFrame) -> None:
    """Every detection's UTC-6 day must follow the 06:00 UTC cutoff."""
    t = fires["acq_time"].to_numpy()
    stamp = pd.to_datetime(fires["acq_date"].astype(str)).dt.normalize()
    got = pd.to_datetime(pd.Series(fires["day"].to_numpy())).reset_index(drop=True)
    before = t < 600
    want = (stamp - pd.Timedelta(days=1)).where(before, stamp).reset_index(drop=True)

    mism = int((got != want).sum())
    check(mism == 0, f"UTC-6 day rule holds for {len(fires):,} detections "
                     f"({int(before.sum()):,} before 06:00 UTC, {mism} mismatches)")

    cross = stamp.dt.month.eq(1) & stamp.dt.day.eq(1) & before
    check(True, f"1 January 00:00-06:00 UTC detections in {int(stamp.dt.year.max())}: "
                f"{int(cross.sum()):,} (they belong to 31 December of the previous year)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the fire extractor")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--year", type=int, default=2015)
    ap.add_argument("--month", type=int, default=7)
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    df = sample_pairs(args.cells, args.year, args.month, args.n)
    cs = cells()
    fires = load_year(args.year)

    got = {f"fire_count_{r}km": [] for r in RADII}
    got.update({f"fire_frp_sum_{r}km": [] for r in RADII})
    got["fire_dist_nearest_km"] = []
    want = {k: [] for k in got}

    for row in df.itertuples():
        day = pd.Timestamp(row.date).date()
        lat, lon = cs.loc[row.cell_id, "lat"], cs.loc[row.cell_id, "lon"]
        f = fires[fires["day"] == day]
        for k in got:
            got[k].append(getattr(row, k))
        if f.empty:
            for r in RADII:
                want[f"fire_count_{r}km"].append(0)
                want[f"fire_frp_sum_{r}km"].append(0.0)
            want["fire_dist_nearest_km"].append(np.nan)
            continue
        d = haversine_km(lat, lon, f["latitude"].to_numpy(), f["longitude"].to_numpy())
        want["fire_dist_nearest_km"].append(float(d.min()))
        for r in RADII:
            m = d <= r
            want[f"fire_count_{r}km"].append(int(m.sum()))
            want[f"fire_frp_sum_{r}km"].append(float(f["frp"].to_numpy()[m].sum()))

    for k in got:
        # counts are integers and must match exactly; FRP sums and distances are
        # accumulated in a different order, so they get a relative tolerance
        exact = k.startswith("fire_count")
        if exact:
            compare(k, np.array(got[k], "float64"), np.array(want[k], "float64"), tol=0.0)
        elif k == "fire_dist_nearest_km":
            # 1 metre: BallTree's haversine and the asin formula differ only in
            # rounding, but a relative test is meaningless for short distances
            compare(k, np.array(got[k], "float64"), np.array(want[k], "float64"),
                    tol=1e-3, rtol=1e-6)
        else:
            compare(k, np.array(got[k], "float64"), np.array(want[k], "float64"),
                    tol=1e-9, rtol=1e-6)

    if "fire_count_500km" in df:
        v = df["fire_count_500km"].to_numpy()
        check(bool((v >= 0).all()), "fire counts non-negative")

    # day-boundary case: a detection at 05:59 UTC belongs to the previous UTC-6
    # day, one at 06:01 to the current day. Compute the expectation by hand with
    # datetime arithmetic, not with the loader's pandas shift.
    edge = fires[(fires["acq_time"] >= 555) & (fires["acq_time"] <= 605)]
    if edge.empty:
        print("no detection within 10 minutes of the 06:00 boundary - case skipped")
    else:
        r = edge.iloc[0]
        stamp = __import__("datetime").datetime.strptime(
            f"{r.acq_date} {int(r.acq_time):04d}", "%Y-%m-%d %H%M")
        expected = (stamp - __import__("datetime").timedelta(hours=6)).date()
        check(expected == r["day"],
              "boundary detection %s UTC -> UTC-6 day %s (loader said %s)"
              % (stamp.strftime("%Y-%m-%d %H:%M"), expected, r["day"]))
    rule_cases(fires)

    return summary()


if __name__ == "__main__":
    raise SystemExit(main())