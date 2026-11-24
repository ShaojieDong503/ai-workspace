#!/usr/bin/env python3
"""2023 smoke-episode diagnostics: what the model predicted, and what the predictors saw.

Rule
----
Development cells only, 2023 only, and nothing else. `data.assert_clean` is called on
the merged frame before a single panel is drawn, so a spatial-test cell, a calibration
cell or a 2024 row would stop the script rather than appear in a figure. The test sets
stay sealed until Phase 3.

Two predictions are drawn, and they mean different things:

* **LOYO** (`loyo/oof.parquet`) -- the fold that held out all of 2023, so the model
  never saw the year. This is the primary line.
* **LORO** (`loro/oof.parquet`) -- 2023 was in training but the cell's region was held
  out. It shows what the model can do when the year is known but the region is not.

MERRA-2 surface PM2.5 is reconstructed from the composition fields following the GMAO
MERRA-2 FAQ (https://gmao.gsfc.nasa.gov/gmao-products/merra-2/faq_merra-2/):

    PM2.5 = DUSMASS25 + OCSMASS + BCSMASS + SSSMASS25 + SO4MASS * (132.14 / 96.06)

The columns are stored in kg/m3, so the sum is multiplied by 1e9 to give ug/m3. The
132.14/96.06 factor converts sulphate mass as SO4 to the mass as (NH4)2SO4, which is
what the FAQ's surface PM2.5 definition assumes.

The wind row draws equal-length unit arrows pointing where the wind blows *toward*, so
it reads as smoke transport. Only the direction is encoded: speed has its own panel, and
a reader takes an arrow as the direction of flow (wind barbs, not arrows, point at the
source).

Everything is read through `data.load()`; no hashed module is modified.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data as D  # noqa: E402

REGION_NAMES = {1: "Toronto", 2: "Central", 3: "East", 4: "West SW", 5: "West NH", 6: "North"}

# Okabe-Ito, colourblind-safe, and used in every figure so colours keep their meaning
C_OBS = "#000000"
C_LOYO = "#0072B2"
C_LORO = "#D55E00"
C_MERRA = "#009E73"
C_AOD = "#CC79A7"
C_SMOKE = "#E69F00"
C_GRID = "#BBBBBB"

MERRA_PM25 = ["merra_dusmass25", "merra_ocsmass", "merra_bcsmass",
              "merra_sssmass25", "merra_so4smass"]
SULPHATE_RATIO = 132.14 / 96.06

PANEL_COLS = ["pm25", "aod055", "aod055_npix", "aod_nbr_50km_d0", "merra_totexttau",
              "hms_any", "hms_density_max", "fire_frp_sum_500km", "fire_count_250km",
              "fire_dist_nearest_km", "merra_pblh", "merra_u10m", "merra_v10m",
              *MERRA_PM25]

# how a region's cells collapse to one daily series: means, except smoke and the count
REGION_AGG = {"pm25": "mean", "yhat_loyo": "mean", "yhat_loro": "mean",
              "aod055": "mean", "aod055_npix": "mean",
              "aod_nbr_50km_d0": "mean", "merra_totexttau": "mean",
              "hms_density_max": "max", "fire_frp_sum_500km": "mean",
              "fire_count_250km": "mean", "fire_dist_nearest_km": "min",
              "merra_pblh": "mean", "merra_u10m": "mean", "merra_v10m": "mean",
              **{c: "mean" for c in MERRA_PM25}}


def load_oof(root: Path, run: str, scheme: str) -> pd.DataFrame:
    """One OOF file, with the path named in the error if it is not there."""
    p = root / run / scheme / "oof.parquet"
    if not p.exists():
        raise SystemExit(f"missing OOF file: {p}\n"
                         f"run with --quick to use data/interim/model/quick/, "
                         f"or --oof-root to point at the SciNet output")
    o = pd.read_parquet(p)
    need = {"cell_id", "date", "y", "yhat"}
    if not need.issubset(o.columns):
        raise SystemExit(f"{p} lacks {sorted(need - set(o.columns))}")
    o = o[["cell_id", "date", "y", "yhat"]].copy()
    o["date"] = pd.to_datetime(o["date"])
    if o.duplicated(["cell_id", "date"]).any():
        raise AssertionError(f"{p}: duplicate (cell_id, date) rows")
    return o


def dev_2023(d: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Development cells, 2023, with the columns the panels need.

    `hms_density_max` is NaN whenever the cell has no smoke over it, which is not the
    same thing as missing data: `hms_any` says which it is. Where `hms_any == 0` the
    density is set to 0, so a clear day reads as "no smoke" rather than as a gap. The
    only NaN left is the handful of days with no HMS file at all, where `hms_any` is
    itself NaN.
    """
    dev = set(splits.loc[splits["role"] == "development", "cell_id"].astype(int))
    m = d["cell_id"].isin(dev) & (d["year"] == 2023)
    out = d.loc[m, ["cell_id", "date", "year", "on_region", *PANEL_COLS]].copy()
    if out.empty:
        raise SystemExit("no development rows in 2023")

    no_smoke = out["hms_any"].fillna(-1) == 0
    out.loc[no_smoke, "hms_density_max"] = 0.0
    still_nan = int(out["hms_density_max"].isna().sum())
    print(f"smoke: {int(no_smoke.sum()):,} clear cell-days set to 0; "
          f"{still_nan:,} left NaN (no HMS file)")
    if (out.loc[no_smoke, "hms_density_max"] != 0).any():
        raise AssertionError("hms_density_max not zero where hms_any == 0")
    return out.rename(columns={"on_region": "region"})


def merge_predictions(base: pd.DataFrame, loyo: pd.DataFrame, loro: pd.DataFrame) -> pd.DataFrame:
    """Attach both predictions, asserting a one-to-one join and that y == pm25."""
    out = base.merge(loyo.rename(columns={"yhat": "yhat_loyo", "y": "y_loyo"}),
                     on=["cell_id", "date"], how="left", validate="one_to_one")
    missing = int(out["yhat_loyo"].isna().sum())
    if missing:
        raise AssertionError(f"{missing} development rows have no LOYO prediction")
    out = out.merge(loro.rename(columns={"yhat": "yhat_loro", "y": "y_loro"}),
                    on=["cell_id", "date"], how="left", validate="one_to_one")
    missing = int(out["yhat_loro"].isna().sum())
    if missing:
        raise AssertionError(f"{missing} development rows have no LORO prediction")
    for col in ("y_loyo", "y_loro"):
        bad = ~np.isclose(out[col].to_numpy(float), out["pm25"].to_numpy(float), atol=1e-4)
        if bad.any():
            raise AssertionError(f"{int(bad.sum())} rows where {col} != pm25 after the join")
    return out.drop(columns=["y_loyo", "y_loro"])


def merra_pm25_ugm3(df: pd.DataFrame) -> pd.Series:
    """MERRA-2 surface PM2.5 in ug/m3, per the GMAO FAQ formula."""
    total = sum(df[c].to_numpy(float) for c in MERRA_PM25[:4])
    so4 = df["merra_so4smass"].to_numpy(float) * SULPHATE_RATIO
    return pd.Series(total + so4, index=df.index) * 1e9


def detect_episodes(daily_max: pd.Series, threshold: float, pad: int = 5) -> list:
    """Days above `threshold`, merged when at most one day separates them, then padded.

    "Consecutive or 1-day-apart" is read as: merge when the gap is at most two days,
    so a single quiet day inside a smoke spell does not split it in two.

    The padding is applied before the windows are returned, and windows that then
    overlap are merged again: otherwise a short spell in the middle of a busy month
    produces a window that reaches into its neighbour's, and one episode reports a
    peak that belongs to the next one.
    """
    hot = daily_max[daily_max > threshold].index.to_numpy()
    if len(hot) == 0:
        return []
    hot = np.sort(hot)
    groups, start, prev = [], hot[0], hot[0]
    for day in hot[1:]:
        if (day - prev) / np.timedelta64(1, "D") <= 2:
            prev = day
            continue
        groups.append((start, prev))
        start = prev = day
    groups.append((start, prev))

    padded = [(pd.Timestamp(a) - pd.Timedelta(days=pad), pd.Timestamp(b) + pd.Timedelta(days=pad))
              for a, b in groups]
    merged = [padded[0]]
    for a, b in padded[1:]:
        last_a, last_b = merged[-1]
        if a <= last_b:                       # the padded windows overlap
            merged[-1] = (last_a, max(last_b, b))
        else:
            merged.append((a, b))
    return merged


def parse_episodes(spec: str) -> list:
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            a, b = chunk.split(":")
        except ValueError:
            raise SystemExit(f"bad --episodes item {chunk!r}; expected START:END")
        out.append((pd.Timestamp(a), pd.Timestamp(b)))
    return out


def series_for(frame: pd.DataFrame, by: str, key: int) -> pd.DataFrame:
    """Daily series for one cell or one region mean, over the frame's window.

    The caller normally passes a frame already indexed by date; `date` is restored
    as a column so the same code works either way.
    """
    if "date" not in frame.columns:
        frame = frame.reset_index()
    if by == "cell":
        sub = frame[frame["cell_id"] == key]
        if sub.empty:
            raise SystemExit(f"cell {key} has no rows in this window")
        out = sub.set_index("date").sort_index()
        out["aod_share"] = (out["aod055_npix"] > 0).astype(float)
        return out
    sub = frame[frame["region"] == key]
    if sub.empty:
        raise SystemExit(f"region {key} has no rows in this window")
    out = sub.groupby("date").agg(REGION_AGG).sort_index()
    out["aod_share"] = sub.groupby("date")["aod055_npix"].apply(
        lambda s: float((s > 0).mean())).reindex(out.index)
    return out


def shade_no_retrieval(ax, frame: pd.DataFrame) -> None:
    """Light bands on days with no MAIAC retrieval, so a gap reads as a gap."""
    miss = frame["aod_share"].fillna(0).to_numpy() == 0
    dates = frame.index.to_numpy()
    i = 0
    while i < len(miss):
        if not miss[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(miss) and miss[j + 1]:
            j += 1
        half = pd.Timedelta(hours=12)
        ax.axvspan(dates[i] - half, dates[j] + half, color=C_GRID, alpha=0.25, lw=0, zorder=0)
        i = j + 1


def plot_episode(series: pd.DataFrame, by: str, key: int, episode: tuple, run: str,
                 quick: bool, outdir: Path) -> dict:
    s, e = episode
    w = series[(series.index >= s) & (series.index <= e)]
    if w.empty:
        raise SystemExit(f"no rows between {s.date()} and {e.date()}")
    label = f"cell{key}" if by == "cell" else f"region{key}"
    title_where = f"cell {key}" if by == "cell" else f"region {key} ({REGION_NAMES.get(key, '?')})"

    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2.2, 2.2, 2, 0.7]})
    ax1, ax2, ax3, ax4, ax5 = axes

    # 1. PM2.5, observed against both out-of-fold predictions and MERRA-2
    merra = merra_pm25_ugm3(w)
    ax1.plot(w.index, w["pm25"], color=C_OBS, lw=2.2, marker="o", ms=3.5, label="observed")
    ax1.plot(w.index, w["yhat_loyo"], color=C_LOYO, lw=1.8, label="OOF LOYO (2023 held out)")
    ax1.plot(w.index, w["yhat_loro"], color=C_LORO, lw=1.4, ls="--",
             label="OOF LORO (region held out)")
    ax1.plot(w.index, merra, color=C_MERRA, lw=1.4, ls=":", label="MERRA-2 surface PM2.5")
    ax1.axhline(25, color=C_GRID, lw=0.9, ls="--")
    ax1.axhline(50, color=C_GRID, lw=0.9, ls="--")
    ax1.set_ylabel("PM$_{2.5}$ (ug m$^{-3}$)")
    ax1.set_title("PM$_{2.5}$: observed vs out-of-fold", loc="left", fontsize=10)
    ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.9, ncol=2)
    top = max(float(w["pm25"].max()), float(np.nanmax(merra)),
              float(np.nanmax(w["yhat_loyo"])), float(np.nanmax(w["yhat_loro"])))
    ax1.set_ylim(0, top * 1.22)          # headroom so the legend clears the peak
    # positional, because every date repeats once per cell in the cell case
    pos = int(np.argmax(w["pm25"].to_numpy(float)))
    peak_day = w.index[pos]
    ax1.axvline(peak_day, color=C_OBS, lw=0.8, alpha=0.5)

    # 2. AOD
    shade_no_retrieval(ax2, w)
    ax2.plot(w.index, w["aod055"], color=C_AOD, marker="o", ms=4, lw=0, label="AOD 550 nm (MAIAC)")
    ax2.plot(w.index, w["aod_nbr_50km_d0"], color=C_AOD, lw=1.2, alpha=0.8,
             label="AOD, 50 km neighbourhood")
    ax2.plot(w.index, w["merra_totexttau"], color=C_MERRA, lw=1.2, ls="--",
             label="MERRA-2 total extinction")
    ax2.set_ylabel("AOD")
    ax2.set_title("AOD: retrieval, neighbourhood, and a reanalysis comparison", loc="left",
                  fontsize=10)
    ax2.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    aod_top = np.nanmax(np.concatenate([w["aod055"].to_numpy(float),
                                        w["aod_nbr_50km_d0"].to_numpy(float),
                                        w["merra_totexttau"].to_numpy(float)]))
    if np.isfinite(aod_top) and aod_top > 0:
        ax2.set_ylim(0, aod_top * 1.35)   # headroom for the legend

    # 3. Smoke and fire
    ax3.step(w.index, w["hms_density_max"], where="mid", color=C_SMOKE, lw=1.6,
             label="HMS smoke density (0/5/16/27)")
    ax3.set_yticks([0, 5, 16, 27])
    ax3.set_ylabel("HMS level")
    ax3.set_title("Smoke and fire: HMS density (left), fire activity (right)", loc="left",
                  fontsize=10, pad=16)
    ax3b = ax3.twinx()
    frp = w["fire_frp_sum_500km"].replace(0, np.nan)
    cnt = w["fire_count_250km"].replace(0, np.nan)
    ax3b.plot(w.index, frp, color="#8B0000", lw=1.3, marker="s", ms=3, label="fire FRP, 500 km")
    ax3b.plot(w.index, cnt, color="#B5651D", lw=1.1, ls="--", marker="^", ms=3,
              label="fire count, 250 km")
    ax3b.set_yscale("symlog", linthresh=1)
    ax3b.set_ylabel("FRP (MW) / count")
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    leg3 = ax3.legend(h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(0.0, 1.0),
                      fontsize=7.5, frameon=False, ncol=3, borderaxespad=0.1)
    leg3.set_zorder(20)                  # above the axes frame, which would cut the text
    ax3.set_ylim(0, 45)                  # above the 27 level, so the legend does not sit on it

    # 4. Boundary layer and wind speed
    ax4.plot(w.index, w["merra_pblh"], color=C_LOYO, lw=1.5, label="PBLH")
    ax4.set_ylabel("PBLH (m)")
    ax4.set_title("Meteorology: boundary layer (left), 10 m wind speed (right), "
                  "transport below", loc="left", fontsize=10)
    ax4b = ax4.twinx()
    speed = np.hypot(w["merra_u10m"].to_numpy(float), w["merra_v10m"].to_numpy(float))
    ax4b.plot(w.index, speed, color=C_LORO, lw=1.5, ls="--", label="wind speed")
    ax4b.set_ylabel("wind speed (m s$^{-1}$)")
    h1, l1 = ax4.get_legend_handles_labels()
    h2, l2 = ax4b.get_legend_handles_labels()
    ax4.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.5, framealpha=0.9)
    ax4.margins(y=0.28)

    # 5. Wind direction as unit arrows, pointing where the wind blows *toward*, so the
    # row reads as smoke transport. Speed is already in the panel above, so every arrow
    # is the same length and only the direction carries information.
    xs = mdates.date2num(w.index.to_pydatetime())
    u = w["merra_u10m"].to_numpy(float)
    v = w["merra_v10m"].to_numpy(float)
    norm = np.hypot(u, v)
    norm[norm == 0] = np.nan                       # calm days would divide by zero
    ax5.quiver(xs, np.zeros_like(xs), u / norm, v / norm, angles="uv", pivot="mid",
               color=C_LORO, scale_units="height", scale=3.2, width=0.004,
               headwidth=4.5, headlength=5.5)
    ax5.set_yticks([])
    ax5.set_ylim(-1, 1)
    ax5.set_ylabel("wind\ntoward", rotation=0, fontsize=8, labelpad=14)
    ax5.set_xlabel("date (UTC-6 day)")
    ax5.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, (e - s).days // 8)))
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    mode = " - QUICK (10% rows, quick output)" if quick else ""
    fig.suptitle(f"{s.date()} to {e.date()}  |  {title_where}  |  {run}{mode}\n"
                 f"OOF: LOYO (2023 held out), LORO (region held out)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    name = f"{s.date()}_{label}"
    png = outdir / f"{name}.png"
    fig.savefig(png, dpi=200)
    fig.savefig(outdir / f"{name}.pdf")
    plt.close(fig)

    w.assign(merra_pm25_ugm3=merra).to_csv(outdir / f"{name}_values.csv",
                                           float_format="%.3f")
    # the frame is indexed by date and several cells can share a day, so the peak is
    # taken positionally rather than by label
    peak = w.iloc[pos]
    merra_peak = merra.iloc[pos]
    return {"start": s.date(), "label": label, "peak_day": peak_day.date(),
            "observed_peak": float(peak["pm25"]),
            "loyo_at_peak": float(peak["yhat_loyo"]),
            "loro_at_peak": float(peak["yhat_loro"]),
            "merra_at_peak": float(merra_peak),
            "aod_on_peak": "yes" if bool(peak["aod_share"] > 0) else "no",
            # NaN only where there is no HMS file, so it must not be cast blindly
            "hms_at_peak": (int(peak["hms_density_max"])
                            if np.isfinite(peak["hms_density_max"]) else "no HMS file"),
            "png": png.name}


def plot_overview(frame: pd.DataFrame, episodes: list, run: str, quick: bool,
                  outdir: Path) -> Path:
    daily = frame.groupby("date")["pm25"].agg(["max", "median"])
    fig, ax = plt.subplots(figsize=(12, 4))
    for a, b in episodes:
        ax.axvspan(a, b, color=C_SMOKE, alpha=0.18, lw=0)
    ax.plot(daily.index, daily["max"], color=C_OBS, lw=1.2, label="Ontario daily maximum")
    ax.plot(daily.index, daily["median"], color=C_LOYO, lw=1.2, label="Ontario daily median")
    ax.axhline(25, color=C_GRID, lw=0.9, ls="--")
    ax.axhline(50, color=C_GRID, lw=0.9, ls="--")
    ax.set_ylabel("PM$_{2.5}$ (ug m$^{-3}$)")
    ax.set_xlabel("2023")
    ax.set_title("2023 development cells: daily maximum and median, shaded episodes"
                 f"{' - QUICK' if quick else ''}", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()
    p = outdir / f"overview_2023_{run}.png"
    fig.savefig(p, dpi=200)
    fig.savefig(outdir / f"overview_2023_{run}.pdf")
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="2023 smoke-episode diagnostics")
    ap.add_argument("--run", default="R1")
    ap.add_argument("--by", default="both", choices=["cell", "region", "both"])
    ap.add_argument("--threshold", type=float, default=50.0,
                    help="Ontario-wide daily maximum PM2.5 that starts an episode")
    ap.add_argument("--episodes", default=None, help="START:END[,START:END] to override")
    ap.add_argument("--oof-root", default=None, help="directory holding <run>/<scheme>/oof.parquet")
    ap.add_argument("--quick", action="store_true", help="use the quick-mode outputs")
    args = ap.parse_args()

    d = D.load()
    splits = pd.read_parquet(D.OUT / "splits.parquet")
    root = Path(args.oof_root) if args.oof_root else D.OUT / ("quick" if args.quick else "full")

    base = dev_2023(d, splits)
    merged = merge_predictions(base, load_oof(root, args.run, "loyo"),
                               load_oof(root, args.run, "loro"))
    # before anything is drawn: development cells, 2010-2023, nothing else
    D.assert_clean(merged, splits, "plot_episodes")
    frame = merged.set_index("date").sort_index()

    daily_max = frame.groupby("date")["pm25"].max()
    if args.episodes:
        episodes = parse_episodes(args.episodes)
        how = "from --episodes"
    else:
        episodes = detect_episodes(daily_max, args.threshold)
        how = f"auto, daily max > {args.threshold:g} ug/m3, padded 5 days"
    if not episodes:
        raise SystemExit(f"no episode found at threshold {args.threshold:g}")

    print(f"run {args.run}, source {root}")
    print(f"episodes ({how}): {len(episodes)}")
    print(f"{'window':<26}{'peak':>7}{'peak day':>13}{'region':>9}{'days>25':>9}")
    rows = []
    for a, b in episodes:
        w = frame[(frame.index >= a) & (frame.index <= b)]
        if w.empty:
            continue
        peak = w["pm25"].max()
        day = w["pm25"].idxmax()
        reg = int(w.loc[(w["pm25"] == peak), "region"].iloc[0])
        n25 = int((w.groupby("date")["pm25"].max() > 25).sum())
        window = f"{a.date()} to {b.date()}"
        print(f"{window:<26}{peak:>7.1f}{str(day.date()):>13}"
              f"{REGION_NAMES.get(reg, reg):>9}{n25:>9}")
        rows.append({"start": a.date(), "end": b.date(), "peak": round(float(peak), 2),
                     "peak_day": day.date(), "peak_region": REGION_NAMES.get(reg, reg),
                     "days_any_cell_gt25": n25})

    outdir = D.OUT / "figures" / "episodes" / args.run / ("quick" if args.quick else "full")
    outdir.mkdir(parents=True, exist_ok=True)

    ov = plot_overview(frame, episodes, args.run, args.quick, outdir)
    print(f"overview: {ov.name}")

    modes = ["cell", "region"] if args.by == "both" else [args.by]
    summary = []
    for a, b in episodes:
        w = frame[(frame.index >= a) & (frame.index <= b)]
        if w.empty:
            continue
        row = w.iloc[int(np.argmax(w["pm25"].to_numpy(float)))]
        key_cell = int(row["cell_id"])
        key_region = int(row["region"])
        for by, key in (("cell", key_cell), ("region", key_region)):
            if by not in modes:
                continue
            series = series_for(w, by, key)
            summary.append(plot_episode(series, by, key, (a, b), args.run, args.quick,
                                        outdir))

    pd.DataFrame(rows).to_csv(outdir / f"episodes_{args.run}.csv", index=False)
    if summary:
        s = pd.DataFrame(summary).sort_values(["start", "label"])
        s.to_csv(outdir / f"summary_{args.run}.csv", index=False)
        print()
        print(s.to_string(index=False))
    print(f"\nfigures in {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
