#!/usr/bin/env python3
"""Development results: runs x groups, AOD-missing split, per-year spread.

Rule
----
Grouped 10-fold is the scheme the parameters were chosen on, so its numbers are
biased by selection; leave-one-region-out and leave-one-year-out are the honest
development estimates. The report says so instead of presenting the three side by
side as equals.

Two splits are reported because they answer questions the pooled numbers cannot:

* **R1 vs R2 by `aod_missing`** -- whether MERRA-2 aerosol earns its place mainly on
  the ~73% of days with no MAIAC retrieval.
* **Leave-one-year-out per year, pooled with and without 2023** -- 2023 is the
  extreme smoke year and dominates that scheme's variance.

Everything is computed from stored OOF files; no model is fitted and no test row is
touched.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import cv
import data as D

OUT = D.OUT


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def collect(quick: bool):
    root = OUT / cv.cfg_quick(quick)
    pooled, folds = [], []
    for p in sorted(root.glob("*/*/metrics_pooled.parquet")):
        t = pd.read_parquet(p)
        t["run"] = p.parent.parent.name
        t["scheme"] = p.parent.name
        pooled.append(t)
    for p in sorted(root.glob("*/*/metrics_by_fold.parquet")):
        t = pd.read_parquet(p)
        t["run"] = p.parent.parent.name
        t["scheme"] = p.parent.name
        folds.append(t)
    if not pooled:
        raise SystemExit(f"no metrics under {root}")
    return (pd.concat(pooled, ignore_index=True).drop_duplicates(["run", "scheme", "group"]),
            pd.concat(folds, ignore_index=True))


def aod_split(root, d: pd.DataFrame, scheme: str = "group10") -> pd.DataFrame:
    """R1 vs R2, overall and with and without an AOD retrieval, from the stored OOF.

    Leave-one-region-out and leave-one-year-out are the schemes that carry weight:
    R1 was tuned on grouped 10-fold and R2 reuses those parameters, so grouped 10-fold
    flatters R1 and is reported for reference only.
    """
    ctx = d.set_index(["cell_id", "date"])["aod_missing"]
    rows = []
    for run in ("R1", "R2"):
        p = root / run / scheme / "oof.parquet"
        if not p.exists():
            continue
        oof = pd.read_parquet(p)
        idx = pd.MultiIndex.from_frame(oof[["cell_id", "date"]])
        oof["aod_missing"] = ctx.reindex(idx).to_numpy()
        groups = [("all", oof)] + [
            (f"aod_missing={int(m)}", g) for m, g in oof.groupby("aod_missing")]
        for label, g in groups:
            rows.append({"run": run, "scheme": scheme, "subgroup": label,
                         "n": int(len(g)),
                         "rmse": rmse(g["y"].to_numpy(float), g["yhat"].to_numpy(float))})
    return pd.DataFrame(rows)


def loyo_drop_2023(root, quick: bool) -> pd.DataFrame:
    """Overall RMSE per run, pooled, and with 2023 removed."""
    base = root
    rows = []
    for p in sorted(base.glob("*/loyo/oof.parquet")):
        run = p.parent.parent.name
        oof = pd.read_parquet(p)
        yr = pd.to_datetime(oof["date"]).dt.year
        rows.append({"run": run,
                     "n": int(len(oof)),
                     "rmse_all": rmse(oof["y"].to_numpy(float), oof["yhat"].to_numpy(float)),
                     "n_no2023": int((yr != 2023).sum()),
                     "rmse_no2023": rmse(oof.loc[yr != 2023, "y"].to_numpy(float),
                                         oof.loc[yr != 2023, "yhat"].to_numpy(float))})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarise development results")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = D.load_config()
    root = OUT / cv.cfg_quick(args.quick)
    pooled, folds = collect(args.quick)
    d = D.load()

    lines = ["# Development results", "",
             f"mode: {'quick' if args.quick else 'full'}", "",
             "**Grouped 10-fold is the scheme the parameters were chosen on, so its "
             "numbers are selection-biased.** Leave-one-region-out and "
             "leave-one-year-out are the honest development estimates; the test sets "
             "have the final word.", ""]

    for scheme in ("group10", "loro", "loyo"):
        t = pooled[pooled["scheme"] == scheme]
        if t.empty:
            continue
        label = " (tuning folds, selection-biased)" if scheme == "group10" else ""
        # plan 7.0 reports R2 too; it is blank (NaN) for gt25 and gt50 by design
        p = (t.pivot_table(index=["run", "group"], values=["rmse", "mae", "bias", "r2", "n"],
                           dropna=False)
             [["rmse", "mae", "bias", "r2", "n"]].round(3))
        lines += [f"## {scheme}{label}", "", "```", p.to_string(), "```", ""]

    for scheme, note in (("loro", "leave-one-region-out — the comparison that counts"),
                         ("loyo", "leave-one-year-out — the comparison that counts"),
                         ("group10", "grouped 10-fold — tuning folds, for reference only")):
        aod = aod_split(root, d, scheme)
        if aod.empty:
            continue
        p = aod.pivot_table(index="run", columns="subgroup",
                            values=["n", "rmse"]).round(3)
        lines += [f"## R1 vs R2 by AOD availability ({scheme})", "", note, "",
                  "`all` is every scored row; `aod_missing=0` has a retrieval, "
                  "`aod_missing=1` has none. R2 asks what MERRA-2 aerosol adds beyond "
                  "AOD, so the `aod_missing=1` column is the one that answers it.", "",
                  "```", p.to_string(), "```", ""]

    if not folds.empty:
        g = folds[(folds["group"] == "all") & (folds["scheme"] == "loyo")]
        if not g.empty:
            per_year = g.pivot_table(index="run", columns="fold", values="rmse").round(3)
            lines += ["## Leave-one-year-out, per held-out year", "",
                      "2023 is the extreme smoke year and dominates this scheme.", "",
                      "```", per_year.to_string(), "```", ""]

    with_year = loyo_drop_2023(root, args.quick)
    if not with_year.empty:
        lines += ["## Leave-one-year-out pooled, with and without 2023", "",
                  "```", with_year.round(3).to_string(index=False), "```", ""]

    if not folds.empty:
        g = folds[folds["group"] == "all"]
        spread = (g.groupby(["run", "scheme"])["rmse"]
                   .agg(["mean", "std", "min", "max", "count"]).round(3))
        lines += ["## Per-fold spread, overall RMSE", "", "```", spread.to_string(), "```"]

    out = OUT / ("results_dev_quick.md" if args.quick else "results_dev.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:24]))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())