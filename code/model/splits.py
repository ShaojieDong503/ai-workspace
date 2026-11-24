#!/usr/bin/env python3
"""Frozen splits and folds (plan step 1).

Rule, by author decision
------------------------
- **Spatial test:** one cell drawn at random from each of the six regions, six
  cells in all, seeded by `seeds.split`. No row-count floor, no other constraint.
- **Calibration:** one cell drawn at random from each region, from the cells the
  test draw left, same seed. Drawn after the test cells, so it never changes them.
- **Folds:** development cells shuffled with `seeds.fold` and dealt round-robin
  into ten folds. No region stratification, no row balancing.
- **Temporal test:** all of 2024, which is never touched here.

The seed is fixed before the draw and the rule is run once. Re-drawing until the
test set looks balanced would be selection on the target.

Files are refused unless --force, because every later number depends on them.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import data as D

GRID = D.ROOT / "data" / "processed" / "grid" / "cells.parquet"


def nearest_dev_km(cells: pd.DataFrame, dev_cells: list[int]) -> pd.DataFrame:
    """Distance from each picked cell to the nearest development cell."""
    g = pd.read_parquet(GRID, columns=["cell_id", "x_3347", "y_3347"])
    g = g[g["cell_id"].isin(cells["cell_id"])]
    xy = g[["x_3347", "y_3347"]].to_numpy() / 1000.0
    dev = g[g["cell_id"].isin(dev_cells)][["x_3347", "y_3347"]].to_numpy() / 1000.0
    tree = cKDTree(dev)
    dist, _ = tree.query(xy, k=1)
    return pd.DataFrame({"cell_id": g["cell_id"].to_numpy(), "nearest_dev_km": dist})


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze splits and folds")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = D.load_config()
    D.OUT.mkdir(parents=True, exist_ok=True)
    paths = {"splits": D.OUT / "splits.parquet", "folds": D.OUT / "folds.parquet"}
    if not args.force:
        existing = [p.name for p in paths.values() if p.exists()]
        if existing:
            print(f"refusing to overwrite {existing}; pass --force")
            return 1

    d = D.load()
    print(f"sanity: {D.sanity_check(d)}")

    dev = d[d["year"].between(D.DEV_FIRST_YEAR, D.DEV_LAST_YEAR)]
    per_cell = (dev.groupby(["cell_id", "on_region"], as_index=False)
                   .agg(n_rows=("pm25", "size"))
                   .rename(columns={"on_region": "region"}))

    rng = np.random.default_rng(int(cfg["seeds"]["split"]))
    per_region = int(cfg["splits"]["test_cells_per_region"])
    test: list[int] = []
    for r in cfg["splits"]["regions"]:
        cells = np.sort(per_cell.loc[per_cell["region"] == r, "cell_id"].to_numpy())
        if len(cells) < per_region:
            raise ValueError(f"region {r} has {len(cells)} cells, need {per_region}")
        test += [int(c) for c in rng.choice(cells, size=per_region, replace=False)]
    if len(test) != int(cfg["splits"]["n_test_cells"]):
        raise AssertionError(f"picked {len(test)} test cells, config says "
                             f"{cfg['splits']['n_test_cells']}")

    # calibration: one cell per region from what the test draw left, same generator,
    # so the test cells above are unchanged by this rule
    calib_per_region = int(cfg["splits"]["calib_cells_per_region"])
    calib: list[int] = []
    for r in cfg["splits"]["regions"]:
        cells = np.sort(per_cell.loc[(per_cell["region"] == r) &
                                     ~per_cell["cell_id"].isin(test), "cell_id"].to_numpy())
        if len(cells) < calib_per_region + 1:
            raise ValueError(f"region {r} has {len(cells)} cells left after the test draw; "
                             f"need {calib_per_region} for calibration and 1 for development")
        calib += [int(c) for c in rng.choice(cells, size=calib_per_region, replace=False)]
    if len(calib) != int(cfg["splits"]["n_calib_cells"]):
        raise AssertionError(f"picked {len(calib)} calibration cells, config says "
                             f"{cfg['splits']['n_calib_cells']}")

    role = pd.Series("development", index=per_cell.index)
    role[per_cell["cell_id"].isin(test)] = "spatial_test"
    role[per_cell["cell_id"].isin(calib)] = "calibration"
    splits = (per_cell.assign(role=role)[["cell_id", "region", "n_rows", "role"]]
              .sort_values(["role", "region", "cell_id"]).reset_index(drop=True))
    splits.to_parquet(paths["splits"], index=False)

    dev_cells = sorted(splits.loc[splits["role"] == "development", "cell_id"].astype(int))
    n_folds = int(cfg["folds"]["n_folds"])
    shuffled = np.random.default_rng(int(cfg["seeds"]["fold"])).permutation(dev_cells)
    folds = pd.DataFrame({"cell_id": shuffled, "fold": np.arange(len(shuffled)) % n_folds})
    folds.to_parquet(paths["folds"], index=False)

    near = nearest_dev_km(splits, dev_cells)
    picked = (splits.merge(near, on="cell_id", how="left")
              .merge(dev.groupby("cell_id", as_index=False).agg(rows=("pm25", "size"),
                                                                gt25=("pm25", lambda v: int((v > 25).sum())),
                                                                gt50=("pm25", lambda v: int((v > 50).sum()))),
                     on="cell_id"))
    r = picked
    lines = ["# Splits and folds (cell level)", "",
             f"config `{D.config_hash(cfg)}`",
             f"seeds: split {cfg['seeds']['split']}, fold {cfg['seeds']['fold']}",
             f"test {len(test)} cells (one per region), calibration {len(calib)}, "
             f"development {len(dev_cells)}", "",
             "## Cells and rows by role and region", "",
             "| role | region | cells | rows | >25 | >50 | nearest dev cell (km) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for (role_, reg), g in r.groupby(["role", "region"]):
        # a development cell is its own nearest development cell; blank it
        if role_ == "development":
            dist = ""
        else:
            dist = f"{g['nearest_dev_km'].min():.1f}"
        lines.append(f"| {role_} | {int(reg)} | {g['cell_id'].nunique()} | {int(g['rows'].sum()):,} | "
                     f"{int(g['gt25'].sum()):,} | {int(g['gt50'].sum()):,} | {dist} |")
    lines += ["", "## Picked cells", "",
              "| cell | role | region | rows 2010-2023 | nearest dev cell (km) |",
              "|---|---|---|---:|---:|"]
    for row in r[r["role"] != "development"].sort_values(["role", "region", "cell_id"]).itertuples():
        lines.append(f"| {row.cell_id} | {row.role} | {int(row.region)} | {int(row.rows):,} | "
                     f"{row.nearest_dev_km:.1f} |")
    lines += ["", "## Fold sizes", "", "| fold | cells | rows | regions |", "|---:|---:|---:|---|"]
    fz = folds.merge(splits, on="cell_id").merge(dev, on="cell_id")
    for f, g in fz.groupby("fold"):
        lines.append(f"| {int(f)} | {g['cell_id'].nunique()} | {len(g):,} | "
                     f"{sorted(int(x) for x in g['region'].unique())} |")
    (D.OUT / "splits_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"test {len(test)} cells, calibration {len(calib)}, development {len(dev_cells)}")
    print(f"test cells: {sorted(test)}")
    print(f"calibration cells: {sorted(calib)}")
    print("wrote splits.parquet, folds.parquet, splits_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())