#!/usr/bin/env python3
"""Spine: the 163,787 in-square cells, indexed 0..n-1 in cell_id order.

Rule
----
Every table in this build is keyed by (date, cell_id) and laid out positionally:
a month is an array of shape (n_days, n_cells), rows date-major, columns in
cell_idx order. That makes joining sources a positional concatenation and gives
the (date, cell_id) sort for free, so the spine has to fix both the membership
and the ordering once.

Coordinates live only here. The big tables carry cell_id and date and nothing
else that is static, because repeating lat/lon over 900 M rows costs more than
joining it back.

Inputs are read-only; the only output is data/interim/spine/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\ai-workspace\project")
GRID = ROOT / "data" / "interim" / "grid" / "cell_table_square.parquet"
OUT = ROOT / "data" / "interim" / "spine"

KEEP = ["cell_id", "row", "col", "x_3347", "y_3347", "lat", "lon",
        "country", "province_state"]


def build(out_dir: Path = OUT) -> pd.DataFrame:
    cells = pd.read_parquet(GRID)
    n_all = len(cells)

    # The square is defined by cell centre; edge cells whose footprint only
    # grazes the boundary are excluded, as the build spec requires.
    cells = cells[cells["in_square"]].copy()
    cells = cells.sort_values("cell_id").reset_index(drop=True)

    cells["cell_idx"] = range(len(cells))
    cells["cell_id"] = cells["cell_id"].astype("int32")
    cells["row"] = cells["row"].astype("int16")
    cells["col"] = cells["col"].astype("int16")
    cells["country"] = cells["country"].astype("string")
    cells["province_state"] = cells["province_state"].astype("string")

    out_dir.mkdir(parents=True, exist_ok=True)
    cells[["cell_idx"] + KEEP].to_parquet(out_dir / "cells.parquet", index=False,
                                          compression="zstd")
    print(f"cells in table : {n_all:,}")
    print(f"in_square      : {len(cells):,}")
    print(f"cell_id range  : {cells.cell_id.min()} .. {cells.cell_id.max()}")
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the cell spine")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    cells = build(Path(args.out))

    manifest = {
        "script": "code/panel/spine.py",
        "inputs": [{"path": str(GRID), "bytes": GRID.stat().st_size}],
        "rows": int(len(cells)),
        "columns": ["cell_idx"] + KEEP,
        "dtypes": {c: str(t) for c, t in cells.dtypes.items()},
    }
    (Path(args.out) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
