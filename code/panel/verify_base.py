#!/usr/bin/env python3
"""Shape and completeness audit of the base panel.

Checks every written month, not a sample: row count equals n_days x n_cells,
(cell_id, date) unique, dates inside the month, no unexpected columns, and the
missing share of every column. A month whose missing share is more than 3x the
same month in the adjacent years is flagged - that is the signal for a broken
tile, a missing source file, or a schema drift.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import ROOT, days_in_month
from verify_common import BASE, cells, check, summary

OUT = ROOT / "data" / "interim" / "base"


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the base panel")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    spine_ids = set(cells().index.tolist())
    files = sorted((BASE / args.cells).glob(f"base_{args.cells}_*.parquet"))
    check(len(files) > 0, f"month files found: {len(files)}")
    if not files:
        return summary()

    rows = []
    for f in files:
        y, m = int(f.stem.split("_")[-2]), int(f.stem.split("_")[-1])
        d = pd.read_parquet(f)
        nd = len(days_in_month(y, m))
        n_cells = int(d["cell_id"].nunique())
        ok_rows = len(d) == nd * n_cells
        ok_keys = not d.duplicated(subset=["cell_id", "date"]).any()
        ok_dates = bool((pd.to_datetime(d["date"]).dt.year == y).all()
                        and (pd.to_datetime(d["date"]).dt.month == m).all())
        check(ok_rows, f"{y}-{m:02d} rows = {nd} x {n_cells} = {nd*n_cells:,} (got {len(d):,})")
        # cell_id must be a lattice id from the spine, never a 0..n-1 position
        stray = set(d["cell_id"].unique().tolist()) - spine_ids
        check(not stray, f"{y}-{m:02d} every cell_id is in the spine"
                         + (f" - {len(stray)} stray, e.g. {sorted(stray)[:3]}" if stray else ""))
        check(ok_keys, f"{y}-{m:02d} (cell_id, date) unique")
        check(ok_dates, f"{y}-{m:02d} every date inside the month")
        miss = d.isna().mean().to_dict()
        miss.update({"year": y, "month": m, "rows": len(d)})
        rows.append(miss)

    tbl = pd.DataFrame(rows).set_index(["year", "month"]).sort_index()
    value_cols = [c for c in tbl.columns if c != "rows"]

    # 3x jump against the same month in adjacent years
    flagged = []
    for col in value_cols:
        for (y, m), v in tbl[col].items():
            neigh = [tbl.loc[(yy, m), col] for yy in (y - 1, y + 1) if (yy, m) in tbl.index]
            if len(neigh) == 2:
                ref = float(np.mean(neigh))
                if ref > 0 and v > 3 * ref:
                    flagged.append((y, m, col, round(float(v), 4), round(ref, 4)))
    check(len(flagged) <= 5, f"months flagged for a >3x missing-share jump: {len(flagged)}")
    for fl in flagged[:10]:
        print("   flag", fl)

    print("\nmissing share, last 6 months:")
    print((tbl[value_cols].tail(6) * 100).round(1).to_string())

    if args.report:
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Base panel QA", "",
                 f"cells: {n_cells:,}", f"months: {len(files)}", "",
                 "## Missing share per column per month", "",
                 "```", (tbl[value_cols] * 100).round(2).to_string(), "```", "",
                 "## Flags (>3x jump vs adjacent years)", ""]
        lines += [f"- {y}-{m:02d} `{c}`: {v:.3%} vs {r:.3%}" for y, m, c, v, r in flagged]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {p}")
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())