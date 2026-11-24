#!/usr/bin/env python3
"""Loader, derived columns, and the only three filters that select rows.

Rule
----
Ontario rows only, defined as `on_region > 0` -- never `province_state`, which
disagrees for three cells because the region follows the monitor and the province
follows the polygon (data/processed/README.md §5).

Every row a model fits on comes from `training_rows`; every row a metric sees
comes from `scoring_rows`; the held-out test rows come only from
`phase3_test_rows`, which the Phase 3 scripts guard. The two development filters
assert that no spatial-test cell, no calibration cell and no 2024 row can pass,
so a mistake in a later module cannot leak silently.

`scoring_rows` restricts to development cells *and* 2010-2023 before intersecting
the fold's held-out cells or year. That matters: without the development-cell
restriction, grouped 10-fold would score the held-out cells' 2024 rows, and
leave-one-year-out would score every Ontario cell that year, test cells included.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# project root: $PM25_ROOT when set (e.g. a copy on SciNet $SCRATCH), otherwise the
# folder two levels above this file, so the code runs unchanged on Windows and Linux
ROOT = Path(os.environ.get("PM25_ROOT", Path(__file__).resolve().parents[2]))
PANEL = ROOT / "code" / "panel"
sys.path.insert(0, str(PANEL))
from utils import fingerprint  # noqa: E402  (content hash, reused from the panel)

TARGET = ROOT / "data" / "processed" / "target" / "target_table.parquet"
CONFIG = Path(__file__).with_name("config.json")
OUT = ROOT / "data" / "interim" / "model"

DEV_FIRST_YEAR = 2010
DEV_LAST_YEAR = 2023
TEST_YEAR = 2024

# plan §7, Ontario 2010-2023, before any cell is removed
SANITY = {"rows": 197_612, "gt25": 1_318, "gt50": 134, "north": 25_005}


def load_config() -> dict:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def load() -> pd.DataFrame:
    """Ontario rows with the derived columns, and the aod_missing assertion."""
    d = pd.read_parquet(TARGET)
    d = d[d["on_region"] > 0].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["year"] = d["date"].dt.year.astype("int16")

    d["aod_missing"] = (d["aod055_npix"] == 0).astype("int8")
    flagged = d["aod_missing"].to_numpy() == 1
    is_na = d["aod055"].isna().to_numpy()
    if not np.array_equal(flagged, is_na):
        bad = int((flagged != is_na).sum())
        raise AssertionError(f"aod_missing disagrees with aod055 on {bad} rows")
    return d.reset_index(drop=True)


def sanity_check(d: pd.DataFrame) -> dict:
    """Fail unless the plan §7 counts hold exactly."""
    o = d[d["year"].between(DEV_FIRST_YEAR, DEV_LAST_YEAR)]
    got = {
        "rows": int(len(o)),
        "gt25": int((o["pm25"] > 25).sum()),
        "gt50": int((o["pm25"] > 50).sum()),
        "north": int((o["on_region"] == 6).sum()),
    }
    bad = {k: (got[k], SANITY[k]) for k in SANITY if got[k] != SANITY[k]}
    if bad:
        raise AssertionError(f"plan §7 counts differ (got, expected): {bad}")
    return got


def _roles(splits: pd.DataFrame) -> tuple[set, set, set]:
    test = set(splits.loc[splits["role"] == "spatial_test", "cell_id"].astype(int))
    calib = set(splits.loc[splits["role"] == "calibration", "cell_id"].astype(int))
    dev = set(splits.loc[splits["role"] == "development", "cell_id"].astype(int))
    return test, calib, dev


def assert_clean(rows: pd.DataFrame, splits: pd.DataFrame, label: str) -> None:
    test, calib, _ = _roles(splits)
    if rows.empty:
        raise AssertionError(f"{label}: empty selection")
    n_test = int(rows["cell_id"].isin(test).sum())
    n_calib = int(rows["cell_id"].isin(calib).sum())
    n_2024 = int((rows["year"] >= TEST_YEAR).sum())
    if n_test or n_calib or n_2024:
        raise AssertionError(f"{label}: test cells {n_test}, calibration cells {n_calib}, "
                             f"rows >= {TEST_YEAR}: {n_2024}")


def training_rows(d: pd.DataFrame, splits: pd.DataFrame, drop_cells=None,
                  drop_years=None) -> pd.DataFrame:
    """Every training row in the package comes from here."""
    _, _, dev = _roles(splits)
    m = d["cell_id"].isin(dev) & d["year"].between(DEV_FIRST_YEAR, DEV_LAST_YEAR)
    out = d[m]
    if drop_cells:
        out = out[~out["cell_id"].isin(set(drop_cells))]
    if drop_years:
        out = out[~out["year"].isin(set(drop_years))]
    assert_clean(out, splits, "training_rows")
    return out


def scoring_rows(d: pd.DataFrame, splits: pd.DataFrame, cells=None,
                 years=None) -> pd.DataFrame:
    """Every row a metric sees in Phases 1-2 comes from here.

    Development cells only, 2010-2023 only, then intersected with the fold's
    held-out cells or year. Passing neither is an error, not "everything".
    """
    if not cells and not years:
        raise ValueError("scoring_rows needs cells or years; refusing to score everything")
    if years:
        stray = set(int(y) for y in years) - set(range(DEV_FIRST_YEAR, DEV_LAST_YEAR + 1))
        if stray:
            raise ValueError(f"scoring years outside {DEV_FIRST_YEAR}-{DEV_LAST_YEAR}: "
                             f"{sorted(stray)}")
    _, _, dev = _roles(splits)
    m = d["cell_id"].isin(dev) & d["year"].between(DEV_FIRST_YEAR, DEV_LAST_YEAR)
    if cells:
        m &= d["cell_id"].isin(set(cells))
    if years:
        m &= d["year"].isin(set(years))
    out = d[m]
    assert_clean(out, splits, "scoring_rows")
    return out


def fold_rows(d: pd.DataFrame, splits: pd.DataFrame, cells=None, years=None):
    """(train, score) for one fold, with every assertion applied.

    The scoring frame must be exactly the held-out cells or year -- not a subset,
    and not more. That is the check that would have caught the original rewrite's
    fault, where leave-one-year-out scored every Ontario cell in the year.
    """
    if cells:
        dev = set(splits.loc[splits["role"] == "development", "cell_id"].astype(int))
        stray = set(int(c) for c in cells) - dev
        if stray:
            raise ValueError(f"held-out list contains non-development cells: {sorted(stray)[:5]}")
    if years:
        stray = set(int(y) for y in years) - set(range(DEV_FIRST_YEAR, DEV_LAST_YEAR + 1))
        if stray:
            raise ValueError(f"held-out years outside {DEV_FIRST_YEAR}-{DEV_LAST_YEAR}: {sorted(stray)}")

    train = training_rows(d, splits, drop_cells=cells, drop_years=years)
    score = scoring_rows(d, splits, cells=cells, years=years)
    if cells:
        want = set(cells) & set(splits.loc[splits["role"] == "development", "cell_id"].astype(int))
        got = set(score["cell_id"].astype(int))
        if got != want:
            raise AssertionError(f"scoring cells {sorted(got)} != held-out {sorted(want)}")
    if years:
        got = set(score["year"].unique())
        if got != set(years):
            raise AssertionError(f"scoring years {sorted(got)} != held-out {years}")
    return train, score


def phase3_test_rows(d: pd.DataFrame, splits: pd.DataFrame, which: str) -> pd.DataFrame:
    """The held-out rows, for Phase 3 only. Callers must be guarded."""
    test, _, _ = _roles(splits)
    if which == "spatial":
        out = d[d["cell_id"].isin(test) & d["year"].between(DEV_FIRST_YEAR, DEV_LAST_YEAR)]
    elif which == "temporal":
        out = d[d["year"] == TEST_YEAR]
    else:
        raise ValueError(f"unknown test set {which!r}")
    if out.empty:
        raise AssertionError(f"phase3_test_rows({which}): empty")
    return out


if __name__ == "__main__":
    cfg = load_config()
    d = load()
    print(f"Ontario rows {len(d):,}, cells {d['cell_id'].nunique()}, "
          f"config {config_hash(cfg)}")
    print(f"sanity: {sanity_check(d)}")