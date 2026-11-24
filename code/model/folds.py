#!/usr/bin/env python3
"""The fold plan: the one place that says which cells or years a fold holds out.

Rule
----
`plan()` is imported by the CV engine and by the leakage test, so the test cannot
pass while the engine disagrees with it. It builds membership from the frozen
`folds.parquet` and `splits.parquet` and does no selection of its own.

Splits are at cell level, by author decision: no site grouping, no region
stratification of the folds, no row balancing.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import data as D


@dataclass(frozen=True)
class FoldSpec:
    label: str
    cells: tuple = ()
    years: tuple = ()


def plan(scheme: str, splits: pd.DataFrame, folds: pd.DataFrame) -> list[FoldSpec]:
    dev = splits[splits["role"] == "development"]
    if scheme == "group10":
        return [FoldSpec(f"fold{int(f)}", cells=tuple(sorted(g["cell_id"].astype(int))))
                for f, g in folds.groupby("fold")]
    if scheme == "loro":
        return [FoldSpec(f"region{int(r)}", cells=tuple(sorted(g["cell_id"].astype(int))))
                for r, g in dev.groupby("region")]
    if scheme == "loyo":
        return [FoldSpec(f"year{y}", years=(int(y),))
                for y in range(D.DEV_FIRST_YEAR, D.DEV_LAST_YEAR + 1)]
    raise ValueError(f"unknown scheme {scheme!r}")