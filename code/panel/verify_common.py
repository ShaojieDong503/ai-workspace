#!/usr/bin/env python3
"""Shared helpers for the verify_* scripts.

Every checker follows the same shape: read a written month, draw random
(cell_id, date) pairs from it, recompute those values from the raw source by a
*different* code path than the builder used, and compare at 1e-6. The point is
that the check shares no logic with the extractor, so a plausible-looking bug in
the extractor cannot also be present in the check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import ROOT

CELLS = ROOT / "data" / "interim" / "spine" / "cells.parquet"
BASE = ROOT / "data" / "interim" / "base"

problems: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        problems.append(msg)


def summary() -> int:
    print(f"\n{len(problems)} failure(s)")
    return 1 if problems else 0


def cells() -> pd.DataFrame:
    return pd.read_parquet(CELLS).set_index("cell_id")


def month_file(kind: str, year: int, month: int) -> Path:
    return BASE / kind / f"base_{kind}_{year}_{month:02d}.parquet"


def sample_pairs(kind: str, year: int, month: int, n: int, seed: int = 0) -> pd.DataFrame:
    """Random rows from a written month, restricted to cells with data."""
    f = month_file(kind, year, month)
    d = pd.read_parquet(f)
    return d.sample(n=min(n, len(d)), random_state=seed).reset_index(drop=True)


def compare(name: str, got: np.ndarray, want: np.ndarray, tol: float = 1e-6,
            rtol: float = 1e-9) -> None:
    """Equal within tol + rtol*|want|, with the NaN pattern also compared.

    A pure absolute tolerance is wrong for distances of ~1000 km and FRP sums of
    ~1e4 MW, where two correct implementations differ only in the floating-point
    summation order.
    """
    both = np.isfinite(got) & np.isfinite(want)
    only_got = int((np.isfinite(got) & ~np.isfinite(want)).sum())
    only_want = int((~np.isfinite(got) & np.isfinite(want)).sum())
    if both.any():
        diff = np.abs(got[both] - want[both])
        allowed = tol + rtol * np.abs(want[both])
        worst, ratio = float(diff.max()), float((diff / np.maximum(allowed, 1e-12)).max())
    else:
        worst, ratio = 0.0, 0.0
    ok = (ratio <= 1.0) and only_got == 0 and only_want == 0
    check(bool(ok), f"{name}: {both.sum()} matched, max|diff| {worst:.2e}, "
                    f"nan-mismatch got/want {only_got}/{only_want}")