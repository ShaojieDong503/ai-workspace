#!/usr/bin/env python3
"""The six reporting groups and four metrics, computed once.

Rule
----
Every run reports the same groups the same way. R2 is not defined for the two
extreme groups: with 1,318 and 134 development rows it is unstable, and the plan
says to leave it out there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS = {
    "all": lambda d: np.ones(len(d), bool),
    "le25": lambda d: (d["y"] <= 25).to_numpy(),
    "gt25": lambda d: (d["y"] > 25).to_numpy(),
    "gt50": lambda d: (d["y"] > 50).to_numpy(),
    "north": lambda d: (d["on_region"] == 6).to_numpy(),
    "heavy_smoke": lambda d: (d["hms_density_max"] == 27).to_numpy(),
}

NO_R2 = ("gt25", "gt50")


def _one(y: np.ndarray, yhat: np.ndarray) -> dict:
    if len(y) == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan}
    err = yhat - y
    r2 = np.nan
    if len(y) > 1 and np.var(y) > 0:
        r2 = float(1.0 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))
    return {"n": int(len(y)), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err)), "r2": r2}


def compute(frame: pd.DataFrame) -> pd.DataFrame:
    """Metrics per group. `frame` needs y, yhat, on_region, hms_density_max."""
    rows = []
    for name, mask in GROUPS.items():
        m = mask(frame)
        r = _one(frame.loc[m, "y"].to_numpy(float), frame.loc[m, "yhat"].to_numpy(float))
        if name in NO_R2:
            r.pop("r2")
        rows.append({"group": name, **r})
    return pd.DataFrame(rows)


def by_fold(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for f, g in frame.groupby("fold"):
        t = compute(g)
        t["fold"] = str(f)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)