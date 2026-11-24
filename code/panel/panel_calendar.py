"""Calendar fields, applied by whoever needs them.\n\nNamed `panel_calendar`, not `calendar`: a module called calendar.py in this\ndirectory shadows the standard library one and breaks pandas (and `utils.py`)\nwhenever a script here is run.

The prediction table deliberately does not store these: year, month, doy and
dow are a pure function of `date`, and repeating four columns over 900 M rows
costs more than recomputing them. The training table may keep them because it
is small.
"""
from __future__ import annotations

import pandas as pd


def add_calendar(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    d = pd.to_datetime(df[date_col])
    out = df.copy()
    out["year"] = d.dt.year.astype("int16")
    out["month"] = d.dt.month.astype("int8")
    out["doy"] = d.dt.dayofyear.astype("int16")
    out["dow"] = d.dt.dayofweek.astype("int8") + 1     # 1 = Monday
    return out
