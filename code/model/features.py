#!/usr/bin/env python3
"""The two feature groups, the two datasets, and the Ontario constancy check.

Rule
----
Groups are defined once, here, so no other module carries a hand-typed column
list. No calendar fields, no coordinates, no `province_state`.

`maiac_covered` is deliberately not a feature: it is constant over Ontario (every
Ontario cell has MAIAC coverage), so it would be a zero-variance column. That is
checked, not assumed -- see `check_maiac_covered_constant`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SATELLITE_AOD = (
    ["aod055", "aod047", "aod_missing"]
    + ["aod_nbr_25km_d0", "aod_nbr_25km_d0_n", "aod_nbr_50km_d0", "aod_nbr_50km_d0_n",
       "aod_nbr_100km_d0", "aod_nbr_100km_d0_n", "aod_nbr_50km_d1_3", "aod_nbr_50km_d1_3_n",
       "aod_last_obs", "aod_last_obs_age_d"]
    + ["injection_height", "smoke_aod055", "smoke_mask_frac"]
)                                                                          # 16

MERRA_AEROSOL = [
    "merra_bccmass", "merra_occmass", "merra_so4cmass", "merra_sscmass25", "merra_ducmass25",
    "merra_bcsmass", "merra_ocsmass", "merra_so4smass", "merra_sssmass25", "merra_dusmass25",
    "merra_bcexttau", "merra_ocexttau", "merra_suexttau", "merra_duexttau", "merra_totexttau",
]                                                                          # 15

MERRA_MET = [
    "merra_pblh", "merra_qlml", "merra_tlml", "merra_ulml", "merra_vlml", "merra_speed",
    "merra_speedmax", "merra_u10m", "merra_v10m", "merra_u50m", "merra_v50m",
    "merra_u250", "merra_v250", "merra_u500", "merra_v500", "merra_u850", "merra_v850",
]                                                                          # 17

FIRE = (["fire_dist_nearest_km"]
        + [f"fire_count_{r}km" for r in (25, 50, 100, 250, 500)]
        + [f"fire_frp_sum_{r}km" for r in (25, 50, 100, 250, 500)])        # 11

HMS = ["hms_any", "hms_n_polygons", "hms_density_max"]                      # 3

BASE = MERRA_MET + FIRE + HMS                                              # 31

GROUPS = {"satellite_aod": SATELLITE_AOD, "merra_aerosol": MERRA_AEROSOL, "base": BASE}

DATASETS = {
    "A": ("satellite_aod", "merra_aerosol", "base"),   # 62
    "C": ("satellite_aod", "base"),                    # 47
}


def columns(dataset: str) -> list[str]:
    cols: list[str] = []
    for g in DATASETS[dataset]:
        cols += GROUPS[g]
    return cols


def check_maiac_covered_constant(d: pd.DataFrame) -> dict:
    """`maiac_covered` must be constant over Ontario, which is why it is excluded."""
    v = d["maiac_covered"]
    if v.nunique(dropna=False) != 1:
        raise AssertionError(f"maiac_covered is not constant over Ontario: {sorted(v.unique())}")
    return {"maiac_covered_unique": int(v.nunique(dropna=False)), "value": int(v.iloc[0])}


def check_no_forbidden(cols: list[str]) -> None:
    forbidden = {"lat", "lon", "x_3347", "y_3347", "year", "month", "doy", "dow",
                 "province_state", "province_code", "maiac_covered", "pm25", "date", "cell_id"}
    bad = sorted(forbidden & set(cols))
    if bad:
        raise AssertionError(f"forbidden columns in a feature set: {bad}")