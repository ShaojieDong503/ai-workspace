#!/usr/bin/env python3
"""The model-input column list, declared once.

Both table builders and the table verifier import this. A column that is meant
to reach the model but is missing from one table is then a hard error rather
than something the intersection of two column sets can hide.
"""
from __future__ import annotations

MAIAC = ["aod055", "aod047", "injection_height", "smoke_aod055", "smoke_mask_frac"]

MERRA = [
    "merra_bccmass", "merra_occmass", "merra_so4cmass", "merra_sscmass25",
    "merra_ducmass25", "merra_bcsmass", "merra_ocsmass", "merra_so4smass",
    "merra_sssmass25", "merra_dusmass25", "merra_bcexttau", "merra_ocexttau",
    "merra_suexttau", "merra_duexttau", "merra_totexttau", "merra_pblh",
    "merra_qlml", "merra_tlml", "merra_ulml", "merra_vlml", "merra_speed",
    "merra_speedmax", "merra_u2m", "merra_v2m", "merra_u10m", "merra_v10m",
    "merra_u50m", "merra_v50m", "merra_u250", "merra_v250", "merra_u500",
    "merra_v500", "merra_u850", "merra_v850",
]

HMS = ["hms_any", "hms_n_polygons", "hms_density_max"]

FIRE = ["fire_dist_nearest_km"] + [f"fire_count_{r}km" for r in (25, 50, 100, 250, 500)] \
    + [f"fire_frp_sum_{r}km" for r in (25, 50, 100, 250, 500)]

CONTEXT = [
    "aod_nbr_25km_d0", "aod_nbr_25km_d0_n",
    "aod_nbr_50km_d0", "aod_nbr_50km_d0_n",
    "aod_nbr_100km_d0", "aod_nbr_100km_d0_n",
    "aod_nbr_50km_d1_3", "aod_nbr_50km_d1_3_n",
    "aod_last_obs", "aod_last_obs_age_d",
]

MODEL_INPUTS = MAIAC + MERRA + HMS + FIRE + CONTEXT      # 63

# keys, and columns that may be carried but are not model inputs
KEYS = ["cell_id", "date"]
ALLOWED_EXTRA = [
    "province_code", "province_state",
    "pm25", "n_stations", "n_hours_min",
    "year", "month", "doy", "dow",
    "aod055_npix", "maiac_npix_total", "aod055_centroid", "aod047_centroid",
    "maiac_covered",
    "dist_ontario_km", "on_region",
]


def check_columns(columns, label: str = "table") -> list[str]:
    """Return a list of problems: missing inputs, or columns nobody classified."""
    have = set(columns)
    problems = []
    missing = [c for c in MODEL_INPUTS if c not in have]
    if missing:
        problems.append(f"{label}: missing model inputs {missing}")
    unknown = sorted(have - set(MODEL_INPUTS) - set(KEYS) - set(ALLOWED_EXTRA))
    if unknown:
        problems.append(f"{label}: unclassified columns {unknown}")
    return problems


if __name__ == "__main__":
    print(f"model inputs: {len(MODEL_INPUTS)}")
    print(f"  MAIAC {len(MAIAC)}  MERRA {len(MERRA)}  HMS {len(HMS)}  "
          f"fire {len(FIRE)}  context {len(CONTEXT)}")