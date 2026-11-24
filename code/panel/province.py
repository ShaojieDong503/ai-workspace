#!/usr/bin/env python3
"""Province / state as a compact column for the cell-day tables.

The name is static per cell, so the prediction table carries an int8 code and
the lookup lives in `data/interim/spine/province_lookup.csv`.

Water cells have no province of their own. They take the province of the nearest
land cell on the lattice, because six monitored cells sit on the St. Lawrence
and the Delaware (Montreal and Philadelphia) and leaving them as "(water)" would
split them out of any per-province grouping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from utils import ROOT

SPINE = ROOT / "data" / "interim" / "spine" / "cells.parquet"
STATIONS = ROOT / "data" / "interim" / "stations" / "stations.parquet"
LOOKUP = ROOT / "data" / "interim" / "spine" / "province_lookup.csv"

WATER_LABEL = "(water)"
PREFERRED = ["Ontario", "Québec", "Manitoba", "Nunavut"]

# stations.parquet carries Canadian provinces as abbreviations and US states in
# full; the lookup uses full names for both
ABBREV = {
    "ON": "Ontario", "QC": "Québec", "MB": "Manitoba", "NU": "Nunavut",
    "SK": "Saskatchewan", "AB": "Alberta", "BC": "British Columbia",
    "NS": "Nova Scotia", "NB": "New Brunswick", "NL": "Newfoundland and Labrador",
    "PE": "Prince Edward Island", "YT": "Yukon", "NT": "Northwest Territories",
}


def build_lookup(cells: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the code table, writing it only if it does not exist yet."""
    if cells is None:
        cells = pd.read_parquet(SPINE)
    if LOOKUP.exists():
        return pd.read_csv(LOOKUP, encoding="utf-8-sig")

    names = cells["province_state"].fillna(WATER_LABEL)
    present = sorted(names.unique())
    ordered = [p for p in PREFERRED if p in present]
    ordered += sorted(p for p in present if p not in ordered and p != WATER_LABEL)
    if WATER_LABEL in present:
        ordered.append(WATER_LABEL)

    lookup = pd.DataFrame({"province_code": range(len(ordered)), "province_state": ordered})
    lookup["province_code"] = lookup["province_code"].astype("int8")
    LOOKUP.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(LOOKUP, index=False, encoding="utf-8-sig")
    return lookup


def _nearest_land(names: pd.Series, cells: pd.DataFrame) -> pd.Series:
    """Replace (water) with the province of the nearest non-water cell."""
    out = names.copy()
    water = (names == WATER_LABEL).to_numpy()
    if not water.any():
        return out

    r0, c0 = int(cells["row"].min()), int(cells["col"].min())
    h = int(cells["row"].max()) - r0 + 1
    w = int(cells["col"].max()) - c0 + 1
    rr = (cells["row"].to_numpy() - r0).astype("int64")
    cc = (cells["col"].to_numpy() - c0).astype("int64")

    mask = np.ones((h, w), bool)              # True = water / empty
    mask[rr[~water], cc[~water]] = False
    grid = np.full((h, w), -1, "int64")
    grid[rr[~water], cc[~water]] = np.flatnonzero(~water)

    _, (iy, ix) = distance_transform_edt(mask, return_indices=True)
    donor = grid[iy[rr[water], cc[water]], ix[rr[water], cc[water]]]
    out.iloc[np.flatnonzero(water)] = names.to_numpy()[donor]
    return out


def _station_provinces(lookup: pd.DataFrame, water_cells) -> pd.Series:
    """cell_id -> province name, for water cells that hold a station.

    Land cells keep their geographic province even when a station disagrees
    (14 border-river cases) and even when two stations disagree (2 cells), so
    only water cells are considered here.
    """
    known = set(lookup["province_state"])
    st = pd.read_parquet(STATIONS)[["cell_id", "province_state"]].copy()
    st = st[st["cell_id"].isin(water_cells)]
    st["prov"] = st["province_state"].map(lambda p: ABBREV.get(p, p))

    unknown = sorted(set(st["prov"]) - known)
    if unknown:
        raise ValueError(f"station province not in the lookup: {unknown}")

    mixed = st.groupby("cell_id")["prov"].nunique()
    mixed = mixed[mixed > 1]
    if len(mixed):
        raise ValueError(f"cells with stations from two provinces: {list(mixed.index)[:5]}")

    return st.drop_duplicates("cell_id").set_index("cell_id")["prov"]


def cell_to_code(cells: pd.DataFrame | None = None) -> dict[int, int]:
    """cell_id -> province_code, water resolved to the nearest land cell."""
    if cells is None:
        cells = pd.read_parquet(SPINE)
    lookup = build_lookup(cells)
    mapping = dict(zip(lookup.province_state, lookup.province_code.astype(int)))
    cells = cells.reset_index(drop=True)
    names = _nearest_land(cells["province_state"].fillna(WATER_LABEL), cells)

    # a water cell that holds a station takes the station's own province: the
    # nearest land can be the wrong side of a river (Chester, PA vs Delaware)
    water = cells["province_state"].isna()
    stations = _station_provinces(lookup, set(cells.loc[water, "cell_id"]))
    hit = water & cells["cell_id"].isin(stations.index)
    if hit.any():
        names.loc[hit] = cells.loc[hit, "cell_id"].map(stations).to_numpy()

    return dict(zip(cells["cell_id"].astype(int), names.map(mapping).astype(int)))


def add_province(df: pd.DataFrame, code_col: str = "province_code",
                 name_col: str | None = None) -> pd.DataFrame:
    """Attach province_code (and optionally the land name) by cell_id."""
    codes = cell_to_code()
    out = df.copy()
    out[code_col] = out["cell_id"].map(codes).astype("int8")
    if name_col:
        lookup = build_lookup()
        names = dict(zip(lookup.province_code, lookup.province_state))
        out[name_col] = out[code_col].map(names)
    return out


if __name__ == "__main__":
    lk = build_lookup()
    print(lk.to_string(index=False))
    print(f"\nlookup at {LOOKUP}")