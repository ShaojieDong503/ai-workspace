#!/usr/bin/env python3
"""Per-cell Ontario geography: distance to Ontario and the thesis region.

`dist_ontario_km` is the distance in EPSG:3347 from the cell centre to the
nearest cell the spine labels Ontario, so 0 for Ontario cells. Ontario is the
spine's own `province_state`, which follows the political border through the
Great Lakes, so this agrees with `province_code` everywhere in the panel. The
lattice step is 5 km, so distances are quantised to the cell centres.

`on_region` is the thesis evaluation region (code = the thesis `fold_id`):
1 Toronto, 2 Central, 3 East, 4 West_SW, 5 West_NH, 6 North; 0 outside
Ontario. The thesis defined regions per monitor, by city, in
data/raw/thesis_regions/naps_id_to_region.csv (46 monitors). Monitors that
file does not list take the region of their city there; Morrisburg, the one
new city, is East (it lies between Cornwall and Ottawa). A monitored cell takes
its monitors' region. An unmonitored Ontario cell takes the region of the
nearest Ontario monitor; that is for mapping only, the thesis had no areal
regions.

A cell holding monitors takes their jurisdiction when they all agree, because
the training target there is their mean: a cell of Ontario monitors whose
centre lies across a border river (Sarnia) is Ontario, and a cell of Michigan
or Quebec monitors whose centre is on the Ontario side (Port Huron, Gatineau)
is not. A cell mixing both (Windsor with four Detroit monitors) keeps the
spine label. A cell moved out of Ontario gets its monitors' distance to the
nearest other Ontario cell.

Both select training monitors and test cells; neither is a model input. The
table is static per cell and lives beside the spine, so it is joined by
`cell_id` rather than carried in the 900 M-row prediction table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from utils import ROOT

SPINE = ROOT / "data" / "interim" / "spine" / "cells.parquet"
STATIONS = ROOT / "data" / "interim" / "stations" / "stations.parquet"
OUT = ROOT / "data" / "interim" / "spine" / "cell_ontario.parquet"
THESIS_REGIONS = ROOT / "data" / "raw" / "thesis_regions" / "naps_id_to_region.csv"
REGION_CODES = {"Toronto": 1, "Central": 2, "East": 3, "West_SW": 4, "West_NH": 5, "North": 6}
NEW_CITIES = {"Morrisburg": "East"}
TO_3347 = Transformer.from_crs("EPSG:4326", "EPSG:3347", always_xy=True).transform


def ontario_monitors(st: pd.DataFrame) -> pd.DataFrame:
    """Ontario monitors with their thesis region code."""
    lk = pd.read_csv(THESIS_REGIONS)
    assert dict(zip(lk["fold_region"], lk["fold_id"])) == REGION_CODES, lk[["fold_region", "fold_id"]]
    by_id = dict(zip(lk["naps_id"], lk["fold_region"]))
    city = lk.groupby("city")["fold_region"].unique()
    assert (city.str.len() == 1).all(), city[city.str.len() > 1]
    by_city = {**city.str[0].to_dict(), **NEW_CITIES}

    on = st[st["province_state"].eq("ON")].copy()
    naps_id = on["station_uid"].str.removeprefix("NAPS:").astype(int)
    name = naps_id.map(by_id).fillna(on["name"].map(by_city))
    assert name.notna().all(), on.loc[name.isna(), ["station_uid", "name"]]
    on["region"] = name.map(REGION_CODES).astype("int8")
    added = on[~naps_id.isin(by_id)]
    print(f"Ontario monitors: {len(on)}, {len(on) - len(added)} in the thesis lookup; "
          f"by city: {sorted(set(added['name']))}")
    return on


def main() -> int:
    cells = pd.read_parquet(SPINE, columns=["cell_id", "x_3347", "y_3347", "province_state"])
    xy = cells[["x_3347", "y_3347"]].to_numpy("float64")
    on = cells["province_state"].eq("Ontario").to_numpy(dtype=bool, na_value=False)
    dist, _ = cKDTree(xy[on]).query(xy)
    dist /= 1000.0

    st = pd.read_parquet(STATIONS, columns=["station_uid", "name", "cell_id", "lat", "lon",
                                            "province_state"])
    st["x"], st["y"] = TO_3347(st["lon"].to_numpy(), st["lat"].to_numpy())
    st["is_on"] = st["province_state"].eq("ON")
    pos_of = pd.Index(cells["cell_id"])

    # jurisdiction of monitored border cells
    ont = on.copy()
    on_idx = np.flatnonzero(on)
    tree = cKDTree(xy[on])
    for cid, r in st.groupby("cell_id")["is_on"].agg(["all", "any"]).iterrows():
        i = pos_of.get_loc(cid)
        mon = st[st["cell_id"] == cid]
        if r["all"] and not on[i]:
            ont[i], dist[i] = True, 0.0
        elif not r["any"] and on[i]:
            d, j = tree.query(mon[["x", "y"]].to_numpy(), k=2)
            other = np.where(on_idx[j[:, 0]] == i, d[:, 1], d[:, 0])
            ont[i], dist[i] = False, other.min() / 1000.0
        else:
            continue
        print(f"cell {cid}: monitors {mon['province_state'].unique().tolist()} -> "
              f"{'Ontario' if ont[i] else 'not Ontario'}, {dist[i]:.1f} km")

    # thesis regions: monitored cells from their monitors, the rest from the nearest monitor
    mon = ontario_monitors(st)
    region = np.zeros(len(cells), "int8")
    nearest = cKDTree(mon[["x", "y"]].to_numpy()).query(xy[ont])[1]
    region[ont] = mon["region"].to_numpy()[nearest]
    per_cell = mon.groupby("cell_id")["region"].unique()
    assert (per_cell.str.len() == 1).all(), per_cell[per_cell.str.len() > 1]
    i = pos_of.get_indexer(per_cell.index)
    keep = ont[i]
    region[i[keep]] = per_cell.str[0].to_numpy()[keep].astype("int8")
    dropped = mon[mon["cell_id"].isin(per_cell.index[~keep])]
    if len(dropped):
        print("Ontario monitors in cells counted outside Ontario:",
              dropped[["station_uid", "name"]].values.tolist())

    out = pd.DataFrame({"cell_id": cells["cell_id"].astype("int32"),
                        "dist_ontario_km": dist.astype("float32"),
                        "on_region": region})
    tmp = OUT.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(OUT)

    names = {v: k for k, v in REGION_CODES.items()}
    d = out["dist_ontario_km"]
    print(f"cells: {len(out):,}  Ontario: {ont.sum():,}  zero distance: {(d == 0).sum():,}")
    for r in (100, 200, 300):
        print(f"  cells within {r} km: {(d <= r).sum():,}")
    print("Ontario cells per region:",
          {names[k]: int(v) for k, v in out["on_region"].value_counts().sort_index().items() if k})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
