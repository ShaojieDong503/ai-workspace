"""Independent check of on_region / dist_ontario_km for monitored cells."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(r"D:\ai-workspace\project")
lk = pd.read_csv(ROOT / "data/raw/thesis_regions/naps_id_to_region.csv", dtype={"naps_id": str})
lk["naps_id"] = lk.naps_id.str.zfill(6)
st = pd.read_parquet(ROOT / "data/interim/stations/stations.parquet")
co = pd.read_parquet(ROOT / "data/processed/grid/cell_ontario.parquet")
cells = pd.read_parquet(ROOT / "data/interim/spine/cells.parquet")
t = pd.read_parquet(ROOT / "data/processed/target/target_table.parquet", columns=["cell_id", "on_region", "dist_ontario_km"]).drop_duplicates("cell_id")
on = st[st.province_state == "ON"].copy()
on["nid"] = on.station_uid.str.replace("NAPS:", "")
print("thesis ids not among current ON stations:", sorted(set(lk.naps_id) - set(on.nid)))
on = on.merge(lk[["naps_id", "fold_id", "city"]], left_on="nid", right_on="naps_id", how="left")
city = lk.groupby("city").fold_id.agg(lambda s: sorted(set(s)))
print("cities with >1 thesis region:", city[city.str.len() > 1].to_dict())
on["exp"] = on.fold_id.fillna(on.name.map(city.str[0]))
on.loc[on.name == "Morrisburg", "exp"] = 3
on = on.merge(t, on="cell_id", how="left")
bad = on[on.exp != on.on_region]
print("ON stations whose cell region != expected:\n", bad[["station_uid", "name", "cell_id", "exp", "on_region"]].to_string(index=False))
# name in stations vs thesis city for thesis stations
mm = on[on.fold_id.notna() & (on.name != on.city)]
print("thesis stations whose stations.parquet name differs from thesis city:", mm[["station_uid", "name", "city"]].values.tolist())
# non-ON monitors in a region>0 cell
nono = st[st.province_state != "ON"].merge(t, on="cell_id")
print("non-ON monitors in on_region>0 cells:", nono[nono.on_region > 0][["station_uid", "name", "cell_id"]].values.tolist())
# dist_ontario_km recomputed by brute force for monitored cells
onc = cells[cells.province_state == "Ontario"][["x_3347", "y_3347"]].to_numpy()
mc = cells[cells.cell_id.isin(t.cell_id)].merge(t, on="cell_id")
d = np.array([np.sqrt(((onc - p) ** 2).sum(1)).min() / 1000 for p in mc[["x_3347", "y_3347"]].to_numpy()])
mc["d_ref"] = d
dif = mc[(mc.d_ref - mc.dist_ontario_km).abs() > 0.01]
print("monitored cells where dist differs from brute force (spine-Ontario):\n",
      dif[["cell_id", "province_state", "dist_ontario_km", "d_ref"]].to_string(index=False))
print("Ontario-region row/cell counts:", pd.read_parquet(ROOT / "data/processed/target/target_table.parquet", columns=["on_region"]).on_region.value_counts().sort_index().to_dict())
