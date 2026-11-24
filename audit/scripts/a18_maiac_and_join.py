"""(a) MAIAC footprint means recomputed for non-Jan/Jul months: pixel centres
tested against the cell's EPSG:3347 box (|x-cx|<=DX/2, |y-cy|<=|DY|/2).
(b) Training rows vs prediction rows, every shared column, three months."""
import json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyproj, rasterio
from rasterio.windows import Window
ROOT = Path(r"D:\ai-workspace\project")
DX, DY = 4999.827, 4998.864
cells = pd.read_parquet(ROOT / "data/interim/spine/cells.parquet").set_index("cell_id")
to_ll = pyproj.Transformer.from_crs(3347, 4326, always_xy=True)
to_xy = pyproj.Transformer.from_crs(4326, 3347, always_xy=True)
meta = json.load(open(ROOT / "data/interim/index/maiac_tiles.json"))
rng = np.random.default_rng(11)
res = []
for ym in ["2018_09", "2023_06", "2014_04"]:
    y = ym[:4]
    p = pq.read_table(ROOT / f"data/processed/prediction/{y}/prediction_{ym}.parquet",
                      columns=["cell_id", "date", "aod055", "aod047", "smoke_aod055", "smoke_mask_frac", "aod055_npix", "maiac_npix_total"]).to_pandas()
    cand = p[(p.aod055.notna())].sample(15, random_state=1)
    cand = pd.concat([cand, p[p.cell_id.isin(cells[cells.lat > 41.2].index)].sample(10, random_state=2)])
    for r in cand.itertuples():
        cx, cy = cells.loc[r.cell_id, ["x_3347", "y_3347"]]
        lons, lats = to_ll.transform([cx - DX/2, cx + DX/2, cx - DX/2, cx + DX/2], [cy - DY/2, cy - DY/2, cy + DY/2, cy + DY/2])
        acc = {b: [] for b in ["AOD055", "AOD047", "SMOKE_AOD055", "SMOKE_MASK"]}; tot = 0
        for t in meta["tiles"]:
            with rasterio.open(ROOT / "data/raw/maiac_daily_stack" / t["file"].replace("2015_07", ym)) as s:
                T = s.transform
                c0 = int(np.floor((min(lons) - T.c) / T.a)) - 1; c1 = int(np.ceil((max(lons) - T.c) / T.a)) + 1
                r0 = int(np.floor((max(lats) - T.f) / T.e)) - 1; r1 = int(np.ceil((min(lats) - T.f) / T.e)) + 1
                # restrict to the indexed window of this tile
                wr0, wc0, wh, ww = t["window"]
                c0, c1 = max(c0, wc0), min(c1, wc0 + ww); r0, r1 = max(r0, wr0), min(r1, wr0 + wh)
                if c1 <= c0 or r1 <= r0:
                    continue
                cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
                X, Y = to_xy.transform(T.c + (cc + .5) * T.a, T.f + (rr + .5) * T.e)
                inside = (np.abs(X - cx) <= DX / 2) & (np.abs(Y - cy) <= DY / 2)
                if not inside.any():
                    continue
                tot += int(inside.sum())
                desc = {d: i + 1 for i, d in enumerate(s.descriptions)}
                day = pd.Timestamp(r.date).strftime("%Y%m%d")
                for b in acc:
                    a = s.read(desc[f"{day}_{b}"], window=Window(c0, r0, c1 - c0, r1 - r0))
                    acc[b].append(a[inside])
        ref = {k: (np.nanmean(np.concatenate(v)) if v and np.isfinite(np.concatenate(v)).any() else np.nan) for k, v in acc.items()}
        npix = int(np.isfinite(np.concatenate(acc["AOD055"])).sum()) if acc["AOD055"] else 0
        for col, key in [("aod055", "AOD055"), ("aod047", "AOD047"), ("smoke_aod055", "SMOKE_AOD055"), ("smoke_mask_frac", "SMOKE_MASK")]:
            res.append((ym, r.cell_id, str(r.date)[:10], col, float(ref[key]), float(getattr(r, col))))
        res.append((ym, r.cell_id, str(r.date)[:10], "aod055_npix", float(npix), float(r.aod055_npix)))
        res.append((ym, r.cell_id, str(r.date)[:10], "maiac_npix_total", float(tot), float(r.maiac_npix_total)))
R = pd.DataFrame(res, columns=["ym", "cell_id", "date", "col", "ref", "got"])
R["ok"] = (R.ref.isna() & R.got.isna()) | ((R.ref - R.got).abs() <= 1e-5 + 1e-6 * R.ref.abs())
R.to_csv(ROOT / "audit/evidence/a18_maiac.csv", index=False)
print("MAIAC:\n", R.groupby("col").ok.agg(["size", "sum"]).to_string()); print(R[~R.ok].to_string())

# (b) training vs prediction
t = pd.read_parquet(ROOT / "data/processed/target/target_table.parquet")
for ym in ["2012_08", "2019_02", "2024_12"]:
    y, m = int(ym[:4]), int(ym[5:])
    p = pd.read_parquet(ROOT / f"data/processed/prediction/{y}/prediction_{ym}.parquet")
    tt = t[(t.year == y) & (t.month == m)]
    j = tt.merge(p, on=["cell_id", "date"], suffixes=("_t", "_p"))
    shared = [c for c in p.columns if c not in ("cell_id", "date")]
    bad = [c for c in shared if not ((j[c + "_t"].astype("float64") == j[c + "_p"].astype("float64")) |
                                     (j[c + "_t"].isna() & j[c + "_p"].isna())).all()]
    print(ym, "training rows", len(tt), "joined", len(j), "columns compared", len(shared), "differing:", bad)
