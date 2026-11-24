"""Brute-force recomputation of all 10 AOD context features for sampled cells,
including days 1-3 of a month (warm-up across the boundary). Uses only the
prediction table's own aod055 and the spine row/col; loops, no convolution."""
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
ROOT = Path(r"D:\ai-workspace\project")
cells = pd.read_parquet(ROOT / "data/interim/spine/cells.parquet")
rc = cells.set_index("cell_id")[["row", "col"]].astype("int64")
rng = np.random.default_rng(7)
out = []
for (y, m) in [(2018, 9), (2021, 1), (2013, 3)]:
    py, pm = (y, m - 1) if m > 1 else (y - 1, 12)
    a = pq.read_table(ROOT / f"data/processed/prediction/{py}/prediction_{py}_{pm:02d}.parquet", columns=["cell_id", "date", "aod055"]).to_pandas()
    b = pq.read_table(ROOT / f"data/processed/prediction/{y}/prediction_{y}_{m:02d}.parquet").to_pandas()
    aod = pd.concat([a, b[["cell_id", "date", "aod055"]]])
    aod = aod[aod.date >= b.date.min() - pd.Timedelta(days=7)]
    piv = aod.pivot(index="date", columns="cell_id", values="aod055")
    ids = piv.columns.to_numpy(); R = rc.loc[ids, "row"].to_numpy(); C = rc.loc[ids, "col"].to_numpy()
    obs_cells = (rng.choice(b[b.aod055.notna()].cell_id.unique(), 40))
    sample = np.concatenate([obs_cells, rng.choice(ids, 40)])
    for cid in sample:
        r0, c0 = rc.loc[cid]
        d2 = (R - r0) ** 2 + (C - c0) ** 2
        for day in [1, 2, 3, 15]:
            D = pd.Timestamp(y, m, day)
            row = b[(b.cell_id == cid) & (b.date == D)].iloc[0]
            ref = {}
            for rad in (25, 50, 100):
                v = piv.loc[D].to_numpy()[d2 <= (rad / 5) ** 2]
                v = v[np.isfinite(v)]
                ref[f"aod_nbr_{rad}km_d0"] = v.mean() if v.size else np.nan
                ref[f"aod_nbr_{rad}km_d0_n"] = float(v.size)
            vv = []
            for k in (1, 2, 3):
                v = piv.loc[D - pd.Timedelta(days=k)].to_numpy()[d2 <= 100]
                vv.append(v[np.isfinite(v)])
            vv = np.concatenate(vv)
            ref["aod_nbr_50km_d1_3"] = vv.mean() if vv.size else np.nan
            ref["aod_nbr_50km_d1_3_n"] = float(vv.size)
            ref["aod_last_obs"] = ref["aod_last_obs_age_d"] = np.nan
            for k in range(0, 8):
                v = piv.loc[D - pd.Timedelta(days=k), cid]
                if np.isfinite(v):
                    ref["aod_last_obs"], ref["aod_last_obs_age_d"] = v, float(k); break
            for k, v in ref.items():
                out.append((y, m, cid, day, k, float(v), float(row[k])))
R_ = pd.DataFrame(out, columns=["y", "m", "cell_id", "day", "col", "ref", "got"])
ok = (R_.ref.isna() & R_.got.isna()) | ((R_.ref - R_.got).abs() <= 1e-5 + 1e-5 * R_.ref.abs())
R_["ok"] = ok
R_.to_csv(ROOT / "audit/evidence/a17_context_bruteforce.csv", index=False)
print(R_.groupby("col").ok.agg(["size", "sum"]).to_string())
print(R_[~R_.ok].head(20).to_string())
