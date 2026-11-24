"""MERRA-2 day window, every year: fit UTC6(D) = a*UTC(D-1) + b*UTC(D) + c*UTC(D+1)
on all pixels of one month per year, against the older UTC-day export. A UTC-6
window gives c ~ 0.2-0.3 and a ~ 0; a UTC window gives b ~ 1, c ~ 0."""
import numpy as np, rasterio, pandas as pd
ZIP = "/vsizip/D:/ai-workspace/project/data/SourceRegion_MERRA_2010_2024-20261103T023716Z-1-001.zip/SourceRegion_MERRA_2010_2024"
NEW = r"D:\ai-workspace\project\data\raw\merra2_utc6_ee"
rows = []
def stack(src, var):
    b = sorted((d, i + 1) for i, d in enumerate(src.descriptions) if d and d.endswith("_" + var))
    return [d[:8] for d, _ in b], src.read([i for _, i in b]).astype("float64")
for y in range(2010, 2025):
    for m in (4, 10):
        for coll, var in (("slv", "U10M"), ("slv", "V10M"), ("flx", "TLML")):
            with rasterio.open(f"{NEW}\source_region_merra_{coll}_daily_stack_{y}_{m:02d}.tif") as a, \
                 rasterio.open(f"{ZIP}/source_region_merra_{coll}_daily_stack_{y}_{m:02d}.tif") as b:
                dn, N = stack(a, var); do, O = stack(b, var)
                # align the two lattices by their geotransforms
                dc = int(round((a.transform.c - b.transform.c) / a.transform.a))
                dr = int(round((a.transform.f - b.transform.f) / a.transform.e))
                h = min(N.shape[1], O.shape[1] - dr); w = min(N.shape[2], O.shape[2] - dc)
                N = N[:, :h, :w]; O = O[:, dr:dr + h, dc:dc + w]
            N[N == -999] = np.nan; O[O == -999] = np.nan
            idx = {d: i for i, d in enumerate(do)}
            X, Y = [], []
            for i, d in enumerate(dn[1:-1], 1):
                if d in idx and 0 < idx[d] < len(do) - 1:
                    j = idx[d]
                    X.append(np.stack([O[j - 1].ravel(), O[j].ravel(), O[j + 1].ravel()], 1)); Y.append(N[i].ravel())
            X = np.concatenate(X); Y = np.concatenate(Y)
            ok = np.isfinite(X).all(1) & np.isfinite(Y)
            coef = np.linalg.lstsq(X[ok], Y[ok], rcond=None)[0]
            rows.append((y, m, var, dr, dc, *np.round(coef, 2)))
R = pd.DataFrame(rows, columns=["year", "month", "var", "row_off", "col_off", "w_Dm1", "w_D", "w_Dp1"])
print(R.to_string(index=False))
print("\nminimum D+1 weight:", R.w_Dp1.min(), " maximum |D-1| weight:", R.w_Dm1.abs().max())
