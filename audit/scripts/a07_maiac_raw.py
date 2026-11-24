"""Raw MAIAC encoding census: nodata, NaN vs exact-zero, value ranges, per band."""
from pathlib import Path
import numpy as np, rasterio
from rasterio.windows import Window
ROOT = Path(r"D:\ai-workspace\project")
for ym, tile in [("2015_07", "0000000000-0000002816"), ("2021_01", "0000002816-0000002816"), ("2023_06", "0000000000-0000005632")]:
    p = ROOT / f"data/raw/maiac_daily_stack/source_region_maiac_daily_stack_{ym}-{tile}.tif"
    with rasterio.open(p) as s:
        print(p.name, s.count, "bands", s.dtypes[0], "nodata", s.nodata, s.width, s.height)
        desc = {d: i + 1 for i, d in enumerate(s.descriptions)}
        days = sorted({d.split("_")[0] for d in s.descriptions})
        win = Window(0, 0, min(s.width, 2048), min(s.height, 2048))
        for day in days[::10]:
            out = []
            for b in ["AOD055", "AOD047", "INJECTION_HEIGHT", "SMOKE_AOD055", "SMOKE_MASK"]:
                a = s.read(desc[f"{day}_{b}"], window=win)
                fin = np.isfinite(a)
                z = int((a == 0).sum())
                vals = a[fin & (a != 0)]
                out.append(f"{b}: fin {fin.mean():.3f} zero {z/a.size:.4f} "
                           f"neg {int((a[fin] < 0).sum())} "
                           f"range {vals.min() if vals.size else np.nan:.3g}..{vals.max() if vals.size else np.nan:.3g} "
                           f"uniq {np.unique(a[fin]).size if b=='SMOKE_MASK' else ''}")
            print(" ", day, " | ".join(out))
