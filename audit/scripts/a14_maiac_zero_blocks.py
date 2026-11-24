"""Are the exact-zero 'no granule' blocks of a MAIAC tile-day always whole 256-row
chunks? If a block of all-five-bands-zero pixels shares a chunk with real data,
the extractor averages those zeros into AOD as genuine AOD = 0."""
import sys, json
import numpy as np, pandas as pd, rasterio
from rasterio.windows import Window
ROOT = r"D:\ai-workspace\project"
meta = json.load(open(ROOT + r"\data\interim\index\maiac_tiles.json"))
pairs = [("2011_06", "20110620"), ("2011_06", "20110610"), ("2012_02", "20120201")] + \
        [(a, b) for a, b in (x.split(":") for x in sys.argv[1:])]
for ym, day in pairs:
    for t in meta["tiles"]:
        f = ROOT + r"\data\raw\maiac_daily_stack\\" + t["file"].replace("2015_07", ym)
        r_off, c_off, h, w = t["window"]
        with rasterio.open(f) as s:
            desc = {d: i + 1 for i, d in enumerate(s.descriptions)}
            idx = [desc[f"{day}_{b}"] for b in ["AOD055", "AOD047", "INJECTION_HEIGHT", "SMOKE_AOD055", "SMOKE_MASK"]]
            a = s.read(idx, window=Window(c_off, r_off, w, h))
        allzero = np.all(a == 0, axis=0)                       # pixel: five bands exactly 0
        aod_zero = a[0] == 0
        n_chunks_full = n_chunks_part = zero_in_part = 0
        for r0 in range(0, h, 256):
            z = allzero[r0:r0 + 256]
            if z.all():
                n_chunks_full += 1
            elif z.any():
                n_chunks_part += 1
                zero_in_part += int(z.sum())
        print(f"{day} {t['tag']}: all-zero pixels {int(allzero.sum()):,} ({allzero.mean():.3%}); "
              f"AOD055==0 pixels {int(aod_zero.sum()):,}; whole all-zero chunks {n_chunks_full}; "
              f"partly all-zero chunks {n_chunks_part} holding {zero_in_part:,} zero pixels")
