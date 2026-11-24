"""Every prediction month: keys unique, sorted (date, cell_id), each day = exact spine set."""
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
ROOT = Path(r"D:\ai-workspace\project")
spine = np.sort(pd.read_parquet(ROOT / "data/interim/spine/cells.parquet", columns=["cell_id"]).cell_id.to_numpy())
bad = []
for f in sorted((ROOT / "data/processed/prediction").glob("*/prediction_*.parquet")):
    t = pq.read_table(f, columns=["cell_id", "date"])
    c = t.column("cell_id").to_numpy(); d = t.column("date").to_numpy().astype("datetime64[D]")
    days = np.unique(d)
    n = len(spine)
    ok_shape = len(c) == len(days) * n
    ok_sorted = bool(np.all(np.diff(d.astype("int64")) >= 0))
    cc = c.reshape(len(days), n) if ok_shape else None
    ok_cells = ok_shape and bool((cc == spine[None, :]).all())
    ok_daydiff = bool(np.all(np.diff(days.astype("int64")) == 1))
    if not (ok_shape and ok_sorted and ok_cells and ok_daydiff):
        bad.append((f.name, ok_shape, ok_sorted, ok_cells, ok_daydiff))
print("months checked:", len(list((ROOT / "data/processed/prediction").glob("*/prediction_*.parquet"))), " failing:", bad)
