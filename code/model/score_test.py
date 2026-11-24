#!/usr/bin/env python3
"""Score the held-out test sets once. Guarded; Phase 3 only.

Rule
----
Refuses unless OPEN_TEST exists and --open-test is passed.

The spatial test is **six cells, two of them 5 km from a development cell**, so it
is reported as "held-out monitor cells within the existing network", never as
unseen locations. Three presentations, all defined before the test sets are opened:

* every spatial-test cell, pooled;
* each cell separately, with its row count, bias and distance to the nearest
  development cell, because six cells make a pooled number hide a lot;
* the subset more than 10 km from any development cell -- the closest thing to an
  unseen location this design allows.

Confidence intervals are cell-block bootstrap (resampled whole cells, seeded) for
every metric and group, and interval coverage is reported per group.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import data as D
import features as F
import metrics as M

OUT = D.OUT
OPEN = OUT / "OPEN_TEST"
GRID = D.ROOT / "data" / "processed" / "grid" / "cells.parquet"

RUNS = {"R1": ("lgbm", "A"), "R2": ("lgbm", "C"), "R3": ("xgb", "A"), "R4": ("rf", "A")}
STATS = ("rmse", "mae", "bias", "r2")
FAR_KM = 10.0
# pre-registered in plan 7.2, fixed before any test row is scored
FAR_CELLS = {910214, 895350, 958101, 924048}


def output_paths(run: str) -> list:
    return [OUT / f"test_results_{run}.csv", OUT / f"test_per_cell_{run}.csv"]


def supersede(paths, reason: str) -> None:
    """Keep earlier results under a UTC timestamp instead of overwriting them."""
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for p in paths:
        if p.exists():
            p.rename(p.with_name(f"{p.stem}_superseded_{stamp}{p.suffix}"))
    print(f"superseded earlier results under {stamp}: {reason}")


def log_scoring(run: str, model_file: str, reason: str, superseded: bool) -> None:
    """One line per scoring, so the history is auditable and append-only."""
    from datetime import datetime, timezone
    entry = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "run": run, "model_file": model_file, "reason": reason,
             "superseded": bool(superseded)}
    with open(OUT / "test_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def guard(open_test: bool) -> None:
    if not OPEN.exists():
        raise SystemExit(f"refusing: {OPEN} does not exist")
    if not open_test:
        raise SystemExit("refusing: pass --open-test")


def cell_distances(splits: pd.DataFrame) -> pd.DataFrame:
    """Distance from every test cell to the nearest development cell, in km."""
    g = pd.read_parquet(GRID, columns=["cell_id", "x_3347", "y_3347"])
    dev = set(splits.loc[splits["role"] == "development", "cell_id"].astype(int))
    test = set(splits.loc[splits["role"] == "spatial_test", "cell_id"].astype(int))
    xy = g[["x_3347", "y_3347"]].to_numpy() / 1000.0
    ids = g["cell_id"].to_numpy()
    tree = cKDTree(xy[np.isin(ids, list(dev))])
    test_xy = xy[np.isin(ids, list(test))]
    dist, _ = tree.query(test_xy, k=1)
    return pd.DataFrame({"cell_id": ids[np.isin(ids, list(test))], "nearest_dev_km": dist})


def bootstrap(frame: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = frame["cell_id"].unique()
    acc = []
    for i in range(draws):
        pick = rng.choice(cells, size=len(cells), replace=True)
        sub = pd.concat([frame[frame["cell_id"] == c] for c in pick], ignore_index=True)
        m = M.compute(sub)
        m["draw"] = i
        acc.append(m)
    s = pd.concat(acc, ignore_index=True)
    rows = []
    for g, gg in s.groupby("group"):
        row = {"group": g}
        for st in STATS:
            row[f"{st}_lo"] = float(np.nanpercentile(gg[st], 2.5))
            row[f"{st}_hi"] = float(np.nanpercentile(gg[st], 97.5))
        rows.append(row)
    return pd.DataFrame(rows)


def coverage(frame: pd.DataFrame, q: float) -> pd.DataFrame:
    inside = np.abs(frame["y"] - frame["yhat"]) <= q
    rows = []
    for name in M.GROUPS:
        m = M.GROUPS[name](frame)
        rows.append({"group": name,
                     "coverage90": float(inside[m].mean()) if m.any() else float("nan"),
                     "n": int(m.sum())})
    return pd.DataFrame(rows)


def as_metric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The four columns metrics.compute needs. The target column is `pm25`."""
    return pd.DataFrame({"cell_id": df["cell_id"].to_numpy(dtype="int64"),
                         "y": df["pm25"].to_numpy(float),
                         "yhat": df["yhat"].to_numpy(float),
                         "on_region": df["on_region"].to_numpy(),
                         "hms_density_max": df["hms_density_max"].to_numpy()})


def per_cell(frame: pd.DataFrame, dist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, g in frame.groupby("cell_id"):
        err = g["yhat"].to_numpy(float) - g["y"].to_numpy(float)
        rows.append({"cell_id": int(cid), "n": int(len(g)),
                     "rmse": float(np.sqrt(np.mean(err ** 2))),
                     "bias": float(np.mean(err)), "mean_y": float(g["y"].mean()),
                     "n_gt25": int((g["y"] > 25).sum())})
    out = pd.DataFrame(rows)
    return out.merge(dist, on="cell_id", how="left")


def summarise_set(name: str, frame: pd.DataFrame, q: float, draws: int, seed: int) -> pd.DataFrame:
    t = M.compute(frame)
    t = t.merge(bootstrap(frame, draws, seed), on="group")
    t = t.merge(coverage(frame, q), on="group")
    t["set"] = name
    t["cells"] = int(frame["cell_id"].nunique())
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the test sets")
    ap.add_argument("--run", default="R1", choices=sorted(RUNS))
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--open-test", action="store_true")
    ap.add_argument("--supersede", action="store_true",
                    help="keep earlier results instead of overwriting them")
    ap.add_argument("--reason", default="", help="required with --supersede")
    args = ap.parse_args()
    guard(args.open_test)

    paths = output_paths(args.run)
    existing = [p for p in paths if p.exists()]
    if existing:
        if not args.supersede:
            raise SystemExit(f"refusing: {existing[0].name} already exists. The test sets "
                             f"are scored once; pass --supersede --reason \"...\" to keep "
                             f"the old results under a timestamp")
        if not args.reason.strip():
            raise SystemExit("refusing: --supersede requires --reason")
        supersede(paths, args.reason)

    model, dataset = RUNS[args.run]
    cfg = D.load_config()
    d = D.load()
    splits = pd.read_parquet(OUT / "splits.parquet")
    cols = F.columns(dataset)
    meta = json.loads((OUT / f"final_{args.run}.json").read_text(encoding="utf-8"))
    q = float(meta["q90"])
    if int(meta.get("n_features", -1)) != len(cols):
        raise SystemExit(f"refusing: {meta.get('model_file')} was trained on "
                         f"{meta.get('n_features')} features, this run has {len(cols)}")
    predict = load_model(model, cols, meta["model_file"])
    seed = int(cfg["seeds"]["bootstrap"])

    dist = cell_distances(splits)
    spatial = D.phase3_test_rows(d, splits, "spatial")
    spatial = spatial.merge(dist, on="cell_id", how="left")
    spatial["cell_id"] = spatial["cell_id"].astype("int64")
    spatial["yhat"] = predict(spatial)

    temporal = D.phase3_test_rows(d, splits, "temporal").assign(yhat=lambda f: predict(f))
    far = set(dist.loc[dist["nearest_dev_km"] > FAR_KM, "cell_id"].astype(int))
    if far != FAR_CELLS:
        raise SystemExit(
            f"refusing: the more-than-{FAR_KM:.0f} km subset computed from the grid is "
            f"{sorted(far)}, but plan 7.2 pre-registers {sorted(FAR_CELLS)}. The grid or "
            f"the splits changed; fix the cause, do not edit the constant")
    spatial["cell_id"] = spatial["cell_id"].astype(int)

    sets = {
        "spatial_all": spatial,
        f"spatial_gt{FAR_KM:.0f}km": spatial[spatial["cell_id"].isin(far)],
        "temporal_2024": temporal,
    }
    rows = [summarise_set(name, as_metric_frame(f), q, int(args.draws), seed)
            for name, f in sets.items() if len(f)]
    out = pd.concat(rows, ignore_index=True)
    paths[0].write_text(out.to_csv(index=False), encoding="utf-8")

    cells = per_cell(as_metric_frame(spatial), dist)
    paths[1].write_text(cells.to_csv(index=False), encoding="utf-8")
    log_scoring(args.run, meta["model_file"], args.reason or "first scoring", bool(existing))

    print(out.to_string(index=False))
    if not cells.empty:
        print("\nper spatial-test cell:")
        print(cells.round(2).to_string(index=False))
        print(f"\n{len(far)} of {len(dist)} spatial-test cells are more than "
              f"{FAR_KM:.0f} km from any development cell; the rest share an airshed "
              f"with a training cell and are reported separately.")
    print("\n2024 has too few smoke days to support the heavy-smoke group; that row is "
          "reported but should not be interpreted.")
    return 0


def load_model(model: str, cols: list[str], model_file: str):
    path = OUT / model_file
    if not path.exists():
        raise SystemExit(f"refusing: {path} does not exist")
    if model == "lgbm":
        import lightgbm as lgb
        b = lgb.Booster(model_file=str(path))
        if b.num_feature() != len(cols):
            raise SystemExit(f"refusing: {model_file} has {b.num_feature()} features, "
                             f"dataset has {len(cols)}")
        return lambda df: b.predict(df[cols])
    if model == "xgb":
        import xgboost as xgb
        b = xgb.Booster(model_file=str(path))
        if b.num_features() != len(cols):
            raise SystemExit(f"refusing: {model_file} has {b.num_features()} features, "
                             f"dataset has {len(cols)}")
        return lambda df: b.predict(xgb.DMatrix(df[cols]))
    import joblib
    rf = joblib.load(path)
    if int(rf.n_features_in_) != len(cols):
        raise SystemExit(f"refusing: {model_file} has {rf.n_features_in_} features, "
                         f"dataset has {len(cols)}")
    return lambda df: rf.predict(df[cols].to_numpy())


if __name__ == "__main__":
    raise SystemExit(main())