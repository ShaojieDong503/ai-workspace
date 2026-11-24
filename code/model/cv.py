#!/usr/bin/env python3
"""One cross-validation engine for every run and scheme.

Rule
----
A fold holds out whole cells (grouped 10-fold, leave-one-region-out) or whole
years (leave-one-year-out), trains on the rest of development, and scores only the
held-out cells or year. Rows come from `data.fold_rows`, which asserts both frames
are clean and disjoint; the engine never selects rows itself.

Every run directory carries a `run_key.json` -- config hash, parameter hash,
feature hash, quick flag. Resume is refused when the key differs, because a fold
file written under different settings would be mixed into the same OOF file
without a trace. Each fold is written to `.tmp` and renamed, so a crash cannot
leave a half-written parquet that the next run accepts.

Seeds are stable across processes (`zlib.crc32`, not `hash`), so a resumed run
reproduces the same inner cell set as an uninterrupted one.
"""
from __future__ import annotations

import hashlib
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import data as D
import features as F
import folds as FLD
import metrics as M
import models

OUT = D.OUT

# editing the fit code invalidates old folds just as changing the config does
CODE_FILES = ("models.py", "cv.py", "data.py", "folds.py", "features.py")


def code_hash() -> str:
    """Hash of the source of the modules that decide how a fold is fitted."""
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update((Path(__file__).with_name(name)).read_bytes())
    return h.hexdigest()[:16]


def fold_seed(cfg: dict, which: str, label: str) -> int:
    """Stable per-fold seed: crc32 is identical across processes."""
    return int(cfg["seeds"][which]) + zlib.crc32(label.encode()) % 1000


def inner_split(train: pd.DataFrame, cfg: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frac = float(cfg["early_stopping"]["inner_fraction"])
    rng = np.random.default_rng(seed)
    cells = train[["cell_id", "on_region"]].drop_duplicates()
    chosen = []
    for r, g in cells.groupby("on_region"):
        k = max(1, int(round(len(g) * frac)))
        chosen += list(g.sample(n=min(k, len(g)), random_state=int(rng.integers(1 << 31)))["cell_id"])
    inner = train[train["cell_id"].isin(chosen)]
    rest = train[~train["cell_id"].isin(chosen)]
    if inner.empty or rest.empty:
        raise AssertionError("inner split produced an empty side")
    return rest, inner


def subsample(train: pd.DataFrame, cfg: dict, quick: bool, seed: int) -> pd.DataFrame:
    if not quick:
        return train
    return train.sample(frac=float(cfg["quick"]["train_row_fraction"]), random_state=seed)


def round_cap(cfg: dict, quick: bool) -> int:
    return int(cfg["quick"]["max_rounds"]) if quick else int(cfg["early_stopping"]["max_rounds"])


def cfg_quick(quick: bool) -> str:
    return "quick" if quick else "full"


def fit_folds(specs, fn, fold_workers: int = 1):
    """Run `fn(spec)` for every spec, concurrently, and return results in spec order.

    Threads, not processes: the fitted libraries release the GIL, so threads do run
    in parallel and share the loaded frame without copying it. Results are collected
    by submission order -- never completion order -- so per-fold RMSE, tree counts
    and predictions are identical however the threads happen to finish.
    """
    specs = list(specs)
    if fold_workers <= 1 or len(specs) <= 1:
        return [fn(s) for s in specs]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(fold_workers, len(specs))) as pool:
        futures = [pool.submit(fn, s) for s in specs]
        return [f.result() for f in futures]


def default_fold_workers(n_folds: int) -> int:
    """One thread per fold, capped by the CPUs this process may use."""
    return max(1, min(int(n_folds), models.detect_cpus()))


def run_dir(tag: str, scheme: str, quick: bool) -> Path:
    base = OUT / cfg_quick(quick) / tag / scheme
    base.mkdir(parents=True, exist_ok=True)
    return base


def fold_rows(d, splits, cfg, spec, quick):
    train, score = D.fold_rows(d, splits, cells=spec.cells or None, years=spec.years or None)
    train = subsample(train, cfg, quick, fold_seed(cfg, "fold", spec.label))
    rest, inner = inner_split(train, cfg, fold_seed(cfg, "es_inner", spec.label))
    return rest, inner, score


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def run_key(cfg: dict, cols: list[str], params: dict, quick: bool) -> dict:
    return {"config": D.config_hash(cfg), "features": _h(cols),
            "params": _h(params), "quick": quick, "code": code_hash(),
            "target": D.fingerprint(D.TARGET)}


def _check_key(base: Path, key: dict, fresh: bool) -> None:
    p = base / "run_key.json"
    folds = list(base.glob("fold_*.parquet"))
    if folds and not p.exists():
        # folds written before the run-key mechanism: provenance unknown, so they
        # are the stale ones B3 was about
        if not fresh:
            raise SystemExit(f"refusing to resume {base}: {len(folds)} fold files but no "
                             f"run_key.json; pass --fresh to discard them")
        for f in folds:
            f.unlink()
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if old != key:
            if not fresh:
                raise SystemExit(
                    f"refusing to resume {base}: run key differs\n  old {old}\n  new {key}\n"
                    f"pass --fresh to discard the old folds")
            for f in base.glob("fold_*.parquet"):
                f.unlink()
    tmp = base / "run_key.tmp"
    tmp.write_text(json.dumps(key, indent=2), encoding="utf-8")
    tmp.replace(p)


def dev_keys(d: pd.DataFrame, splits: pd.DataFrame) -> set:
    dev = set(splits.loc[splits["role"] == "development", "cell_id"].astype(int))
    m = d[d["cell_id"].isin(dev) & d["year"].between(D.DEV_FIRST_YEAR, D.DEV_LAST_YEAR)]
    return set(zip(m["cell_id"].astype(int), m["date"].dt.strftime("%Y-%m-%d")))


def leak_check(oof: pd.DataFrame, d: pd.DataFrame, splits: pd.DataFrame) -> None:
    """Before metrics: right cells, right years, every development row exactly once."""
    dev = set(splits.loc[splits["role"] == "development", "cell_id"].astype(int))
    bad = set(oof["cell_id"].astype(int)) - dev
    if bad:
        raise AssertionError(f"OOF holds non-development cells: {sorted(bad)[:5]}")
    years = set(pd.to_datetime(oof["date"]).dt.year)
    if years - set(range(D.DEV_FIRST_YEAR, D.DEV_LAST_YEAR + 1)):
        raise AssertionError(f"OOF holds years outside development: {sorted(years)}")
    if int(oof.duplicated(["cell_id", "date"]).sum()):
        raise AssertionError("OOF repeats (cell, date) rows")
    got = set(zip(oof["cell_id"].astype(int), pd.to_datetime(oof["date"]).dt.strftime("%Y-%m-%d")))
    want = dev_keys(d, splits)
    if got != want:
        raise AssertionError(f"OOF does not cover development exactly: "
                             f"{len(want - got)} missing, {len(got - want)} extra")


def run_cv(*, tag: str, dataset: str, scheme: str, cfg: dict, d: pd.DataFrame,
           splits: pd.DataFrame, model: str, params: dict, cols: list[str],
           quick: bool, fresh: bool = False, fold_workers: int = 1) -> Path:
    fold_workers = max(1, int(fold_workers))
    models.set_parallel(1, fold_workers)
    base = run_dir(tag, scheme, quick)
    _check_key(base, run_key(cfg, cols, params, quick), fresh)
    specs = FLD.plan(scheme, splits, folds=pd.read_parquet(OUT / "folds.parquet"))
    cap = round_cap(cfg, quick)
    patience = int(cfg["early_stopping"]["patience"])
    t0 = time.time()

    def fit_one(spec):
        """Fit one fold and write it atomically. Skips nothing: the caller filtered."""
        part = base / f"fold_{spec.label}.parquet"
        train, inner, score = fold_rows(d, splits, cfg, spec, quick)
        yhat, n_trees, _not_stopped = models.fit_predict(
            model, cfg, params, cols, train, inner, score, cap, patience)
        tmp = part.with_suffix(".tmp")
        pd.DataFrame({"cell_id": score["cell_id"].to_numpy(), "date": score["date"].to_numpy(),
                      "fold": spec.label, "y": score["pm25"].to_numpy(),
                      "yhat": np.asarray(yhat, float),
                      "n_trees": np.int32(n_trees)}).to_parquet(tmp, index=False)
        tmp.replace(part)
        return spec.label

    # "finished folds are skipped" is decided before anything is submitted
    todo = [s for s in specs if not (base / f"fold_{s.label}.parquet").exists()]
    fit_folds(todo, fit_one, fold_workers)

    oof = pd.concat([pd.read_parquet(p) for p in sorted(base.glob("fold_*.parquet"))],
                    ignore_index=True)
    leak_check(oof, d, splits)
    oof.to_parquet(base / "oof.parquet", index=False)
    # counted per FOLD from the stored tree counts, so a resumed run reports every
    # fold. Counting rows here would report ~145,000 instead of ~10, because every
    # scored cell-day repeats its fold's tree count.
    if model == "rf":
        not_stopped = None            # Random Forest has no early stopping to miss
    else:
        per_fold = oof.groupby("fold")["n_trees"].first()
        not_stopped = int((per_fold > cap - patience).sum())

    ctx = d.set_index(["cell_id", "date"])[["on_region", "hms_density_max"]]
    idx = pd.MultiIndex.from_frame(oof[["cell_id", "date"]])
    frame = oof.assign(on_region=ctx["on_region"].reindex(idx).to_numpy(),
                       hms_density_max=ctx["hms_density_max"].reindex(idx).to_numpy())
    M.compute(frame).to_parquet(base / "metrics_pooled.parquet", index=False)
    M.by_fold(frame).to_parquet(base / "metrics_by_fold.parquet", index=False)

    import sklearn, lightgbm, xgboost
    meta = {"tag": tag, "dataset": dataset, "scheme": scheme, "model": model,
            "quick": quick, "config_hash": D.config_hash(cfg), "features_hash": _h(cols),
            "params_hash": _h(params), "n_features": len(cols),
            "target_fingerprint": D.fingerprint(D.TARGET), "folds": len(specs),
            "rows_oof": int(len(oof)), "folds_not_stopped": not_stopped,
            "cap": int(cap), "wall_seconds": round(time.time() - t0, 1),
            **models.parallelism(),
            "threads_per_fit": models.resolve_threads(cfg, models.model_key(model)),
            "folds_fitted_this_call": len(todo),
            "versions": {"lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__,
                         "sklearn": sklearn.__version__, "numpy": np.__version__,
                         "pandas": pd.__version__}}
    (base / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return base


def dry_run(scheme: str, cfg: dict, d: pd.DataFrame, splits: pd.DataFrame,
            quick: bool = False) -> list[dict]:
    """Fold plan -> train/inner/score keys, with every assertion, without fitting."""
    out = []
    for spec in FLD.plan(scheme, splits, folds=pd.read_parquet(OUT / "folds.parquet")):
        train, inner, score = fold_rows(d, splits, cfg, spec, quick)
        k = lambda f: set(zip(f["cell_id"].astype(int), f["date"].dt.strftime("%Y-%m-%d")))
        out.append({"label": spec.label,
                    "train": k(train), "inner": k(inner), "score": k(score),
                    "inner_cells": int(inner["cell_id"].nunique()),
                    "score_cells": int(score["cell_id"].nunique())})
    return out