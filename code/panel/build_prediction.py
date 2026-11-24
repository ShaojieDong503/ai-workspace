#!/usr/bin/env python3
"""Prediction table: every cell x every day, gap-free on the model inputs.

One file per month, row groups of 1-2 M rows. The table carries cell_id, date,
province_code and the 63 model inputs - no coordinates and no calendar, which
are joined back from the spine or recomputed from the date.

A month is only written when

  * base and features have identical keys (a short features file would otherwise
    look like ordinary AOD gaps, because every context column allows NaN);
  * every model input from model_inputs.MODEL_INPUTS is present;
  * every column is either a model input, a key, or an allowed extra;
  * every column outside ALLOWED_NAN has zero NaN.

Anything else is a hard error for that month, logged with a traceback.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from model_inputs import MODEL_INPUTS, check_columns
from province import add_province
from utils import ROOT, Failures, fingerprint, months, rebuild_reason, setup_logging

BASE = ROOT / "data" / "interim" / "base" / "all"
FEAT = ROOT / "data" / "interim" / "features" / "all"
OUT = ROOT / "data" / "processed" / "prediction"

# model inputs that may legitimately be NaN, with the reason
ALLOWED_NAN = {
    "aod055": "raw MAIAC, missing under cloud and south of 41 N (no gap fill)",
    "aod047": "raw MAIAC, as above",
    "injection_height": "raw MAIAC, as above",
    "smoke_aod055": "raw MAIAC, as above",
    "smoke_mask_frac": "raw MAIAC, as above",
    "hms_any": "no HMS file for these 5 days",
    "hms_n_polygons": "no HMS file for these 5 days",
    "hms_density_max": "no file, or smoke with unknown density",
    "fire_dist_nearest_km": "no detection that day",
    "aod_last_obs": "no observation in the past 7 days",
    "aod_last_obs_age_d": "no observation in the past 7 days",
    "aod_nbr_25km_d0": "no observation within 25 km",
    "aod_nbr_25km_d0_n": "no observation within 25 km",
    "aod_nbr_50km_d0": "no observation within 50 km",
    "aod_nbr_50km_d0_n": "no observation within 50 km",
    "aod_nbr_100km_d0": "no observation within 100 km",
    "aod_nbr_100km_d0_n": "no observation within 100 km",
    "aod_nbr_50km_d1_3": "no observation within 50 km in D-1..D-3",
    "aod_nbr_50km_d1_3_n": "no observation within 50 km in D-1..D-3",
}


def contract(columns):
    """(must be complete, allowed NaN) for a month's columns."""
    required = [c for c in columns if c in MODEL_INPUTS and c not in ALLOWED_NAN]
    allowed = [c for c in columns if c in ALLOWED_NAN]
    return required, allowed


def assert_keys_match(base: pd.DataFrame, feat: pd.DataFrame) -> None:
    if len(base) != len(feat):
        raise RuntimeError(f"base {len(base):,} rows vs features {len(feat):,}")
    b = base[["cell_id", "date"]].to_numpy()
    f = feat[["cell_id", "date"]].to_numpy()
    if not np.array_equal(b, f):
        raise RuntimeError("base and features have different (cell_id, date) keys")


def build_month(base_path: Path, feat_path: Path, dst: Path, inputs: dict) -> int:
    base = pd.read_parquet(base_path)
    feat = pd.read_parquet(feat_path)
    assert_keys_match(base, feat)

    d = base.merge(feat, on=["cell_id", "date"], how="inner")
    if len(d) != len(base):
        raise RuntimeError(f"merge kept {len(d):,} of {len(base):,} rows")

    problems = check_columns(d.columns)
    if problems:
        raise RuntimeError("; ".join(problems))

    required, _ = contract(d.columns)
    nan = {c: int(d[c].isna().sum()) for c in required}
    broken = {c: v for c, v in nan.items() if v}
    if broken:
        raise RuntimeError(f"gap-free contract broken: {broken}")

    d = add_province(d, name_col=None)
    cols = list(d.columns)
    cols.remove("province_code")
    cols.insert(2, "province_code")
    d = d[cols].sort_values(["date", "cell_id"]).reset_index(drop=True)

    dst.parent.mkdir(parents=True, exist_ok=True)          # out/YYYY may not exist
    tbl = pa.Table.from_pandas(d, preserve_index=False)
    md = dict(tbl.schema.metadata or {})
    md[b"inputs"] = json.dumps(inputs).encode()
    tmp = dst.with_suffix(".tmp")
    pq.write_table(tbl.replace_schema_metadata(md), tmp, compression="zstd",
                   row_group_size=1_000_000)
    tmp.replace(dst)
    return len(d)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the prediction table")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--month", type=int, nargs="*", default=None)
    ap.add_argument("--check", action="store_true", help="verify only, do not write")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out / "build.log")
    fails = Failures()

    for y, m in months():
        if args.month and m not in args.month:
            continue
        fb = BASE / f"base_all_{y}_{m:02d}.parquet"
        ff = FEAT / f"features_{y}_{m:02d}.parquet"
        if not (fb.exists() and ff.exists()):
            continue
        dst = out / str(y) / f"prediction_{y}_{m:02d}.parquet"
        inputs = {"base": fingerprint(fb), "features": fingerprint(ff)}
        if not args.check:
            reason = rebuild_reason(dst, inputs)
            if reason is None:
                continue
            log.info("prediction %d-%02d: %s", y, m, reason)
        rows = fails.guard(log, f"prediction {y}-{m:02d}", build_month, fb, ff, dst, inputs)
        if rows is not None:
            log.info("  rows %s", f"{rows:,}")

    fails.write(out / "failures.json")
    log.info("failures: %d", len(fails.items))
    return 1 if fails.items else 0


if __name__ == "__main__":
    raise SystemExit(main())