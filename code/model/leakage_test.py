#!/usr/bin/env python3
"""Leakage test: every scheme, through the engine's own dry run.

Rule
----
The evidence is `cv.dry_run`, not a second implementation of the fold plan: for
every scheme and fold, the engine reports the keys it would train on, early-stop
on and score. This test asserts

* the three sets are pairwise disjoint;
* the early-stopping set never touches a held-out or test cell. It is a subset of
  the training set, and `data.fold_rows` already keeps the training set clean, so
  pairwise disjointness plus a clean training set is what makes this hold;
* across a scheme the scored keys are exactly the development rows, each once.

Plus three deliberate breakages through the real functions, so the guards are
shown to fire rather than assumed to.

Nothing here computes a metric.
"""
from __future__ import annotations

import argparse

import pandas as pd

import cv
import data as D

SCHEMES = ("group10", "loro", "loyo")
LOG = D.OUT / "leakage_test.log"


def say(msg: str) -> None:
    print(msg)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fold-level leakage test")
    ap.add_argument("--scheme", default="all", choices=("all",) + SCHEMES)
    args = ap.parse_args()

    LOG.write_text("", encoding="utf-8")
    cfg = D.load_config()
    d = D.load()
    splits = pd.read_parquet(D.OUT / "splits.parquet")
    want = cv.dev_keys(d, splits)
    say(f"development rows available: {len(want):,}")
    say(f"roles: {splits.groupby('role')['cell_id'].nunique().to_dict()}")

    failures = 0
    schemes = SCHEMES if args.scheme == "all" else (args.scheme,)
    for scheme in schemes:
        rows = cv.dry_run(scheme, cfg, d, splits, quick=False)
        score_union, dup, cross = set(), 0, 0
        for r in rows:
            cross += len(r["train"] & r["score"]) + len(r["inner"] & r["score"]) \
                + len(r["train"] & r["inner"])
            score_union |= r["score"]
        total = sum(len(r["score"]) for r in rows)
        dup = total - len(score_union)
        ok = (cross == 0 and dup == 0 and score_union == want)
        failures += 0 if ok else 1
        say(f"{scheme:8s} folds {len(rows):2d}  scored {total:,}  pairwise overlaps {cross}  "
            f"duplicates {dup}  missing {len(want - score_union)}  "
            f"extra {len(score_union - want)}  {'OK' if ok else 'FAIL'}")

    say("")
    say("breakages through the real functions (each must raise):")
    bad = 0
    test_cell = int(splits.loc[splits["role"] == "spatial_test", "cell_id"].iloc[0])
    calib_cell = int(splits.loc[splits["role"] == "calibration", "cell_id"].iloc[0])
    fold0 = pd.read_parquet(D.OUT / "folds.parquet").query("fold == fold.min()")["cell_id"].tolist()

    cases = [
        ("held-out list containing a test cell",
         lambda: D.fold_rows(d, splits, cells=sorted(set(fold0) | {test_cell}))),
        ("scoring the test year",
         lambda: D.scoring_rows(d, splits, years=[D.TEST_YEAR])),
        ("a calibration cell as the held-out list",
         lambda: D.fold_rows(d, splits, cells=[calib_cell])),
    ]
    for label, fn in cases:
        try:
            fn()
        except (AssertionError, ValueError) as exc:
            say(f"  {label}: raised -> {str(exc)[:90]}")
        else:
            say(f"  {label}: DID NOT RAISE")
            bad += 1

    say("")
    say(f"{'PASS' if failures == 0 and bad == 0 else 'FAIL'}  "
        f"scheme failures {failures}, breakages not caught {bad}")
    return 0 if failures == 0 and bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())