#!/usr/bin/env python3
"""Feature reduction report and draft list (plan step 2).

Rule
----
A correlation between two columns is only meaningful where both are observed.
`DataFrame.corr` computes each pair on its pairwise-complete rows, which for a
NaN-heavy pair can be a few percent of the data and can measure the wrong thing
entirely: `aod_last_obs` equals `aod055` on every observed AOD day by
construction, yet it exists to cover the ~70% of days `aod055` is missing.

So a pair is only considered for grouping when its partial overlap is small: if
one column is present where the other is missing on more than
`max_partial_overlap` of rows, the two carry different information and are not
merged, however high their co-observed correlation.

This writes a report and a DRAFT. It never freezes the list, and it does not
choose representatives -- it ranks candidates and says so.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import data as D
import features as F

OUT = D.OUT


def constant_columns(train: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if train[c].nunique(dropna=False) <= 1]


def overlap_stats(train: pd.DataFrame, a: str, b: str) -> dict:
    pa = train[a].notna().to_numpy()
    pb = train[b].notna().to_numpy()
    n = len(train)
    return {
        "both": float((pa & pb).mean()),
        "a_only": float((pa & ~pb).mean()),
        "b_only": float((~pa & pb).mean()),
        "penalty": float(max((pa & ~pb).mean(), (~pa & pb).mean())),
        "n": n,
    }


EPS = 1e-12


def correlation_groups(train: pd.DataFrame, cols: list[str], thr: float, method: str,
                       max_partial: float,
                       nested_families=()) -> tuple[list[tuple[str, ...]], list[dict]]:
    """Greedy groups, NaN-safe, with a dominance rule before the overlap penalty.

    A pair is merged when

    * |rho| >= thr, tested so that a NaN correlation is never treated as passing
      (`NaN < thr` is False); and
    * one column is present wherever the other is (a nested pair, such as a 25 km
      neighbourhood mean inside a 50 km one), or the asymmetric overlap is small.

    Nesting is **declared**, not inferred: `nested_families` lists pairs where the
    wider member contains the narrower one (a 25 km mean inside a 50 km mean).
    Statistically, a gap-filled copy (`aod055` / `aod_last_obs`) looks identical to
    a nested pair -- same correlation, one column present wherever the other is --
    but it must not be merged, because the copy exists to cover the gaps.
    """
    corr = train[cols].astype("float64").corr(method=method).abs()
    nested = {frozenset(pair) for pair in nested_families}
    unused = set(cols)
    groups: list[tuple[str, ...]] = []
    rejected: dict[frozenset, dict] = {}

    for c in cols:
        if c not in unused:
            continue
        members = [c]
        for d in cols:
            if d == c or d not in unused:
                continue
            rho = corr.loc[c, d]
            if not (rho >= thr):
                continue
            ov = overlap_stats(train, c, d)
            if frozenset((c, d)) in nested:
                members.append(d)                     # declared nested pair
            elif ov["penalty"] <= max_partial:
                members.append(d)
            else:
                rejected.setdefault(frozenset((c, d)),
                                    {"a": c, "b": d, "corr": float(rho), **ov})
        if len(members) > 1:
            groups.append(tuple(members))
            unused -= set(members)
    return groups, list(rejected.values())


def rank_group(group: tuple[str, ...], ranked: pd.DataFrame,
               nested_families=()) -> list[str]:
    """Order a group for the suggestion column.

    A declared nested family puts the wider member first, because the narrower one
    is contained in it. Every other group is ranked by gain, then completeness --
    coverage alone would just reproduce list order for complete columns.
    """
    for family in nested_families:
        if set(family) == set(group) and len(family) >= 2:
            wider = family[-1]
            rest = [c for c in group if c != wider]
            return [wider] + sorted(
                rest, key=lambda c: (-ranked.loc[c, "gain_mean"], -ranked.loc[c, "completeness"]))
    return sorted(group, key=lambda c: (-ranked.loc[c, "gain_mean"], -ranked.loc[c, "completeness"]))


def check_nesting(train: pd.DataFrame, nested_families=()) -> None:
    """Assert every declared family really is nested on the development rows.

    The narrower column must never be present where the wider one is missing.
    Declaring a family that is not nested would silently drop coverage.
    """
    for family in nested_families:
        if len(family) < 2:
            continue
        narrow, wide = family[0], family[-1]
        stray = int((train[narrow].notna() & train[wide].isna()).sum())
        if stray:
            raise AssertionError(
                f"declared nested pair {narrow} / {wide}: {stray} rows where "
                f"{narrow} is present and {wide} is missing")


def gain_importance(train: pd.DataFrame, cols: list[str], cfg: dict) -> pd.Series:
    import lightgbm as lgb
    params = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 31, "verbose": -1,
              "num_threads": int(cfg["lightgbm"]["n_jobs"]), "seed": int(cfg["seeds"]["tune"])}
    booster = lgb.train(params, lgb.Dataset(train[cols], label=train["pm25"]),
                        num_boost_round=200)
    return pd.Series(booster.feature_importance("gain"), index=cols)


def main() -> int:
    ap = argparse.ArgumentParser(description="Feature reduction report")
    ap.add_argument("--dataset", default="A", choices=sorted(F.DATASETS))
    args = ap.parse_args()

    cfg = D.load_config()
    d = D.load()
    splits = pd.read_parquet(OUT / "splits.parquet")
    folds = pd.read_parquet(OUT / "folds.parquet")

    train = D.training_rows(d, splits)
    cols = F.columns(args.dataset)
    dropped = constant_columns(train, cols)
    cols = [c for c in cols if c not in dropped]

    families = cfg["feature_reduction"]["nested_families"]
    check_nesting(train, families)
    print(f"declared nesting verified for {len(families)} families")

    thr = float(cfg["feature_reduction"]["corr_threshold"])
    method = cfg["feature_reduction"]["corr_method"]
    max_partial = float(cfg["feature_reduction"]["max_partial_overlap"])

    appearances: dict[tuple, int] = {}
    rejected_all: dict[tuple, dict] = {}
    gains = []
    for f, g in folds.groupby("fold"):
        ftr = D.training_rows(d, splits, drop_cells=g["cell_id"].tolist())
        grps, rej = correlation_groups(ftr, cols, thr, method, max_partial,
                                       families)
        for grp in grps:
            appearances[grp] = appearances.get(grp, 0) + 1
        for r in rej:
            key = (r["a"], r["b"])
            rejected_all.setdefault(key, r)
        gains.append(gain_importance(ftr, cols, cfg))
        print(f"  fold {int(f)}: {len(ftr):,} rows, {len(grps)} groups, {len(rej)} pairs rejected")

    gain = pd.concat(gains, axis=1)
    summary = pd.DataFrame({
        "column": cols,
        "completeness": [float(train[c].notna().mean()) for c in cols],
        "gain_mean": gain.mean(axis=1).reindex(cols).to_numpy(),
        "gain_sd": gain.std(axis=1).reindex(cols).to_numpy(),
    }).sort_values("gain_mean", ascending=False)

    ranked = summary.set_index("column")
    families = cfg["feature_reduction"]["nested_families"]
    suggestions = {}
    for grp in appearances:
        suggestions[", ".join(grp)] = rank_group(grp, ranked, families)

    monthly = {c: {int(m): float(v) for m, v in train.groupby(train["date"].dt.month)[c]
                   .apply(lambda s: s.notna().mean()).items()} for c in cols}

    lines = ["# Feature reduction", "",
             f"dataset {args.dataset}: {len(cols)} candidate columns, {len(train):,} development rows",
             "",
             f"constant columns dropped: {', '.join(f'`{c}`' for c in dropped) or 'none'}", "",
             "## Correlation groups", "",
             f"|Spearman| >= {thr}, computed inside each training fold, but a pair is only "
             f"merged when neither column is present where the other is missing on more than "
             f"{max_partial:.0%} of rows. Groups are listed with the folds they appear in.", "",
             "Pairs declared in `feature_reduction.nested_families` are merged whatever "
             "their overlap, because the wider member carries the narrower one. Nesting is "
             "declared, not inferred: a gap-filled copy looks statistically identical to a "
             "nested pair but must stay separate. Each declaration is verified before use.", "",
             "| folds | columns |", "|---:|---|"]
    for grp, n in sorted(appearances.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {n}/10 | {', '.join(f'`{c}`' for c in grp)} |")

    if rejected_all:
        lines += ["", "## Correlated pairs NOT merged (poor co-observation)", "",
                  "| pair | corr | both present | A only | B only |", "|---|---:|---:|---:|---:|"]
        for (a, b), r in sorted((tuple(sorted(k)), v) for k, v in rejected_all.items()):
            lines.append(f"| `{a}` / `{b}` | {r['corr']:.2f} | {r['both']:.3f} | "
                         f"{r['a_only']:.3f} | {r['b_only']:.3f} |")

    lines += ["", "## Per-column completeness and gain importance", "",
              "Gain is split among correlated columns: when two columns carry the same "
              "information, the tree picks one and the other looks unimportant. A low gain "
              "on a member of a correlation group is not evidence that the variable is "
              "useless.", "",
              "| column | completeness | gain mean | gain sd |", "|---|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| `{row.column}` | {row.completeness:.3f} | "
                     f"{row.gain_mean:,.1f} | {row.gain_sd:,.1f} |")
    lines += ["", "Monthly completeness is in `feature_reduction_monthly.json`.", "",
              "## Ranking within correlation groups (suggestion only)", ""]
    for grp, order in suggestions.items():
        lines.append(f"- {grp}: {', '.join(f'`{c}`' for c in order)}")

    (OUT / "feature_reduction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    draft = {
        "dataset": args.dataset,
        "all_columns": cols,
        "dropped_constant": dropped,
        "correlation_groups": [list(g) for g in appearances],
        "ranked_within_group": suggestions,
        "representatives": {},
        "note": ("representatives left blank on purpose: rank by meaning, completeness and "
                 "gain, then save as features_frozen.json"),
    }
    (OUT / "features_draft.json").write_text(json.dumps(draft, indent=2), encoding="utf-8")
    (OUT / "feature_reduction_monthly.json").write_text(json.dumps(monthly, indent=2),
                                                        encoding="utf-8")
    print(f"{len(cols)} columns, {len(dropped)} constant, {len(appearances)} groups, "
          f"{len(rejected_all)} correlated pairs rejected for overlap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
