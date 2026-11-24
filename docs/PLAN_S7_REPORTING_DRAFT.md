# Draft for `MODEL_RUN_PLAN.md` §7 — reporting rules, fixed before the test sets open

Paste into the plan. Written while the test sets are still sealed (`OPEN_TEST` does
not exist and no script has scored them), so the choices are pre-registered rather
than chosen after seeing results.

---

## 7. Reporting rules

Fixed before any test row is scored. Any deviation is a finding to be reported, not
a choice to be made quietly.

**7.1 The spatial test is "held-out monitor cells within the existing network".**
Not "unseen locations". Two of the six cells (918734, 856931) are 5 km from a
development cell, and 856931 is the same monitor as 856930 before it moved in 2013.

**7.2 The spatial test is reported three ways, always together.**

| presentation | cells | purpose |
|---|---|---|
| pooled | 6 | headline number |
| per cell | 6 | RMSE, bias, n, mean PM2.5, days > 25, distance to the nearest development cell |
| more than 10 km from development | 4 (910214, 895350, 958101, 924048) | closest available proxy for an unseen location |

The subset is fixed by distance alone, before scoring, so it cannot be a post-hoc
selection. The per-cell table is not optional: with six cells the pooled number
hides the spread.

**7.3 Bootstrap intervals on the spatial test are indicative, not confirmatory.**
A cell-block bootstrap over six cells resamples six units. Say so in the text, and
present the 2024 temporal test (about 29 cells) as the better-powered of the two
sets.

**7.4 North.** The North spatial-test number is reported with its `n` and with the
caveat that the cell covers 2010-2012 and one day above 25 ug/m3. The North
evidence that carries weight is leave-one-region-out with region 6 held out, and the
North group inside the 2024 test.

**7.5 Selection bias is labelled.** Grouped 10-fold is the scheme the parameters were
chosen on. Its block in the results is titled "tuning folds (selection-biased)".
Leave-one-region-out and leave-one-year-out are the development estimates; the test
sets have the final word.

**7.6 Leave-one-year-out is reported per year, and pooled twice** — with and without
2023 — with 2023 named as the extreme smoke year that dominates that scheme's
variance.

**7.7 R1 versus R2 is also reported by AOD availability** (`aod_missing` 0/1), so the
question "what does MERRA-2 aerosol add beyond AOD" is answered on days with a
retrieval and on days without.

**7.8 The test sets are scored once.** `score_test.py` is run once per chosen model
after `OPEN_TEST` is created. Any change to the model, the features or the split
afterwards is recorded in the decision note together with the reason, and the test
results are reported as superseded rather than overwritten.

---
