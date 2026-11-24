# Concerns before Phase 2 runs — corrected

Revised after review (`REVIEW_CONCERNS_MODEL_PHASE2.md`). Three concerns rested on
an out-of-date split: calibration was redrawn one cell per region on 19 Nov, and the
approved Phase 1 state is what `splits_summary.md` now shows. Everything below is
recomputed from that file.

The code actions that came out of the review are done: `run_key` and the Optuna
study name now include a hash of the fit code; `summarise.py` reports R1 vs R2 by
AOD availability and leave-one-year-out with and without 2023; `score_test.py`
reports per cell and for the cells more than 10 km from development.

## A. Evaluation design

**A-1. Six spatial-test cells.** One per region. A block bootstrap over six cells
resamples six units, so intervals are wide and unstable; one unusual cell moves
them. **Reporting rule:** pool the six, but also report every cell separately with
its row count, bias and distance to the nearest development cell, and describe the
bootstrap as indicative. The 2024 temporal test (~29 cells) is the better-powered
of the two sets and should be presented that way.

**A-2. The North spatial-test cell is thin.** Cell 856931: 1,095 rows, 2010–2012
only, **one day above 25 ug/m3**. North does now have a calibration cell — 863269,
5,027 rows, 256.7 km from development — so a North interval is possible. The
evidence that carries weight for North is leave-one-region-out (region 6 held out,
4 development cells) and the North group inside the 2024 test, not the single
spatial-test cell.

**A-3. Two test cells sit 5 km from development cells.** The cost of the cell-level
decision, accepted by the author:

| role | cell | region | nearest development cell |
|---|---|---:|---:|
| spatial test | 918734 | 1 | **5.0 km** |
| spatial test | 910214 | 2 | 30.4 km |
| spatial test | 895350 | 3 | 68.0 km |
| spatial test | 958101 | 4 | 62.6 km |
| spatial test | 924048 | 5 | 21.2 km |
| spatial test | 856931 | 6 | **5.0 km** (same monitor as 856930, before it moved in 2013) |
| calibration | 918737 | 1 | 5.0 km |
| calibration | 928317 | 2 | 11.2 km |
| calibration | 864472 | 3 | 5.0 km |
| calibration | 933610 | 4 | 58.3 km |
| calibration | 930445 | 5 | 5.0 km |
| calibration | 863269 | 6 | 256.7 km |

**Reporting rule:** call the spatial test **"held-out monitor cells within the
existing network"**, never "unseen locations", and report it twice — all six cells,
and the four more than 10 km from any development cell (910214, 895350, 958101,
924048). That subset is fixed by distance before the test sets are opened, so it is
not a post-hoc choice.

**A-4. Selection bias and 2023.** Grouped 10-fold is where R1 and R3 were chosen;
the summary labels it. Leave-one-year-out is reported per year, and pooled both with
and without 2023, which is the extreme smoke year and dominates that scheme's
variance (quick-mode per-fold RMSE ranged 2.47–9.58).

## B. Target and features

**B-1. Negative hourly values were treated as missing — but almost none of them
reach this dataset.** The rule removed 11,130 AQS and 107 NAPS station-days; the
models here use **Ontario rows only**, which are NAPS. So the effect on this
dataset is about **107 station-days, roughly 0.05% of Ontario rows**, not the 1%
figure I gave earlier. The upward-bias argument is correspondingly small. Keep the
one-line caveat — negative hours were treated as missing before the 18-hour rule —
and drop the AQS numbers from this concern.

**B-2. AOD is ~73% missing, and the A-vs-C comparison is framed correctly.** R2's
question is what MERRA-2 aerosol adds *beyond* AOD, not AOD against MERRA. The new
`aod_missing` split in `summarise.py` answers it directly: R1 and R2 are compared on
days with a retrieval and on days without, so the reader can see whether MERRA-2
earns its place mainly on the days the satellite has nothing.

## C. Compute and environment

**C-1. The learning rate is now searched, and the cap is the constraint.** The
search draws `learning_rate` log-uniformly from **0.03–0.2** for both boosted models,
so the earlier advice "fix it at 0.1" no longer applies. What matters instead is how
the search interacts with `max_rounds = 5000` and `patience = 100`:

| learning rate | trees needed (approx.) | outcome |
|---|---|---|
| 0.2 | ~1,250 | stops early, fast trial |
| 0.05 (measured) | 4,956 | stopped one round short of the cap |
| 0.03 | ~8,000 | **hits the cap**: under-converged, and `folds_not_stopped` rises |

So roughly the lower third of the range is cap-bound: those trials are the slowest
*and* the least converged, and if one of them wins the 1-SE band, `n_trees_final`
becomes the cap rather than a measured number.

**C-1b. The 1-SE tie-break does not look at tree count.** The plan's rule is "fewest
leaves, then largest minimum leaf size". With the learning rate fixed that was
enough; now a low-rate trial and a high-rate trial can tie on RMSE while differing
by 4x in trees and prediction time. Adding median trees as a final tie-break would
pick the cheaper model, but it is a deviation from the plan's written rule, so it is
the author's decision, not mine.

**C-2. The full Phase 2 is 33–37 hours on 12 cores at the measured 77 s per fit**,
and the search range makes the average trial 1.5–2x cheaper than that worst case —
0.2 converges in about a quarter of the rounds. **On SciNet (4–8 CPUs) it is
slower**, not faster: LightGBM's thread scaling is sublinear. Two levers:

1. **`--workers`** runs trials concurrently, `cpus/workers` threads each. On 8 CPUs,
   `--workers 8` (one thread per fit) is usually better than one eight-threaded fit.
   `models.detect_cpus()` reads `SLURM_CPUS_PER_TASK`, so nothing oversubscribes.
2. **Measure one fit first** — `code/model/SCINET.md` has the command. The 77 s was
   12 threads on this machine; the cluster number replaces it.

`code/model/SCINET.md` has the batch script and a one-fit timing command to run
before committing to the full tuning.

**C-3. Early stopping may still be inert for the slow trials.** `folds_not_stopped`
in `run_meta.json` says so directly, and with the rate searched it is now also a
diagnostic of which trials were cap-bound: a high count usually means a low-rate
trial. If the chosen trial is cap-bound, describe its tree count as a round budget
rather than an early-stopped number.

**C-3. Random Forest stays at `n_jobs = -1`.** Phase 2 will run where joblib works;
no serial compromise for this machine.

## D. Residual code risks

- **The early-stopping set is small**: one cell per region, about six of the 32
  development cells. It is **not** the same six in every fold — the inner seed is
  per fold — but six cells is noisy whatever the draw. Note it in the methods.
- **Quick mode is for timing only.** Its numbers are a 10% sample.
- **`run_key` now covers the code** as well as config, parameters, features and the
  target fingerprint, and the Optuna study name carries the same code hash. Editing
  the fit code now forces `--fresh` instead of silently resuming old folds.
- **The Random Forest grid is cached per point**, so a crash resumes rather than
  restarting 120 fits.

## E. What remains open

| question | status |
|---|---|
| Report the spatial test as "unseen locations"? | No — relabel, and report per cell and for the >10 km subset (A-3). |
| Redraw calibration one per region? | Done; North calibration is 863269. |
| Learning rate 0.05 or 0.1? | **Open — author.** 0.1 is recommended before the full tuning. |
| Leave-one-year-out with 2023 separate? | Done in `summarise.py`. |

Still the author's decision: the learning rate, and where Phase 2 runs.
