# Daily PM2.5 for Ontario, 2010–2024

Gradient-boosted models of daily PM2.5 on a 5 km grid, from MAIAC AOD, MERRA-2, NOAA HMS
smoke and FIRMS fire data, trained on NAPS monitors. Every source is on the UTC-6 day.

## Where to start

| What | Where |
| --- | --- |
| Manuscript draft | [docs/MANUSCRIPT_DRAFT.md](docs/MANUSCRIPT_DRAFT.md) |
| Phase 2 results (development cross-validation) | [docs/PHASE2_RESULTS.md](docs/PHASE2_RESULTS.md) |
| Analysis plan | [docs/MODEL_RUN_PLAN.md](docs/MODEL_RUN_PLAN.md) |
| Data dictionary, day definition, quality checks | [data/processed/README.md](data/processed/README.md) |
| Independent data audit | [audit/04_findings.md](audit/04_findings.md) |

## Layout

- `code/data_acquisition/`, `code/colab/` — downloads and Earth Engine exports
- `code/data_preparation/`, `code/panel/` — the data build, including the UTC-6 conversion of each source, and its `verify_*.py` checks
- `code/model/` — splits, tuning, cross-validation, summaries and plots; `code/model/SCINET.md` for running on Trillium
- `data/processed/target/target_table.parquet` — the training table (Git LFS, 156 MB); the models use the Ontario rows (`on_region > 0`)
- `data/interim/model/` — splits, folds, tuned parameters, out-of-fold predictions (`full/<run>/<scheme>/`), results and figures
- `logs/trillium/` — Slurm logs of the Phase 2 run (2026-11-22)

Raw downloads, the intermediate panels and the 2010–2024 prediction grid are not in the
repository; they are on SciNet project space (`/project/def-kuan/dongsha3/ai-workspace`).

The spatial and 2024 test sets are in the training table but have not been scored; they
stay sealed until the Phase 2 settings are final.

## Clone

```bash
git lfs install
git clone <repository URL>
```
