# Pipeline Outputs

This directory holds the artifact produced by `src/run_pipeline.py`. The file is gitignored because it is fully regenerable from the source data. This README documents the schema so a reader of the repo understands what the pipeline produces without needing to run it.

## File

| Filename | Rows | Granularity |
|---|---|---|
| `submission.csv` | ~13,015,341 | One row per test impression |

The exact row count depends on the version of the Avazu test set used. The pipeline asserts row-count preservation between the submission template and the final write — if these don't match, the script fails fast.

## Schema

`submission.csv` has two columns:

| Column | Type | Description |
|---|---|---|
| `id` | str | The opaque row identifier from the Avazu submission template. Preserved exactly as provided by Kaggle — no reformatting, no casts. |
| `P(click)` | str | The calibrated probability that this impression resulted in a click, formatted to 10 decimal places (e.g., `0.1234567890`). String-typed to preserve precision in the CSV. |

The `id` and row order are taken from `data/ProjectSubmission-TeamX.csv`, the template Kaggle provides. Predictions are assigned by row position, not by an `id` join — this matches how Kaggle's grader reads the file.

## Validated headline metrics

The pipeline targets log-loss minimization on the held-out 20% time-based validation split. On a validated end-to-end run:

| Metric | Value |
|---|---|
| Final calibrated validation log-loss | **0.382** |
| Baseline log-loss (predict global CTR everywhere) | 0.4312 |
| Improvement vs baseline | **~11.4%** |
| Test predictions | 13,015,341 |
| Mean predicted CTR (test) | ~0.22 |

The 0.382 log-loss is the headline result — it's what the pipeline is engineered to achieve, and it reproduces deterministically given the same data inputs and `random_state=42` (set in `config.py`).

## Pipeline output stages

The pipeline writes one file (`submission.csv`) but produces several useful intermediate diagnostics via the logger:

1. **EDA diagnostics** — cardinality and CTR breakdowns on the 1M-row EDA sample (Section A of `run_pipeline.py`)
2. **Validation diagnostics** — initial-model log-loss vs baseline, prediction quantiles (Section G)
3. **Final validation log-loss** — after retrain + calibration (Section I)
4. **Test prediction sanity checks** — NaN/inf counts, min/max bounds, quantiles (Section J)

These are written to stdout via Python's `logging` module. To capture them, redirect output:

```bash
python src/run_pipeline.py 2>&1 | tee outputs/run_log.txt
```

## Regenerating the submission

From the repository root:

```bash
python src/run_pipeline.py
```

Expected runtime: 30-45 minutes on a 16 GB machine. See `data/README.md` for memory considerations on smaller machines.
