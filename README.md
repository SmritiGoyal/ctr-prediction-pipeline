# Click-Through Rate Prediction Pipeline

![Log Loss](https://img.shields.io/badge/Validation%20Log%20Loss-0.382-2ea44f?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-32M%20rows-blue?style=for-the-badge)
![Stack](https://img.shields.io/badge/Stack-Python%20%7C%20scikit--learn%20%7C%20scipy-orange?style=for-the-badge)

End-to-end machine learning pipeline for predicting click probabilities on the Avazu dataset (32M training rows, 13M test predictions). The pipeline implements memory-safe feature hashing, leakage-safe encoding, time-based validation, and post-hoc probability calibration — design choices that mirror production ad-tech systems while running on a single machine.

---

## The Problem

Click-Through Rate prediction sits at the core of digital advertising. Every served impression is scored in milliseconds and the resulting probability feeds auction pricing, ad ranking, and revenue optimization. Three properties make it harder than it first appears:

- **Severe class imbalance.** Clicks are rare events — the global CTR on the Avazu training data is ~17%. A model that predicts "no click" for every row achieves 83% accuracy but is useless for ranking. Calibrated probabilities matter far more than hard labels, which is why **log loss** is the right metric.
- **High-cardinality categoricals.** Identifier features like `device_id`, `device_ip`, and `site_id` carry hundreds of thousands of unique values. One-hot encoding is computationally infeasible; rare-value handling and hashing become unavoidable.
- **Temporal dynamics.** User behavior drifts over time. A random train/validation split leaks future patterns backward into training and produces optimistic offline scores that do not survive production deployment.

This project addresses all three within a reproducible single-machine pipeline.

---

## Results

The pipeline is benchmarked against the naive global-CTR baseline and an uncalibrated logistic regression variant. The final result represents an **11.4% improvement in log loss over baseline** with no GPU and full dataset coverage.

| Model | Validation log loss | Δ vs baseline |
|---|---:|---:|
| Global CTR baseline (constant 0.175) | 0.431 | — |
| Logistic regression, no calibration | 0.385 | −10.6% |
| **Logistic regression + logit calibration** | **0.382** | **−11.4%** |

The calibration step alone — a closed-form logit shift with zero retraining — accounts for a meaningful slice of the improvement.

### Test predictions (13,015,341 rows)

| Statistic | Value |
|---|---:|
| Mean predicted CTR | 0.218 |
| Median predicted CTR | 0.210 |
| Min / max predicted CTR | 0.000377 / 0.987 |
| NaN / Inf count | 0 / 0 |
| Quantiles (1%, 50%, 99%) | 0.015, 0.210, 0.631 |

The distribution is smooth, unimodal, and tightly bracketed — no out-of-range predictions or degenerate clusters.

---

## Pipeline at a Glance

```
                 ┌─────────────────────────────────┐
                 │   Chunked EDA sampling (1M)     │  Section A
                 │   diagnostics + cardinality     │
                 └────────────────┬────────────────┘
                                  ▼
                 ┌─────────────────────────────────┐
                 │   Full train + test load        │  Sections B–C
                 │   Time features, rare bucketing │
                 │   Categorical interactions      │
                 └────────────────┬────────────────┘
                                  ▼
                 ┌─────────────────────────────────┐
                 │   Time-based 80/20 split         │  Section D
                 │   (chronological, no shuffle)    │
                 └────────────────┬────────────────┘
                                  ▼
                 ┌─────────────────────────────────┐
                 │   Leakage-safe encoders          │  Section E
                 │   • Smoothed CTR (α = 50)        │
                 │   • Frequency encoding (log1p)   │
                 └────────────────┬────────────────┘
                                  ▼
                 ┌─────────────────────────────────┐
                 │   Feature hashing                │  Section F
                 │   2²² ≈ 4.19M sparse dims        │
                 │   Batched (500k rows/block)      │
                 └────────────────┬────────────────┘
                                  ▼
                 ┌─────────────────────────────────┐
                 │   Logistic regression, L2        │  Sections G–H
                 │   Retrain on 10M-row sample      │
                 └────────────────┬────────────────┘
                                  ▼
                 ┌─────────────────────────────────┐
                 │   Logit-shift calibration        │  Section I
                 │   Validation log loss 0.382      │
                 └────────────────┬────────────────┘
                                  ▼
                 ┌─────────────────────────────────┐
                 │   Test predictions + sanity      │  Sections J–K
                 │   13,015,341 rows → submission   │
                 └─────────────────────────────────┘
```

---

## Key Technical Decisions

The choices below are the ones that drove the result. Each emerged from explicit EDA findings or measured failures of an alternative, not from defaults.

### 1. Time-based split, not random shuffle

The Avazu data spans 10 days. A random shuffle would let the model train on `Day 10 hour 14` and evaluate on `Day 7 hour 09` — that is, it would let future user behavior leak backward into training and produce optimistic scores. The pipeline sorts on the raw `YYMMDDHH` timestamp and reserves the most recent ~20% of unique dates for validation. The training-set CTR (0.175) is measurably higher than the validation-set CTR (0.154), confirming the distribution shift the split is designed to expose.

### 2. Frequency and smoothed-CTR encoding learned **only on train**

This was the single largest source of subtle leakage in the early iterations. Frequency and CTR statistics computed on the full dataset look harmless — they're per-category aggregates, not row-level targets — but they let the model see the validation set's category distribution at training time, inflating offline scores. The fix:

```python
# Bayesian-smoothed CTR, computed on the training split only
smoothed_ctr(category) = (clicks + α·global_ctr_train) / (count + α)
```

with `α = 50` virtual observations of the training-set mean. Unseen categories in the validation or test set fall back to `global_ctr_train`. The same pattern applies to frequency encoding (log1p-transformed). All maps are persisted in a `LearnedEncoders` dataclass and re-applied verbatim downstream — never re-fit.

### 3. Feature hashing at 2²² dimensions instead of one-hot

The combined cardinality of `site_id`, `site_domain`, `app_id`, `device_*`, and engineered interactions (`app_site`, `appdom_sitedom`, `C14_C17`) explodes beyond what dense one-hot encoding can hold in memory. Feature hashing trades a small risk of hash collisions for a fixed footprint of 2²² ≈ 4.19 million sparse columns. At that dimensionality, collision-induced log loss inflation is negligible (verified empirically against smaller hash dimensions).

### 4. Batched hashing for memory safety on Section H

The initial single-shot `FeatureHasher.transform()` call hit a 2 GB internal allocation when hashing the 10M-row retrain sample on a 16 GB machine. The pipeline now hashes in 500k-row blocks and `vstack`s the resulting CSR matrices — bit-identical output, ~100 MB peak per block, ~5 minutes total for 10M rows.

### 5. Post-hoc logit calibration

The model produces systematically over-confident probabilities (mean predicted CTR ~0.19, true CTR ~0.15). Rather than retraining with class weights or sample reweighting, the pipeline applies a closed-form shift in log-odds space:

```
offset      = logit(target_mean) − mean(logit(predicted_probs))
calibrated  = sigmoid(logit(predicted_probs) + offset)
```

Test-set calibration is anchored on the **training** CTR — never on validation or test labels — to keep the pipeline strictly leakage-safe. This single transformation reduces log loss from 0.385 to 0.382 at zero additional training cost.

### 6. Logistic regression beat the alternatives

The team evaluated decision trees, random forests, and an MLP (5-5-1, ReLU, dropout, L2) against L2-regularized logistic regression on the hashed feature space. Logistic regression won on every metric: lowest log loss, fastest training, lowest memory. The MLP underperformed largely because high-cardinality IDs were passed as ordinal integers — a representation neural networks misread as having distance and magnitude. The right neural fix would be learned embeddings per categorical column, which would have multiplied development cost without a clear log-loss advantage on this dataset.

---

## Repository Structure

\`\`\`
ctr-prediction-pipeline/
├── README.md                  # This file
├── LICENSE                    # MIT
├── requirements.txt           # Pinned dependencies (numpy, pandas, scikit-learn, scipy)
├── .gitignore                 # Excludes data/, outputs/, venvs, local config.py
├── config.example.py          # PipelineConfig template — copy to config.py to override defaults
│
├── src/
│   ├── ingestion.py           # Memory-efficient CSV sampling
│   ├── feature_engineering.py # Time features + rare bucketing + interactions + time-based split
│   ├── encoding.py            # CTR + frequency + column-type partition + feature hashing
│   ├── modeling.py            # Logistic regression + logit-shift calibration
│   ├── diagnostics.py         # EDA + validation + sanity checks
│   └── run_pipeline.py        # End-to-end orchestrator
│
├── data/
│   └── README.md              # Avazu data access instructions
│
├── outputs/
│   └── README.md              # Submission CSV schema documentation
│
└── docs/
    ├── features.md            # Per-feature documentation with formulas + rationale
    └── methodology.md         # Full technical writeup
\`\`\`

The pipeline is organized into six focused modules under `src/`, each owning a single concern. The orchestrator (`src/run_pipeline.py`) composes them into the 11-section flow shown in the pipeline diagram above. The original monolithic implementation was refactored into this structure while preserving byte-identical numerical output — the validated reproduction below confirms the refactor is faithful.

---

## Reproducing the Results

### Prerequisites

- Python 3.10+
- 16 GB RAM minimum (the 30M-row CSV load peaks around 8–10 GB)
- ~10 GB free disk for the dataset

### Setup

```bash
git clone https://github.com/SmritiGoyal/ctr-prediction-pipeline.git
cd ctr-prediction-pipeline
python -m venv .venv
.venv/Scripts/activate           # Windows
# source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

### Get the data

The Avazu dataset is hosted on Kaggle and is not redistributed in this repo. Instructions for downloading and placing the files are in [`data/README.md`](data/README.md).

### Run

\`\`\`bash
cp config.example.py config.py    # Or: copy config.example.py config.py  (Windows)
python src/run_pipeline.py
\`\`\`

The pipeline produces structured logs at each section and writes the final submission to `outputs/submission.csv`. On a typical laptop (16 GB RAM, no GPU) end-to-end runtime is ~27 minutes.

### Runtime breakdown

| Section | Wall time |
|---|---:|
| A — EDA sampling | 1m 35s |
| B — Full data load | 2m 19s |
| C — Time features + rare bucketing | 4m 00s |
| D — Time-based split | 50s |
| E — Encoder learning | 3m 01s |
| F — Feature hashing (train + val) | 4m 23s |
| G — Initial LR fit | 2m 50s |
| H — Retrain on 10M-row sample | 5m 03s |
| I — Final validation + calibration | 2s |
| J — Test prediction + hashing | 1m 56s |
| K — Submission writer | 1m 09s |
| **Total** | **~27 min** |

---

## What I'd Do Differently

Treating this as version 1, the obvious improvements for a v2:

- **Replace logistic regression with FTRL-Proximal.** Logistic regression's `lbfgs` solver requires the full hashed matrix in memory. FTRL is the de-facto production CTR algorithm (the same family that powered Google's Smart Bid) because it learns online in a single streaming pass, supports L1+L2 regularization for free, and never materializes the full design matrix. Expected gains: 2-3× speedup, similar log loss, no retrain step needed.
- **Field-aware factorization machines (FFM).** The MLP failed because dense layers can't represent high-cardinality categoricals natively. FFM explicitly models pairwise interactions between fields with learned embeddings, and consistently wins on CTR benchmarks. This is what would belong in a "phase 2" model card.
- **Proper calibration evaluation.** The pipeline reports log loss and mean-prediction-vs-truth, but a reliability diagram with quantile bins and an Expected Calibration Error (ECE) number would be more honest reporting. The logit-shift calibration corrects the mean but doesn't necessarily fix bin-level miscalibration.
- **Day-of-week extraction is approximate.** The current `is_weekend` flag uses `(day_of_month % 7) >= 5`, which is a proxy that happens to correlate with actual weekends over Avazu's 10-day window but is not mathematically day-of-week. A proper `datetime` parse would fix it. Left in place here to preserve reproducibility of the original result.
- **Hash dimension sensitivity sweep.** 2²² was chosen on heuristic grounds (large enough to make collisions rare). A formal sweep over `{2¹⁸, 2²⁰, 2²², 2²⁴}` measuring log loss vs. memory would justify the choice quantitatively.

---

### Refactor history

This repository was originally a single 1,100-line `ctr_pipeline.py`. In May 2026 it was refactored into the modular `src/` structure above. The refactor preserves byte-identical numerical output — validated by re-running the full pipeline and confirming the validation log loss reproduces to 0.381919 (matches the original 0.382 within rounding).

---

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — `FeatureHasher`, `LogisticRegression`, `log_loss`
- **pandas / numpy** — data manipulation, vectorized features
- **scipy** — sparse matrix operations, `expit` / `logit` for calibration
- Standard library: `pathlib`, `logging`, `dataclasses`

No deep learning frameworks, no GPU, no cloud infrastructure — by design.

---

## Context

This project was completed as the capstone for an applied machine learning course in the Emory MSBA program. This repository is the cleaned, refactored, and documented version of the team's final submission; the modeling decisions and results are unchanged.
