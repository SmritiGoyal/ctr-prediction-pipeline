# CTR Prediction Pipeline

> **L2-regularized logistic regression with smoothed CTR encoding, frequency encoding, and feature hashing for click-through rate prediction on the Avazu dataset.** Validation log-loss **0.382** (vs naive baseline 0.4312, ~**11.4% improvement**) on 32M+ training rows, 13M+ test rows.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

## What this is

This repository contains a complete, reproducible implementation of a click-through rate (CTR) prediction model for the **Avazu CTR Prediction Challenge** on Kaggle. The pipeline ingests 32M+ training impressions, applies leakage-safe categorical encodings, projects features into a 2^22-dim hashed space, fits an L2-regularized logistic regression, and produces calibrated click probabilities for the 13M-row test set.

The headline result — **0.382 validation log-loss** — represents an ~11.4% improvement over the naive global-CTR baseline.

## Headline results

| Metric | Value |
|---|---:|
| Validation log-loss | **0.382** |
| Naive baseline log-loss | 0.4312 |
| Improvement | **~11.4%** |
| Test predictions | 13,015,341 |
| Training rows | ~32M |
| Pipeline runtime | ~30-45 min on 16GB laptop |

## How it works

```
                Avazu raw CSV (~32M rows)
                          |
                          v
              src/ingestion.py
              - Chunked sampling for EDA (1M rows)
              - Bernoulli sampling for retrain (10M rows)
                          |
                          v
              src/feature_engineering.py
              - Time decomposition (YYMMDDHH -> 7 features)
              - Rare category bucketing (threshold=50)
              - Categorical interactions (3 pairs)
              - Column drops (5 noisy/redundant columns)
              - Time-based train/validation split (~75/25 by rows)
                          |
                          v
              src/encoding.py
              - Smoothed Bayesian CTR encoding (alpha=50)
              - Frequency encoding with log1p
              - Column-type partition (numeric vs categorical)
              - Feature hashing to 2^22 dimensions
                          |
                          v
              src/modeling.py
              - L2-regularized logistic regression
              - Logit-space mean calibration
                          |
                          v
              outputs/submission.csv
              13M rows of calibrated P(click) predictions
```

## Repository structure

```
ctr-prediction-pipeline/
|-- README.md                  This file
|-- LICENSE                    MIT
|-- requirements.txt
|-- .gitignore
|-- config.example.py          Copy to config.py if you want to override
|
|-- src/
|   |-- ingestion.py           Memory-efficient CSV sampling
|   |-- feature_engineering.py Time features + rare bucketing + interactions + split
|   |-- encoding.py            CTR + frequency + column-type + hashing
|   |-- modeling.py            Logistic regression + calibration
|   |-- diagnostics.py         EDA + validation + sanity checks
|   `-- run_pipeline.py        End-to-end orchestrator
|
|-- data/
|   `-- README.md              How to obtain Avazu data from Kaggle
|
|-- outputs/
|   `-- README.md              Output schema documentation
|
`-- docs/
    |-- features.md            Per-feature documentation
    `-- methodology.md         Full technical writeup
```

## Quick start

### Prerequisites

- Python 3.10+
- 16 GB RAM recommended (8 GB works with reduced settings — see `data/README.md`)
- Kaggle account for the Avazu dataset

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/SmritiGoyal/ctr-prediction-pipeline.git
cd ctr-prediction-pipeline

# 2. Set up a virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.\.venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Create local config to override defaults
cp config.example.py config.py
# Edit config.py if needed

# 5. Download Avazu data per data/README.md, place in data/

# 6. Run the pipeline
python src/run_pipeline.py
```

Expected runtime: 30-45 minutes on a modern laptop. The retrain stage (10M-row sample, hashed and fit) is the dominant cost at ~15-20 minutes.

## Pipeline output

`src/run_pipeline.py` produces `outputs/submission.csv` with two columns:

| Column | Description |
|---|---|
| `id` | Avazu impression identifier (from the submission template) |
| `P(click)` | Calibrated click probability, formatted to 10 decimal places |

Plus extensive logging to stdout: EDA diagnostics, validation log-loss, prediction quantiles, and sanity checks. Capture the log with:

```bash
python src/run_pipeline.py 2>&1 | tee outputs/run_log.txt
```

## Reproducibility

All randomness flows from `config.PipelineConfig.random_state = 42`. Given the same Avazu CSV files, the pipeline produces deterministic outputs (the same submission CSV, byte-for-byte).

## Key design decisions

A summary; see `docs/methodology.md` for the full rationale.

1. **Time-based train/validation split.** The latest 20% of unique dates are held out (7 train dates, 2 val dates on Avazu's 9-date span, ~75/25 by row count). A random split would leak future user behavior backward into training.

2. **Smoothed Bayesian CTR encoding.** Categorical columns are replaced with `(clicks + 50 * global_ctr) / (count + 50)`. Smoothing pulls low-count categories toward the global mean, preventing overfitting to rare values.

3. **Feature hashing at 2^22 dimensions.** After categorical interactions, the unique-value count is ~8-10M. Hashing into a 4.2M-dim sparse space keeps memory bounded while preserving signal.

4. **L2-regularized logistic regression.** LR natively optimizes log-loss, scales linearly with feature dimensionality, and handles sparse input efficiently. With our explicit interaction features, it captures most of the signal a tree model would find.

5. **Logit-space mean calibration.** Raw LR predictions are systematically overconfident. A single-offset shift in logit space pulls the mean to a target (training CTR for test, validation CTR for validation) at zero retraining cost.

## Limitations

In the spirit of honest documentation, this implementation does not address:

- **Hyperparameter tuning** — `alpha`, `C`, `hash_dim`, etc. chosen by sensitivity analysis on the EDA sample, not formal grid search
- **More expressive models** — XGBoost or DeepFM would likely improve log-loss by 0.005-0.015
- **Additional feature interactions** — only 3 pairs explicitly encoded; greedy search would find more
- **Production concerns** — no inference API, A/B framework, drift detection, or feature store

See `docs/methodology.md` Section 5 for the full discussion.

## License

MIT — see [LICENSE](LICENSE).

The Avazu dataset is **not redistributed** with this repository — see Kaggle's competition rules. The `.gitignore` prevents accidental commits of data files.

## Citation

If you reference this work:

```
Goyal, S. (2026). CTR Prediction Pipeline.
GitHub repository: https://github.com/SmritiGoyal/ctr-prediction-pipeline
```
