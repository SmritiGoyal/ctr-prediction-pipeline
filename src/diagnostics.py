"""
diagnostics.py
==============
Logging and sanity checks for the CTR pipeline.

This module collects every non-modeling, non-data-transformation function
in one place. None of these functions affect the modeling outputs — they
only log statistics or fail-fast on invariant violations. Separating them
into their own module keeps the modeling and feature engineering modules
free of logging clutter.

Three concerns:

    1. EDA diagnostics       — drive the downstream feature engineering
                                decisions (which columns to drop, bucket,
                                or frequency-encode) by surfacing
                                cardinality and CTR statistics on the
                                EDA sample
    2. Validation diagnostics — surface model performance against the
                                naive baseline, plus prediction
                                distribution health checks
    3. Test sanity checks    — fail-fast assertions before writing the
                                submission CSV. Production submissions
                                must contain exactly one finite
                                probability per row, strictly in (0, 1)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

logger = logging.getLogger(__name__)


# =====================================================================
# SECTION 1: EDA DIAGNOSTICS
# =====================================================================

def log_eda_diagnostics(eda_df: pd.DataFrame) -> None:
    """Log cardinality and CTR diagnostics for the EDA sample.

    These diagnostics drive the downstream feature engineering decisions:
    which columns to drop, which to bucket, and which to frequency-encode.
    """
    global_ctr = eda_df["click"].mean()
    logger.info("Global CTR (EDA sample): %.6f", global_ctr)

    cardinality = (
        eda_df.drop(columns=["click", "hour"], errors="ignore")
        .nunique(dropna=False)
        .sort_values(ascending=False)
    )
    logger.info("Top 15 columns by cardinality:\n%s", cardinality.head(15))
    logger.info("Bottom 15 columns by cardinality:\n%s", cardinality.tail(15))

    high_card_cols = ["site_id", "site_domain", "app_id", "device_id", "device_ip"]
    diag = pd.DataFrame(
        [_high_cardinality_diagnostic(eda_df, c) for c in high_card_cols if c in eda_df.columns]
    )
    logger.info("High-cardinality diagnostics:\n%s", diag)


def _high_cardinality_diagnostic(df: pd.DataFrame, col: str) -> dict[str, float | str]:
    """Compute summary statistics describing a high-cardinality column."""
    vc = df[col].value_counts()
    return {
        "feature": col,
        "n_unique": vc.size,
        "singleton_rate": (vc == 1).mean(),
        "top10_share": vc.head(10).sum() / len(df),
    }


# =====================================================================
# SECTION 2: VALIDATION DIAGNOSTICS
# =====================================================================

def log_validation_diagnostics(
    y_true: pd.Series,
    preds: np.ndarray,
    baseline_ctr: float,
) -> None:
    """Log validation log loss, baseline log loss, and prediction distribution."""
    baseline = np.full_like(y_true, fill_value=baseline_ctr, dtype=float)
    logger.info("Baseline log-loss (global CTR): %.6f", log_loss(y_true, baseline))
    logger.info("Model log-loss:                %.6f", log_loss(y_true, preds))
    logger.info("Predicted min/max: %.6f / %.6f", preds.min(), preds.max())
    logger.info("Mean predicted CTR: %.6f   True CTR: %.6f", preds.mean(), y_true.mean())
    logger.info(
        "Predicted quantiles: %s",
        np.quantile(preds, [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]),
    )
    assert np.isfinite(preds).all(), "Found NaN or inf in predictions"


# =====================================================================
# SECTION 3: TEST SANITY CHECKS
# =====================================================================

def run_test_sanity_checks(test_preds: np.ndarray) -> None:
    """Validate the test prediction array before writing the submission.

    Production submissions must contain exactly one finite probability
    per row, strictly within (0, 1). These assertions fail fast if any
    invariant is violated, surfacing pipeline bugs before submission.

    Raises:
        AssertionError: If any sanity check fails.
    """
    n_nan = int(np.isnan(test_preds).sum())
    n_inf = int(np.isinf(test_preds).sum())

    logger.info("Test predictions: %s", f"{len(test_preds):,}")
    logger.info("  Min:    %.6f", float(np.min(test_preds)))
    logger.info("  Max:    %.6f", float(np.max(test_preds)))
    logger.info("  Mean:   %.6f", float(np.mean(test_preds)))
    logger.info("  Median: %.6f", float(np.median(test_preds)))
    logger.info("  NaN count:      %d", n_nan)
    logger.info("  Infinite count: %d", n_inf)

    quantiles = np.percentile(test_preds, [1, 5, 25, 50, 75, 95, 99])
    logger.info("Prediction quantiles (1, 5, 25, 50, 75, 95, 99): %s", quantiles)

    assert len(test_preds) > 0, "No predictions generated"
    assert n_nan == 0, "NaN values found in predictions"
    assert n_inf == 0, "Infinite values found in predictions"
    assert float(np.min(test_preds)) > 0.0, "Zero-probability prediction detected"
    assert float(np.max(test_preds)) < 1.0, "Probability of 1 detected"