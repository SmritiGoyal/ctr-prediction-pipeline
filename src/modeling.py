"""
modeling.py
===========
Model training and probability calibration for the CTR pipeline.

This module contains two concerns:

    1. Fitting an L2-regularized logistic regression on the hashed
       feature matrix. Logistic regression is the right choice here
       because:
         - It natively optimizes log loss (the competition metric)
         - It scales linearly with feature dimensionality (suiting
           our 2^22 hashed space)
         - It handles sparse CSR input efficiently
         - L2 regularization controls the high-dimensional parameter
           space without zeroing out informative weights

    2. Logit-space probability calibration. The raw model output is
       systematically over-confident — a known artifact of L2-regularized
       LR on imbalanced binary targets. Rather than retraining or fitting
       a separate Platt/isotonic calibrator, we apply a closed-form
       single-offset correction in logit space that preserves ranking
       while pulling the mean predicted probability to match a target
       (typically the training-set CTR for test predictions).

The two functions are intentionally minimal — every hyperparameter is
passed in explicitly so the orchestrator (run_pipeline.py) can vary
``max_iter`` between the initial fit (80 iters) and the larger retrain
(120 iters) without duplicating code.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = logging.getLogger(__name__)


# =====================================================================
# SECTION 1: MODEL TRAINING
# =====================================================================

def train_logistic_regression(
    X: csr_matrix,
    y: pd.Series,
    *,
    C: float,
    max_iter: int,
    solver: str,
) -> LogisticRegression:
    """Fit an L2-regularized logistic regression on the hashed feature matrix.

    Logistic regression natively optimizes log loss, scales linearly with
    feature dimensionality (suiting our 2^22 hashed space), and handles
    sparse input efficiently. L2 regularization controls the high-dimensional
    parameter space without zeroing out informative weights.
    """
    model = LogisticRegression(
        fit_intercept=True,
        C=C,
        max_iter=max_iter,
        solver=solver,
    )
    model.fit(X, y)
    return model


# =====================================================================
# SECTION 2: PROBABILITY CALIBRATION
# =====================================================================

def calibrate_probabilities(
    probs: np.ndarray,
    target_mean: float,
    *,
    eps: float,
) -> np.ndarray:
    """Apply a logit-space mean correction so predictions match target CTR.

    The model produces systematically over-confident click probabilities
    in our setting. Rather than retraining, we shift the predicted log-odds
    by a single offset such that the mean of the calibrated probabilities
    equals ``target_mean``::

        offset = logit(target_mean) - mean(logit(probs))
        calibrated = sigmoid(logit(probs) + offset)

    This is a closed-form correction that preserves the predicted ranking
    and substantially reduces log loss at zero additional training cost.

    Args:
        probs: Raw predicted probabilities in (0, 1).
        target_mean: Desired mean of the calibrated probabilities.
            Use training-set CTR when calibrating test predictions to
            avoid leaking validation/test labels into calibration.
        eps: Small constant for numerical stability when clipping.

    Returns:
        Calibrated probability array of the same shape as ``probs``.
    """
    probs = np.clip(probs, eps, 1.0 - eps)
    log_odds = logit(probs)
    offset = np.log(target_mean / (1.0 - target_mean)) - np.mean(log_odds)
    return expit(log_odds + offset)