"""
encoding.py
===========
Leakage-safe categorical encoding and feature hashing for the CTR pipeline.

This module covers everything between feature engineering and model fit:

    1. CTR encoding (smoothed)  — replace low-cardinality categoricals
                                   with their alpha-smoothed Bayesian CTR
    2. Frequency encoding       — replace high-cardinality categoricals
                                   with their training-set frequency
                                   (log1p-transformed)
    3. Column-type separation   — partition columns into numeric (for
                                   float feeding to the hasher) and
                                   categorical (for "col=value": 1 dict
                                   feeding to the hasher)
    4. Feature hashing          — project the engineered feature frame
                                   into a fixed 2^22-dim sparse matrix
                                   via scikit-learn's FeatureHasher

The `LearnedEncoders` dataclass bundles every encoder fit on the
training split so the same encoders can be re-applied to validation,
test, and the retrain sample without ever re-fitting. This is the
single most important leakage-prevention pattern in the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack as sparse_vstack
from sklearn.feature_extraction import FeatureHasher

logger = logging.getLogger(__name__)


# =====================================================================
# LEARNED ENCODERS — bundles all train-fit state
# =====================================================================

@dataclass
class LearnedEncoders:
    """Container bundling all encoders fit on the training split.

    Encoders are learned exclusively on the training split (the older 80%)
    and re-applied consistently to validation, test, and retrain samples.
    This is the key mechanism for preventing target leakage from the
    validation or test sets back into the training features.

    Attributes:
        rare_vals_map: For each rare-bucketed column, the set of rare values
            observed in training. Reused so rare categories in test/retrain
            are bucketed consistently.
        ctr_maps: For each low-cardinality column, a Series mapping each
            category to its alpha-smoothed Bayesian CTR estimate.
        freq_maps: For each high-cardinality column, a Series mapping each
            category to its training-set frequency (normalized count).
        global_ctr_train: Mean click rate on the training split. Used as
            the smoothing prior and as a fallback for unseen categories.
        numeric_cols: Columns to be treated as float features in hashing.
        categorical_cols: Columns to be treated as string features
            (one-hot encoded via the hashing trick).
    """

    rare_vals_map: dict[str, set[str]] = field(default_factory=dict)
    ctr_maps: dict[str, pd.Series] = field(default_factory=dict)
    freq_maps: dict[str, pd.Series] = field(default_factory=dict)
    global_ctr_train: float = 0.0
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)


# =====================================================================
# SECTION 1: CTR ENCODING (SMOOTHED, LEAKAGE-SAFE)
# =====================================================================

def learn_ctr_maps(
    train_df: pd.DataFrame,
    cols: tuple[str, ...],
    *,
    alpha: float,
    global_ctr: float,
) -> dict[str, pd.Series]:
    """Compute alpha-smoothed Bayesian CTR for each value in each column.

    For each category ``v`` in column ``c``::

        smoothed_ctr(v) = (clicks(v) + alpha * global_ctr) / (count(v) + alpha)

    The smoothing prior pulls low-count categories toward the global CTR,
    preventing the encoding from overfitting to rare categories with
    extreme observed rates. Higher ``alpha`` = stronger pull to the prior.

    Args:
        train_df: Training split only — never the full dataset.
        cols: Columns to encode.
        alpha: Smoothing strength (number of "virtual" observations of
            the global CTR added to every category).
        global_ctr: Mean click rate on the training split.

    Returns:
        Dictionary of column -> Series mapping category to smoothed CTR.
    """
    ctr_maps: dict[str, pd.Series] = {}
    for col in cols:
        if col not in train_df.columns:
            continue
        stats = train_df.groupby(col)["click"].agg(["sum", "count"])
        ctr = (stats["sum"] + alpha * global_ctr) / (stats["count"] + alpha)
        ctr_maps[col] = ctr
    return ctr_maps


def apply_ctr_features(
    df: pd.DataFrame,
    ctr_maps: dict[str, pd.Series],
    global_ctr_fallback: float,
) -> pd.DataFrame:
    """Replace each low-cardinality column with its smoothed CTR encoding.

    Unseen categories (present in test/validation but absent from training)
    fall back to ``global_ctr_fallback``, which is the training-set CTR.
    """
    df = df.copy()
    for col, ctr_map in ctr_maps.items():
        if col in df.columns:
            df[col + "_ctr"] = (
                df[col].map(ctr_map).fillna(global_ctr_fallback).astype("float32")
            )
            df.drop(columns=[col], inplace=True)
    return df


# =====================================================================
# SECTION 2: FREQUENCY ENCODING
# =====================================================================

def learn_frequency_maps(
    train_df: pd.DataFrame,
    cols: tuple[str, ...],
) -> dict[str, pd.Series]:
    """Learn normalized value frequencies per column from training data."""
    return {
        col: train_df[col].value_counts(normalize=True)
        for col in cols if col in train_df.columns
    }


def apply_frequency_encoding(
    df: pd.DataFrame,
    freq_maps: dict[str, pd.Series],
) -> pd.DataFrame:
    """Replace each column with ``log1p`` of its training-set frequency.

    The ``log1p`` transform compresses the long tail of rare categories
    and ensures that unseen categories (mapped to 0) yield 0 after
    transformation, which is the natural fallback for frequency encoding.

    Args:
        df: DataFrame to transform.
        freq_maps: Dictionary of column -> training frequency Series.

    Returns:
        Copy of ``df`` with each frequency-encoded column replaced by a
        ``<col>_freq`` column of dtype float32.
    """
    df = df.copy()
    for col, freq in freq_maps.items():
        if col in df.columns:
            f = df[col].map(freq).fillna(0.0)
            df[col + "_freq"] = np.log1p(f).astype("float32")
            df.drop(columns=[col], inplace=True)
    return df


# =====================================================================
# SECTION 3: COLUMN TYPE SEPARATION
# =====================================================================

def identify_numeric_and_categorical_columns(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Separate engineered numeric features from columns still categorical.

    By naming convention, columns ending in ``_freq`` or ``_ctr`` are
    numeric encodings of formerly categorical features. All other columns
    at this stage are still categorical and will be one-hot encoded via
    feature hashing.
    """
    numeric_cols = [c for c in df.columns if c.endswith("_freq") or c.endswith("_ctr")]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def cast_categoricals_to_str(
    df: pd.DataFrame,
    categorical_cols: list[str],
) -> pd.DataFrame:
    """Cast categorical columns to string dtype for feature-dict hashing."""
    df = df.copy()
    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


# =====================================================================
# SECTION 4: FEATURE HASHING
# =====================================================================

def _row_to_feature_dict(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Iterator[dict[str, float]]:
    """Stream each row of ``df`` as a feature dict suitable for FeatureHasher.

    Categorical features are emitted as ``"<col>=<value>": 1`` to mimic
    one-hot encoding inside the hash space. Numeric features are emitted
    as ``<col>: float(value)``. The generator approach avoids materializing
    a dense intermediate representation.
    """
    for row in df.itertuples(index=False):
        d: dict[str, float] = {}
        for c in categorical_cols:
            try:
                d[f"{c}={getattr(row, c)}"] = 1
            except AttributeError:
                pass
        for c in numeric_cols:
            try:
                d[c] = float(getattr(row, c))
            except AttributeError:
                pass
        yield d


def hash_features(
    df: pd.DataFrame,
    hasher: FeatureHasher,
    numeric_cols: list[str],
    categorical_cols: list[str],
    *,
    batch_rows: int = 500_000,
) -> csr_matrix:
    """Project a feature DataFrame into a fixed-dimensional sparse matrix.

    Feature hashing trades a small risk of hash collisions for a fixed
    memory footprint regardless of the cardinality of input features.
    At 2^22 dimensions, collisions for this dataset are rare enough to
    have negligible impact on log loss.

    To stay within memory limits on large inputs (10M+ rows), this
    function hashes ``df`` in row batches of size ``batch_rows`` and
    vertically stacks the resulting CSR matrices. The output is
    bit-for-bit identical to hashing the full frame in one shot.

    Args:
        df: Feature DataFrame to hash.
        hasher: Configured ``FeatureHasher`` instance.
        numeric_cols: Names of numeric-valued columns.
        categorical_cols: Names of string-valued (one-hot) columns.
        batch_rows: Rows hashed per batch. Smaller = less memory but
            more Python-side overhead. 500k is a reasonable default
            for 8-16 GB machines.

    Returns:
        Sparse CSR matrix of shape ``(len(df), hasher.n_features)``.
    """
    n_rows = len(df)
    if n_rows <= batch_rows:
        return hasher.transform(
            _row_to_feature_dict(df, numeric_cols, categorical_cols)
        )

    blocks: list[csr_matrix] = []
    for start in range(0, n_rows, batch_rows):
        stop = min(start + batch_rows, n_rows)
        block_df = df.iloc[start:stop]
        block = hasher.transform(
            _row_to_feature_dict(block_df, numeric_cols, categorical_cols)
        )
        blocks.append(block)
        logger.info("  Hashed rows %s / %s", f"{stop:,}", f"{n_rows:,}")

    return sparse_vstack(blocks, format="csr")