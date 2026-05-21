"""
ingestion.py
============
Memory-efficient sampling from the Avazu training CSV.

This module handles every stage where the pipeline reads the (very large)
training file. The Avazu training file has ~30 million rows and ~5.9 GB
on disk; loading it in full peaks at over 20 GB of RAM. The functions
here read the file in chunked passes and Bernoulli/random-sample down
to a manageable size.

Two sampling strategies are used:

1. `sample_for_eda`         — for exploratory data analysis. Samples a
                              fixed number of rows per chunk (preserves
                              temporal distribution, not uniform).
2. `sample_full_train_for_retrain` — for the final model retrain.
                              Uniform Bernoulli sampling, preserves
                              exact temporal density of the source file.

Both are deterministic given the same `random_state`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def sample_for_eda(
    train_path: Path,
    *,
    target_rows: int,
    chunk_size: int,
    per_chunk_sample: int,
    random_state: int,
) -> pd.DataFrame:
    """Memory-efficient stratified-by-time sampling from a large CSV.

    Reads the training CSV in chunks of ``chunk_size`` rows and randomly
    samples up to ``per_chunk_sample`` rows from each chunk until
    ``target_rows`` rows have been collected. This approach is preferred
    over reading the first N rows because it preserves the dataset's
    temporal distribution and captures any distribution drift that
    occurs over the training period.

    Args:
        train_path: Path to the training CSV file.
        target_rows: Total number of rows to collect for EDA.
        chunk_size: Number of rows per pandas read_csv chunk.
        per_chunk_sample: Maximum rows to sample from each chunk.
        random_state: Seed for the random number generator.

    Returns:
        DataFrame containing approximately ``target_rows`` rows randomly
        sampled across the temporal span of the training data.
    """
    rng = np.random.default_rng(random_state)
    samples: list[pd.DataFrame] = []
    rows_collected = 0

    for chunk in pd.read_csv(train_path, chunksize=chunk_size, low_memory=False):
        if rows_collected >= target_rows:
            break

        take = min(per_chunk_sample, target_rows - rows_collected)
        if len(chunk) > take:
            idx = rng.choice(len(chunk), size=take, replace=False)
            chunk = chunk.iloc[idx]

        samples.append(chunk)
        rows_collected += len(chunk)
        logger.info("  Collected %s rows for EDA", f"{rows_collected:,}")

    eda_df = pd.concat(samples, ignore_index=True)
    logger.info("EDA sample shape: %s", eda_df.shape)
    return eda_df


def sample_full_train_for_retrain(
    train_path: Path,
    *,
    target_rows: int,
    est_total_rows: int,
    chunk_size: int,
    random_state: int,
) -> pd.DataFrame:
    """Stream the full training CSV and Bernoulli-sample ~target_rows rows.

    Unlike :func:`sample_for_eda`, which samples a fixed number per chunk,
    this function applies a uniform per-row sampling probability so the
    output preserves the exact temporal density of the source file. Used
    for the final-model retrain on a much larger sample (10M rows).

    Args:
        train_path: Path to the training CSV.
        target_rows: Desired sample size.
        est_total_rows: Estimated total rows in the source file (used to
            compute the Bernoulli sampling probability).
        chunk_size: Chunk size for the streaming read.
        random_state: Seed for the Bernoulli sampling.

    Returns:
        DataFrame of approximately ``target_rows`` rows.
    """
    rng = np.random.default_rng(random_state)
    sampling_p = target_rows / est_total_rows
    parts: list[pd.DataFrame] = []
    kept = 0

    for chunk in pd.read_csv(train_path, chunksize=chunk_size, low_memory=False):
        mask = rng.random(len(chunk)) < sampling_p
        part = chunk.loc[mask]
        parts.append(part)
        kept += len(part)
        if kept >= target_rows:
            break

    sample_df = pd.concat(parts, ignore_index=True)
    logger.info("Sample rows collected for retrain: %s", f"{len(sample_df):,}")
    return sample_df