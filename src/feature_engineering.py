"""
feature_engineering.py
======================
Feature engineering for the Avazu CTR pipeline.

This module covers all transformations from raw Avazu columns to the
pre-encoding feature set:

    1. Time decomposition  — split the YYMMDDHH `hour` column into
                              hour_of_day, day, month, is_weekend,
                              hour_x_weekend, hour_bin, and a `date`
                              column used for the temporal train/val split
    2. Rare bucketing      — fold low-frequency category values into
                              a single __RARE__ sentinel (leakage-safe:
                              learned on training only, reapplied elsewhere)
    3. Interactions        — explicit cross-category features that linear
                              models can't otherwise capture
    4. Column drops        — remove noisy / low-signal columns
    5. Time-based split    — chronological train/validation split with
                              fallback to stratified random split when
                              only one date is present

Column groups are also defined here as module-level constants so they
can be imported by other modules without re-declaration.

The encoding stage (CTR encoding, frequency encoding, feature hashing)
lives in `encoding.py` rather than here.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# =====================================================================
# COLUMN GROUPS (referenced by name across multiple modules)
# =====================================================================

LOW_CARDINALITY_FEATURES: tuple[str, ...] = (
    "hour_of_day", "is_weekend", "banner_pos",
    "site_category", "app_category",
    "device_type", "device_conn_type",
    "C16", "C18", "C1",
)
"""Features with low cardinality, suitable for smoothed CTR encoding."""

FREQUENCY_ENCODE_FEATURES: tuple[str, ...] = (
    "site_id", "site_domain", "app_id", "C14", "C17", "C20",
)
"""High-cardinality features replaced with their training-set frequency."""

COLUMNS_TO_DROP: tuple[str, ...] = (
    "id",            # Row identifier, no predictive value
    "device_ip",     # Extreme cardinality with weak per-IP signal
    "device_id",     # Mostly anonymous default value, weak signal
    "device_model",  # Redundant with device_type
    "C15",           # Near-constant in EDA, no signal
)
"""Columns dropped before modeling for noise, redundancy, or low signal."""

RARE_BUCKETING_FEATURES: tuple[str, ...] = (
    "site_id", "site_domain", "app_id",
)
"""High-cardinality features where rare values are folded into a single bucket."""


# =====================================================================
# SECTION 1: TIME FEATURE ENGINEERING
# =====================================================================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose the Avazu ``hour`` column into interpretable temporal features.

    The raw ``hour`` field is a compact integer timestamp in YYMMDDHH format
    (e.g., 14102100 = 2014-10-21 hour 00). EDA showed that user engagement
    varies significantly by hour-of-day and day-of-week, so we extract these
    components plus a weekday/weekend flag and two interaction terms.

    Args:
        df: DataFrame containing the raw ``hour`` integer column.

    Returns:
        Copy of ``df`` with new columns: ``hour_of_day``, ``day``, ``month``,
        ``is_weekend``, ``hour_x_weekend``, ``hour_bin``, and ``date``.
        The original ``hour`` column is dropped. The ``date`` column is
        used downstream for time-based train/validation splitting and is
        dropped before model fitting.
    """
    df = df.copy()
    hour = df["hour"].astype(np.int64)

    df["hour_of_day"] = (hour % 100).astype("uint8")
    df["day"] = ((hour // 100) % 100).astype("uint8")
    df["month"] = ((hour // 10000) % 100).astype("uint8")
    df["is_weekend"] = ((df["day"] % 7) >= 5).astype("uint8")

    # Interaction features capturing weekend-specific hourly patterns
    df["hour_x_weekend"] = (df["hour_of_day"] * df["is_weekend"]).astype("uint8")
    df["hour_bin"] = (df["hour_of_day"] // 4).astype("uint8")  # 6 four-hour buckets

    # `date` is used for the time-based split; dropped before model fitting
    df["date"] = (hour // 100).astype("int32")
    return df.drop(columns=["hour"])


# =====================================================================
# SECTION 2: RARE CATEGORY BUCKETING (LEAKAGE-SAFE)
# =====================================================================

def learn_rare_buckets(
    df: pd.DataFrame,
    cols: tuple[str, ...],
    threshold: int,
) -> dict[str, set[str]]:
    """Identify rare values per column based on training-set frequency.

    A category is considered "rare" if it appears fewer than ``threshold``
    times in ``df`` (the training set). These rare values are later folded
    into a single ``__RARE__`` bucket to reduce noise from sparsely
    observed categories.

    Args:
        df: Training DataFrame.
        cols: Column names eligible for rare bucketing.
        threshold: Minimum count for a value to be kept as itself.

    Returns:
        Dictionary mapping each column to its set of rare values. Persists
        in :class:`LearnedEncoders` so the same buckets can be applied to
        validation, test, and retrain samples.
    """
    rare_vals_map: dict[str, set[str]] = {}
    for col in cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts()
        rare_vals = vc[vc < threshold].index
        rare_vals_map[col] = set(rare_vals.tolist())
    return rare_vals_map


def apply_rare_buckets(
    df: pd.DataFrame,
    rare_vals_map: dict[str, set[str]],
) -> pd.DataFrame:
    """Replace rare values in each column with the ``__RARE__`` sentinel.

    Operates in-place on a copy of ``df``. Columns missing from
    ``rare_vals_map`` are left untouched.
    """
    df = df.copy()
    for col, rare_set in rare_vals_map.items():
        if col in df.columns:
            df[col] = df[col].where(~df[col].isin(rare_set), "__RARE__")
    return df


# =====================================================================
# SECTION 3: CATEGORICAL INTERACTIONS
# =====================================================================

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-feature interactions for highly correlated category pairs.

    The pairs (app_id, site_id), (app_domain, site_domain), and (C14, C17)
    were identified in EDA as containing complementary signal that linear
    models cannot capture without explicit interaction features.

    Args:
        df: DataFrame after time features and rare bucketing.

    Returns:
        Copy of ``df`` with three new interaction columns where the
        constituent columns exist.
    """
    df = df.copy()
    if "app_id" in df.columns and "site_id" in df.columns:
        df["app_site"] = df["app_id"].astype(str) + "_" + df["site_id"].astype(str)
    if "app_domain" in df.columns and "site_domain" in df.columns:
        df["appdom_sitedom"] = df["app_domain"].astype(str) + "_" + df["site_domain"].astype(str)
    if "C14" in df.columns and "C17" in df.columns:
        df["C14_C17"] = df["C14"].astype(str) + "_" + df["C17"].astype(str)
    return df


def drop_noisy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns identified in EDA as having low or no predictive signal."""
    return df.drop(columns=list(COLUMNS_TO_DROP), errors="ignore")


# =====================================================================
# SECTION 4: TIME-BASED TRAIN / VALIDATION SPLIT
# =====================================================================

def time_based_split(
    df: pd.DataFrame,
    *,
    val_frac: float,
    target_col: str = "click",
    date_col: str = "date",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically so validation is strictly newer than training.

    A time-based split is essential for CTR systems because production
    models must predict future clicks from a model trained on past data.
    A random split would leak future user behavior patterns backward into
    training, producing optimistic validation scores that wouldn't hold
    in production.

    Args:
        df: Full DataFrame containing the date and target columns.
        val_frac: Fraction of unique dates to hold out for validation.
        target_col: Name of the target column.
        date_col: Name of the date column.
        random_state: Used only by the random-split fallback.

    Returns:
        Tuple ``(train_df, val_df)``. If only one unique date is present,
        falls back to a stratified random split.
    """
    dates = np.sort(df[date_col].unique())

    if len(dates) >= 2:
        cutoff = dates[int((1.0 - val_frac) * len(dates))]
        train_df = df[df[date_col] < cutoff].copy()
        val_df = df[df[date_col] >= cutoff].copy()
        logger.info(
            "Time-based split: train < date %s (%d rows), val >= date %s (%d rows)",
            cutoff, len(train_df), cutoff, len(val_df),
        )
        return train_df, val_df

    logger.warning("Only one unique date — falling back to stratified random split")
    return train_test_split(
        df, test_size=val_frac, stratify=df[target_col], random_state=random_state
    )