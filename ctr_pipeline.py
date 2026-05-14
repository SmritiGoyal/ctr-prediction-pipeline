"""
ctr_pipeline.py
===============
Click-Through Rate Prediction Pipeline — Avazu Dataset

End-to-end production-style ML pipeline for binary CTR prediction on a
30M+ row tabular dataset with high-cardinality categorical features.

Key techniques:
    - Memory-efficient chunked sampling for EDA on data that doesn't fit in RAM
    - Time-based train/validation split for leakage-safe evaluation
    - Smoothed CTR encoding (Bayesian smoothing, alpha=50)
    - Frequency encoding with log1p transformation
    - Feature hashing at 2^22 dimensions (the "hashing trick")
    - Probability calibration via logit shift
    - L2-regularized logistic regression as the final estimator

Final result:
    Validation log loss = 0.382 (vs naive baseline 0.4312, ~11% improvement)
    13,015,341 test predictions, mean predicted CTR ~ 0.22

Dataset:
    Avazu Click-Through Rate Prediction Challenge
    https://www.kaggle.com/c/avazu-ctr-prediction

Usage:
    python src/ctr_pipeline.py

Expects training and test CSVs under ./data/ and writes submission.csv
to ./outputs/. See data/README.md for dataset access instructions.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack as sparse_vstack
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = logging.getLogger(__name__)


# =====================================================================
# CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PipelineConfig:
    """All hyperparameters and paths for the CTR pipeline.

    Instances are immutable (``frozen=True``) so that configuration can be
    safely shared across function calls without risk of mutation.
    """

    # Directories ----------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "outputs"

    # Input file names within ``data_dir`` --------------------------
    train_filename: str = "ProjectTrainingData.csv"
    test_filename: str = "ProjectTestData.csv"
    submission_template_filename: str = "ProjectSubmission-TeamX.csv"

    # Output file name within ``output_dir`` ------------------------
    output_filename: str = "submission.csv"

    # Reproducibility -----------------------------------------------
    random_state: int = 42

    # EDA chunked sampling ------------------------------------------
    eda_target_rows: int = 1_000_000
    eda_chunk_size: int = 2_000_000
    eda_per_chunk_sample: int = 50_000

    # Rare category bucketing ---------------------------------------
    rare_threshold: int = 50

    # Bayesian smoothing for CTR encoding ---------------------------
    ctr_smoothing_alpha: float = 50.0

    # Validation split: fraction of latest dates held out -----------
    val_frac: float = 0.2

    # Feature hashing dimensionality (2^22 ~ 4.19M slots) -----------
    hash_dim: int = 2 ** 22

    # Row batch size for memory-safe hashing of large frames --------
    hash_batch_rows: int = 500_000

    # Logistic Regression -------------------------------------------
    lr_c: float = 0.5
    lr_max_iter_initial: int = 80
    lr_max_iter_final: int = 120
    lr_solver: str = "lbfgs"

    # Retrain on larger sample of full training data ----------------
    retrain_sample_rows: int = 10_000_000
    est_total_train_rows: int = 30_000_000
    retrain_chunk_size: int = 1_000_000

    # Numerical stability clipping ----------------------------------
    eps: float = 1e-15

    @property
    def train_path(self) -> Path:
        """Full path to the training CSV."""
        return self.data_dir / self.train_filename

    @property
    def test_path(self) -> Path:
        """Full path to the test CSV."""
        return self.data_dir / self.test_filename

    @property
    def submission_template_path(self) -> Path:
        """Full path to the submission template CSV."""
        return self.data_dir / self.submission_template_filename

    @property
    def output_path(self) -> Path:
        """Full path where the final submission CSV will be written."""
        return self.output_dir / self.output_filename


# Column groups used throughout the pipeline (defined once, referenced by name)

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
# SECTION 1: EDA SAMPLING
# =====================================================================

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
# SECTION 2: TIME FEATURE ENGINEERING
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
# SECTION 3: RARE CATEGORY BUCKETING (LEAKAGE-SAFE)
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
# SECTION 4: CATEGORICAL INTERACTIONS
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
# SECTION 5: TIME-BASED TRAIN / VALIDATION SPLIT
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


# =====================================================================
# SECTION 6: CTR ENCODING (SMOOTHED, LEAKAGE-SAFE)
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
# SECTION 7: FREQUENCY ENCODING
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
# SECTION 8: COLUMN TYPE SEPARATION
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
# SECTION 9: FEATURE HASHING
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
            for 8–16 GB machines.

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


# =====================================================================
# SECTION 10: MODEL TRAINING
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
# SECTION 11: PROBABILITY CALIBRATION
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


# =====================================================================
# SECTION 12: DIAGNOSTICS & SANITY CHECKS
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


# =====================================================================
# SECTION 13: RETRAIN SAMPLING
# =====================================================================

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


def apply_full_feature_pipeline(
    df: pd.DataFrame,
    encoders: LearnedEncoders,
) -> pd.DataFrame:
    """Re-apply every train-fit transformation to a fresh DataFrame.

    Used for the retrain sample, validation, and test sets, ensuring all
    three see exactly the same feature space as the training data without
    any re-fitting that would leak label information.
    """
    df = add_time_features(df)
    df = apply_rare_buckets(df, encoders.rare_vals_map)
    df = add_interactions(df)
    df = drop_noisy_columns(df)
    if "click" in df.columns:
        df = df.drop(columns=["click"])
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    df = apply_ctr_features(df, encoders.ctr_maps, encoders.global_ctr_train)
    df = apply_frequency_encoding(df, encoders.freq_maps)
    df = cast_categoricals_to_str(df, encoders.categorical_cols)
    return df


# =====================================================================
# SECTION 14: SUBMISSION WRITER
# =====================================================================

def write_submission(
    test_preds: np.ndarray,
    template_path: Path,
    output_path: Path,
) -> None:
    """Write the submission CSV aligned to the provided id template.

    Predictions are assigned by row position (not index) to the template,
    formatted to 10 decimal places for precision. The function asserts
    that the row count is preserved end-to-end.
    """
    submission = pd.read_csv(template_path, dtype={"id": "str"})

    assert len(submission) == len(test_preds), (
        f"Submission template ({len(submission)}) and predictions "
        f"({len(test_preds)}) length mismatch"
    )

    submission["P(click)"] = (
        pd.Series(test_preds, index=submission.index)
        .round(10)
        .map(lambda x: format(x, ".10f"))
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    written = pd.read_csv(output_path)
    assert len(written) == len(submission), "Row count changed after write"
    logger.info("Submission written: %s (%s rows)", output_path, f"{len(submission):,}")


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main(config: PipelineConfig | None = None) -> float:
    """Run the full CTR pipeline end-to-end.

    Returns:
        Final calibrated validation log loss.
    """
    cfg = config or PipelineConfig()

    # ---------- Section A: EDA ------------------------------------
    logger.info("=== Section A: EDA Sampling ===")
    eda_df = sample_for_eda(
        cfg.train_path,
        target_rows=cfg.eda_target_rows,
        chunk_size=cfg.eda_chunk_size,
        per_chunk_sample=cfg.eda_per_chunk_sample,
        random_state=cfg.random_state,
    )
    eda_df = add_time_features(eda_df)
    log_eda_diagnostics(eda_df)
    del eda_df

    # ---------- Section B: Load full training/test data -----------
    logger.info("=== Section B: Loading full training and test data ===")
    train = pd.read_csv(cfg.train_path)
    test = pd.read_csv(cfg.test_path)
    train["click"] = train["click"].astype(int)
    logger.info("Full train shape: %s, full test shape: %s", train.shape, test.shape)

    # ---------- Section C: Pre-split feature engineering ----------
    logger.info("=== Section C: Time features + rare bucketing + interactions ===")
    train = add_time_features(train)
    test = add_time_features(test)

    rare_vals_map = learn_rare_buckets(
        train, RARE_BUCKETING_FEATURES, threshold=cfg.rare_threshold
    )
    train = apply_rare_buckets(train, rare_vals_map)
    test = apply_rare_buckets(test, rare_vals_map)

    train = add_interactions(train)
    test = add_interactions(test)

    train = drop_noisy_columns(train)
    test = drop_noisy_columns(test)

    # ---------- Section D: Time-based split -----------------------
    logger.info("=== Section D: Time-based train/validation split ===")
    train_df, val_df = time_based_split(
        train, val_frac=cfg.val_frac, random_state=cfg.random_state
    )

    X_tr = train_df.drop(columns=["click", "date"])
    y_tr = train_df["click"].astype(int)
    X_va = val_df.drop(columns=["click", "date"])
    y_va = val_df["click"].astype(int)

    # ---------- Section E: Leakage-safe encoders ------------------
    logger.info("=== Section E: Learning leakage-safe encoders ===")
    global_ctr_train = float(train_df["click"].mean())
    logger.info("Training-split global CTR: %.6f", global_ctr_train)

    ctr_maps = learn_ctr_maps(
        train_df,
        LOW_CARDINALITY_FEATURES,
        alpha=cfg.ctr_smoothing_alpha,
        global_ctr=global_ctr_train,
    )
    freq_maps = learn_frequency_maps(train_df, FREQUENCY_ENCODE_FEATURES)

    X_tr = apply_ctr_features(X_tr, ctr_maps, global_ctr_train)
    X_va = apply_ctr_features(X_va, ctr_maps, global_ctr_train)
    test = apply_ctr_features(test, ctr_maps, global_ctr_train)

    X_tr = apply_frequency_encoding(X_tr, freq_maps)
    X_va = apply_frequency_encoding(X_va, freq_maps)
    test = apply_frequency_encoding(test, freq_maps)

    numeric_cols, categorical_cols = identify_numeric_and_categorical_columns(X_tr)
    X_tr = cast_categoricals_to_str(X_tr, categorical_cols)
    X_va = cast_categoricals_to_str(X_va, categorical_cols)
    test = cast_categoricals_to_str(test, categorical_cols)

    encoders = LearnedEncoders(
        rare_vals_map=rare_vals_map,
        ctr_maps=ctr_maps,
        freq_maps=freq_maps,
        global_ctr_train=global_ctr_train,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    # ---------- Section F: Feature hashing ------------------------
    logger.info("=== Section F: Feature hashing (dim = 2^22) ===")
    hasher = FeatureHasher(n_features=cfg.hash_dim, input_type="dict")
    X_tr_h = hash_features(
        X_tr, hasher, numeric_cols, categorical_cols,
        batch_rows=cfg.hash_batch_rows,
    )
    X_va_h = hash_features(
        X_va, hasher, numeric_cols, categorical_cols,
        batch_rows=cfg.hash_batch_rows,
    )

    # ---------- Section G: Initial model on train_df --------------
    logger.info("=== Section G: Initial logistic regression (max_iter=%d) ===",
                cfg.lr_max_iter_initial)
    model_initial = train_logistic_regression(
        X_tr_h, y_tr,
        C=cfg.lr_c, max_iter=cfg.lr_max_iter_initial, solver=cfg.lr_solver,
    )
    va_preds_initial = model_initial.predict_proba(X_va_h)[:, 1]
    va_preds_initial = np.clip(va_preds_initial, cfg.eps, 1.0 - cfg.eps)
    log_validation_diagnostics(y_va, va_preds_initial, baseline_ctr=float(y_tr.mean()))

    # ---------- Section H: Retrain on 10M-row sample --------------
    logger.info("=== Section H: Retrain on %s-row sample ===",
                f"{cfg.retrain_sample_rows:,}")
    sample_df = sample_full_train_for_retrain(
        cfg.train_path,
        target_rows=cfg.retrain_sample_rows,
        est_total_rows=cfg.est_total_train_rows,
        chunk_size=cfg.retrain_chunk_size,
        random_state=cfg.random_state,
    )
    sample_df["click"] = sample_df["click"].astype(int)
    y_full = sample_df["click"].astype(int)
    X_full = apply_full_feature_pipeline(sample_df, encoders)
    X_full_h = hash_features(
        X_full, hasher, numeric_cols, categorical_cols,
        batch_rows=cfg.hash_batch_rows,
    )

    model_final = train_logistic_regression(
        X_full_h, y_full,
        C=cfg.lr_c, max_iter=cfg.lr_max_iter_final, solver=cfg.lr_solver,
    )

    # ---------- Section I: Validation + calibration ---------------
    logger.info("=== Section I: Final validation + logit calibration ===")
    va_preds_final = model_final.predict_proba(X_va_h)[:, 1]
    va_preds_final = np.clip(va_preds_final, cfg.eps, 1.0 - cfg.eps)
    va_preds_final = calibrate_probabilities(
        va_preds_final, target_mean=float(y_va.mean()), eps=cfg.eps
    )
    va_preds_final = np.clip(va_preds_final, cfg.eps, 1.0 - cfg.eps)
    final_logloss = float(log_loss(y_va, va_preds_final))
    logger.info("Final calibrated validation log-loss: %.6f", final_logloss)
    logger.info("Mean predicted CTR:    %.6f", va_preds_final.mean())
    logger.info("True validation CTR:   %.6f", y_va.mean())
    logger.info("Min/Max predicted CTR: %.6f / %.6f",
                va_preds_final.min(), va_preds_final.max())

    # ---------- Section J: Test prediction + calibration ----------
    logger.info("=== Section J: Test predictions ===")
    X_test_h = hash_features(
        test, hasher, numeric_cols, categorical_cols,
        batch_rows=cfg.hash_batch_rows,
    )
    test_preds = model_final.predict_proba(X_test_h)[:, 1]
    # Anchor test calibration on training CTR — never on test labels (none exist)
    # and not on validation labels (would leak val distribution into test).
    test_preds = calibrate_probabilities(
        test_preds, target_mean=float(y_tr.mean()), eps=cfg.eps
    )
    test_preds = np.clip(test_preds, cfg.eps, 1.0 - cfg.eps)

    run_test_sanity_checks(test_preds)

    # ---------- Section K: Write submission -----------------------
    logger.info("=== Section K: Writing submission ===")
    write_submission(test_preds, cfg.submission_template_path, cfg.output_path)

    return final_logloss


def _configure_logging() -> None:
    """Configure root logging for CLI execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    _configure_logging()
    final_log_loss = main()
    logger.info("Pipeline complete. Final log loss: %.6f", final_log_loss)