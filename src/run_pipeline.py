"""
run_pipeline.py
===============
End-to-end orchestrator for the Avazu CTR prediction pipeline.

Runs all 11 stages in order, calling out to the modular src/* files:

    Section A — EDA sampling                 (ingestion + diagnostics)
    Section B — Load full training/test data
    Section C — Pre-split feature engineering (feature_engineering)
    Section D — Time-based train/val split    (feature_engineering)
    Section E — Leakage-safe encoders         (encoding)
    Section F — Feature hashing               (encoding)
    Section G — Initial model + diagnostics   (modeling + diagnostics)
    Section H — Retrain on 10M-row sample     (ingestion + encoding + modeling)
    Section I — Final validation + calibration (modeling)
    Section J — Test prediction + calibration (modeling + diagnostics)
    Section K — Write submission

Saves one output artifact to the configured output directory:
    - submission.csv  (id, P(click) — 10-decimal-place predictions)

To run:
    python -m src.run_pipeline

Or directly:
    python src/run_pipeline.py

Configuration is read from config.py at the repository root. Copy
config.example.py to config.py first if you need to override defaults.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import log_loss

# Make the repo root importable so `config` resolves regardless of where
# this script is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Pull PipelineConfig from local config.py if present, else fall back to
# config.example.py. This lets users override defaults without editing
# the template.
try:
    from config import PipelineConfig
except ImportError:
    from config_example import PipelineConfig  # type: ignore[no-redef]
    # If the user hasn't run `cp config.example.py config.py`, Python
    # can't import `config_example` either (the dot-named file). Fall
    # through to a hard import error with a helpful message.

# Module imports
from ingestion import sample_for_eda, sample_full_train_for_retrain
from feature_engineering import (
    LOW_CARDINALITY_FEATURES,
    FREQUENCY_ENCODE_FEATURES,
    RARE_BUCKETING_FEATURES,
    add_time_features,
    add_interactions,
    apply_rare_buckets,
    drop_noisy_columns,
    learn_rare_buckets,
    time_based_split,
)
from encoding import (
    LearnedEncoders,
    apply_ctr_features,
    apply_frequency_encoding,
    cast_categoricals_to_str,
    hash_features,
    identify_numeric_and_categorical_columns,
    learn_ctr_maps,
    learn_frequency_maps,
)
from modeling import calibrate_probabilities, train_logistic_regression
from diagnostics import (
    log_eda_diagnostics,
    log_validation_diagnostics,
    run_test_sanity_checks,
)


logger = logging.getLogger(__name__)


# =====================================================================
# FEATURE PIPELINE COMPOSITION
# =====================================================================
# This helper sits at the orchestrator level (not in feature_engineering
# or encoding) because it composes operations from both modules. Putting
# it in either module would create a circular import.

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
# SUBMISSION WRITER
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