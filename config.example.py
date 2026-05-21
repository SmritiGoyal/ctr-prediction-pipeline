"""
config.example.py
=================
Template for local configuration. Copy this file to `config.py` and fill in
any path overrides needed for your environment. `config.py` is gitignored
so personal customizations don't leak into the repo.

Usage:
    cp config.example.py config.py
    # then edit config.py if you need to override defaults

The PipelineConfig dataclass below is the single source of truth for every
hyperparameter, path, and tunable in this pipeline. It is consumed by
src/run_pipeline.py at runtime.

All defaults are tuned for the Avazu CTR competition (~30M training rows,
~13M test rows). If you're running on a smaller machine, lower
``hash_dim`` (try 2**20), ``retrain_sample_rows`` (try 5M), and
``hash_batch_rows`` (try 250_000).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    # See data/README.md for instructions on obtaining these from Kaggle.
    train_filename: str = "ProjectTrainingData.csv"
    test_filename: str = "ProjectTestData.csv"
    submission_template_filename: str = "ProjectSubmission-TeamX.csv"

    # Output file name within ``output_dir`` ------------------------
    output_filename: str = "submission.csv"

    # Reproducibility -----------------------------------------------
    random_state: int = 42

    # EDA chunked sampling ------------------------------------------
    # Sample ~1M rows distributed across the temporal span of the data
    # for EDA. Uses chunked reads to keep memory bounded.
    eda_target_rows: int = 1_000_000
    eda_chunk_size: int = 2_000_000
    eda_per_chunk_sample: int = 50_000

    # Rare category bucketing ---------------------------------------
    # Categories appearing < 50 times in training are folded into __RARE__
    rare_threshold: int = 50

    # Bayesian smoothing for CTR encoding ---------------------------
    # Smooths low-count categories toward the global CTR.
    # Higher = stronger pull to the prior; 50 was selected via EDA.
    ctr_smoothing_alpha: float = 50.0

    # Validation split: fraction of latest dates held out -----------
    # Time-based split: validation uses the last 20% of unique dates.
    val_frac: float = 0.2

    # Feature hashing dimensionality (2^22 ~ 4.19M slots) -----------
    # Trade-off: higher = fewer collisions, more memory.
    # 2^22 was chosen as the sweet spot for ~8M unique categorical values
    # across all engineered features at the cost of ~40 MB sparse vector
    # space per million rows.
    hash_dim: int = 2 ** 22

    # Row batch size for memory-safe hashing of large frames --------
    # Hashes 10M+ row frames in batches to avoid peak memory blowup.
    # Smaller batches = less RAM but more Python-side overhead.
    hash_batch_rows: int = 500_000

    # Logistic Regression -------------------------------------------
    # C = inverse regularization strength. Lower C = stronger regularization.
    lr_c: float = 0.5
    lr_max_iter_initial: int = 80
    lr_max_iter_final: int = 120
    lr_solver: str = "lbfgs"

    # Retrain on larger sample of full training data ----------------
    # After the initial fit on ~25M-row train_df, retrain the final model
    # on a 10M-row uniform sample of the full training file for stability.
    retrain_sample_rows: int = 10_000_000
    est_total_train_rows: int = 30_000_000
    retrain_chunk_size: int = 1_000_000

    # Numerical stability clipping ----------------------------------
    # Used to keep probabilities strictly in (eps, 1-eps) before log loss
    # and logit transformations. Avoids log(0) and inf.
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