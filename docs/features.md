# Features

This document describes the feature engineering pipeline for the Avazu CTR prediction model. Each feature group is documented with:

- **Source columns** — which raw Avazu fields it draws from
- **Transformation** — how the raw field becomes a model input
- **Rationale** — why this transformation, grounded in either EDA findings or domain reasoning
- **Implementation location** — which `src/` module owns the transformation

The four feature-engineering modules form a pipeline:

```
raw CSV
    │
    ▼
┌─────────────────────────┐
│ src/ingestion.py        │  Memory-efficient sampling
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ src/feature_engineering │  Time decomposition + rare bucketing +
│   .py                   │  interactions + column drops + split
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ src/encoding.py         │  CTR encoding + frequency encoding +
│                         │  column-type partition + feature hashing
└─────────────────────────┘
    │
    ▼
2^22-dim sparse CSR matrix → src/modeling.py
```

## 1. Time decomposition (`add_time_features`)

**Source:** the raw `hour` column, a YYMMDDHH integer (e.g., `14102100` = 2014-10-21 hour 00).

**Output columns:**

| New column | Formula | Type |
|---|---|---|
| `hour_of_day` | `hour % 100` | uint8, range 0-23 |
| `day` | `(hour // 100) % 100` | uint8, range 1-31 |
| `month` | `(hour // 10000) % 100` | uint8, typically 10 (October) |
| `is_weekend` | `(day % 7) >= 5` | uint8, 0 or 1 |
| `hour_x_weekend` | `hour_of_day * is_weekend` | uint8, range 0-23 |
| `hour_bin` | `hour_of_day // 4` | uint8, range 0-5 (six 4-hour buckets) |
| `date` | `hour // 100` | int32, used only for the time-based split |

**Rationale:** EDA showed CTR varies meaningfully by hour-of-day (peak engagement during evening hours) and by weekday/weekend (different mobile vs. desktop usage patterns). The `hour_x_weekend` interaction captures weekend-specific hourly behavior that a linear model couldn't otherwise represent. The `hour_bin` provides a coarser temporal grouping at lower cardinality, useful for the CTR encoding stage.

The `date` column is created solely to support the time-based train/validation split in Section D of the pipeline. It is dropped before model fitting.

**Implementation:** `src/feature_engineering.py`, Section 1.

## 2. Rare category bucketing (`learn_rare_buckets`, `apply_rare_buckets`)

**Source:** three high-cardinality categorical columns: `site_id`, `site_domain`, `app_id`.

**Transformation:** Any value appearing **fewer than 50 times** in the training set is folded into a single `__RARE__` sentinel.

**Rationale:** EDA revealed that these three columns have tens of thousands of unique values, but the long tail consists of values appearing only once or twice. Treating each rare value as its own category produces near-zero signal AND inflates the hash space. Folding them into `__RARE__` lets the model learn one effective "rare site/app" coefficient, which captures the small but consistent CTR pattern associated with low-traffic sources.

**Leakage safety:** The set of rare values is learned exclusively on the training split and applied unchanged to the validation, test, and retrain samples. If a value is rare in training but common in test, it stays bucketed — this is intentional. The opposite case (rare in test, common in training) doesn't apply since the training set determines the bucket boundary.

The `threshold=50` value is configurable via `config.py` (`rare_threshold`).

**Implementation:** `src/feature_engineering.py`, Section 2.

## 3. Categorical interactions (`add_interactions`)

**Source:** three column pairs identified in EDA as having complementary signal.

**Transformation:** For each pair, create a new column that concatenates the two values with an underscore:

| New column | Source | Reason |
|---|---|---|
| `app_site` | `app_id` + `_` + `site_id` | The pair identifies the exact placement (in-app vs. on-site). |
| `appdom_sitedom` | `app_domain` + `_` + `site_domain` | The pair identifies the publisher/advertiser combination. |
| `C14_C17` | `C14` + `_` + `C17` | Two anonymized Avazu features that EDA shows are highly correlated. |

**Rationale:** Linear models (including logistic regression) can't represent multiplicative interactions between categorical features. Explicit interaction columns force the hashing step to allocate hash buckets to the combined signal, letting the model learn that "app X on site Y" can have a very different CTR than the marginal CTRs of X and Y would suggest.

**Implementation:** `src/feature_engineering.py`, Section 3.

## 4. Column drops (`drop_noisy_columns`)

**Dropped columns:**

| Column | Reason |
|---|---|
| `id` | Row identifier, no predictive value |
| `device_ip` | Extreme cardinality (millions of unique values) with weak per-IP signal |
| `device_id` | Mostly anonymous default value, weak signal |
| `device_model` | Redundant with `device_type` (which has the same coarse information at lower cardinality) |
| `C15` | Near-constant in EDA — virtually all values are 320, so no signal |

**Rationale:** Dropping these columns reduces the hash space without losing meaningful signal. EDA confirmed each drop with cardinality and CTR-vs-value diagnostics.

**Implementation:** `src/feature_engineering.py`, Section 3.

## 5. Time-based train/validation split (`time_based_split`)

**Source:** the engineered `date` column.

**Transformation:** Sort the unique dates, then split chronologically. The earliest 80% of unique dates become the training set; the latest 20% become the validation set.

**Rationale:** A random train/test split would leak future user behavior patterns backward into training, producing optimistic validation log-loss that wouldn't hold in production. CTR systems must predict *future* clicks from a model trained on *past* data, and the train/validation split must mirror that constraint.

**Fallback:** If the data has only one unique date (rare edge case for the EDA sample), the function falls back to a stratified random split with `random_state=42`.

**Implementation:** `src/feature_engineering.py`, Section 4.

## 6. Smoothed Bayesian CTR encoding (`learn_ctr_maps`, `apply_ctr_features`)

**Source:** 10 low-cardinality columns (`LOW_CARDINALITY_FEATURES` in `feature_engineering.py`):

```
hour_of_day, is_weekend, banner_pos,
site_category, app_category,
device_type, device_conn_type,
C16, C18, C1
```

**Formula:** For each category `v` in column `c`:

```
smoothed_ctr(v) = (clicks(v) + alpha * global_ctr) / (count(v) + alpha)
```

Where `alpha = 50` and `global_ctr` is the mean click rate on the training split.

**Rationale:** Replacing a categorical value with its observed CTR is a powerful encoding for low-cardinality columns — it directly conveys "is this value a high-click or low-click context" to the linear model. But the naive version overfits: a category that appears 3 times with 3 clicks would encode as 1.0, which is almost certainly an overestimate.

The Bayesian smoothing pulls low-count categories toward the global CTR. `alpha=50` means a category needs to appear ~50 times before its observed CTR dominates over the prior. This stabilizes the encoding for rare categories without sacrificing precision for common ones.

**Leakage safety:** The CTR map is learned exclusively on the training split. Unseen categories in validation/test fall back to `global_ctr_train` (the training-set CTR).

The output column is named `<original_col>_ctr` and is `float32`-typed. The original column is dropped.

**Implementation:** `src/encoding.py`, Section 1.

## 7. Frequency encoding (`learn_frequency_maps`, `apply_frequency_encoding`)

**Source:** 6 high-cardinality columns (`FREQUENCY_ENCODE_FEATURES`):

```
site_id, site_domain, app_id, C14, C17, C20
```

**Formula:** Replace each value with `log1p(frequency)`, where `frequency` is the training-set frequency of that value.

**Rationale:** For high-cardinality columns where CTR encoding would have too many small-count categories to be stable, frequency encoding captures a useful signal: "popular" values vs. "rare" values often have different conversion patterns. The `log1p` transform compresses the long tail of rare values and gives unseen categories a natural fallback of `log1p(0) = 0`.

The output column is named `<original_col>_freq` and is `float32`-typed. The original column is dropped.

**Implementation:** `src/encoding.py`, Section 2.

## 8. Column-type separation (`identify_numeric_and_categorical_columns`)

After CTR and frequency encoding, columns ending in `_ctr` or `_freq` are numeric encodings. Everything else (interaction columns, anonymized `C` columns, time bins, etc.) is still string-categorical and needs to go through feature hashing.

The pipeline partitions columns by this naming convention. Categorical columns are cast to string dtype (`cast_categoricals_to_str`) so the feature dict can produce string-valued one-hot keys for hashing.

**Implementation:** `src/encoding.py`, Section 3.

## 9. Feature hashing (`hash_features`)

**Output dimension:** 2^22 = 4,194,304 slots.

**Transformation:** Each row is converted to a feature dict:

- **Categorical columns** emit `{"col=value": 1}` entries (one-hot encoding within the hash space)
- **Numeric columns** (the `_ctr` and `_freq` floats) emit `{"col": float_value}` entries

scikit-learn's `FeatureHasher` then maps each key deterministically into one of the 2^22 slots via a hash function, producing a sparse CSR matrix.

**Rationale:** Without feature hashing, one-hot encoding the categorical columns (with millions of unique values after interactions) would produce a feature matrix with billions of columns — impossible to materialize or fit on. Feature hashing trades a small risk of hash collisions for a fixed memory footprint. At 2^22 dimensions, collisions for this dataset are rare enough to have negligible impact on log loss.

The implementation processes large frames in batches of 500,000 rows to keep memory bounded during the retrain stage (10M rows × 2^22 sparse columns).

**Implementation:** `src/encoding.py`, Section 4.

## Summary of the engineered feature space

After all transformations, before hashing:

| Column type | Count | Source |
|---|---|---|
| Numeric (`_ctr`) | 10 | Smoothed Bayesian CTR encodings |
| Numeric (`_freq`) | 6 | log1p-transformed frequency encodings |
| Categorical strings | ~10 | Time bins, anonymized `C` columns, interactions, rare-bucketed identifiers |
| **Total before hashing** | **~26** | |
| **After hashing** | **2^22** | Sparse CSR matrix |

The 26 → 4.2M expansion happens because each *value* in a categorical column becomes its own one-hot dimension in the hash space. The resulting matrix is over 99.9% sparse.
