# Data

This project uses the **Avazu Click-Through Rate Prediction** dataset, a public competition dataset hosted on Kaggle.

- **Source:** https://www.kaggle.com/c/avazu-ctr-prediction
- **License:** Kaggle competition rules apply
- **Size:** ~5.9 GB compressed, ~30M training rows + ~13M test rows

## Files expected in this directory

| Filename | Description |
|---|---|
| `ProjectTrainingData.csv` | Training data with 24 columns including the `click` target |
| `ProjectTestData.csv` | Test data without the `click` column |
| `ProjectSubmission-TeamX.csv` | Template file with row order and `id` column for the submission |

These files are not committed to the repository because of their size and Kaggle's terms of use.

## How to obtain the data

1. Create a Kaggle account if you don't have one
2. Accept the competition rules at https://www.kaggle.com/c/avazu-ctr-prediction/rules
3. Download `train.gz` and `test.gz` from the **Data** tab
4. Decompress and rename to the filenames listed above
5. Place the files in this `data/` directory

The pipeline expects the standard Avazu schema:

```
id, click, hour, C1, banner_pos, site_id, site_domain, site_category,
app_id, app_domain, app_category, device_id, device_ip, device_model,
device_type, device_conn_type, C14, C15, C16, C17, C18, C19, C20, C21
```

The `hour` column is a YYMMDDHH integer timestamp (e.g., `14102100` = 2014-10-21 hour 00).

## Reproducing the pipeline

Once the data is in place, from the repository root:

```bash
pip install -r requirements.txt
cp config.example.py config.py
python src/run_pipeline.py
```

The output submission file will be written to `outputs/submission.csv`.

Expected runtime: ~30-45 minutes on a modern laptop with 16GB RAM (the retrain stage with 10M rows is the slowest, ~15-20 minutes). The hashing stage on the full test set (~13M rows) is the second-most-expensive at ~5-8 minutes.

## Memory considerations

The full pipeline peaks at roughly 14-16 GB of RAM during the retrain stage when the 10M-row sample is hashed to a 2^22-dim CSR matrix. If you're running on an 8 GB machine, lower these in `config.py`:

```python
hash_dim = 2 ** 20             # Was 2 ** 22
retrain_sample_rows = 5_000_000  # Was 10_000_000
hash_batch_rows = 250_000      # Was 500_000
```

These changes will marginally degrade log-loss (~0.001-0.002) but make the pipeline tractable on smaller machines.
