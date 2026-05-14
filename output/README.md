# Outputs

This folder is where the pipeline writes its final artifact at runtime.

## Files generated here

| File | Description | Size |
|---|---|---|
| `submission.csv` | Final predictions on the 13,015,341-row test set, formatted to 10 decimal places. Two columns: `id` and `P(click)`. | ~370 MB |

## Why this folder is empty on GitHub

`submission.csv` is not committed to the repository — it's ~370 MB and is generated deterministically by running the pipeline. The `.gitignore` excludes everything in this folder except this README.

## Reproducing the output

From the repository root:

```bash
python ctr_pipeline.py
```

The pipeline takes approximately 27 minutes on a typical laptop (16 GB RAM, no GPU) and writes `submission.csv` here when complete. See the main [README](../README.md) for full setup instructions and runtime breakdown.

## Expected output characteristics

If your run completes successfully, the generated `submission.csv` should match these properties:

- **Row count:** 13,015,341
- **Mean predicted CTR:** ~0.218
- **Min / max:** ~0.0004 / 0.987
- **No NaN or infinite values**
- **Validation log loss reported in stdout:** ~0.382
