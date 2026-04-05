---
name: setup
description: Use this skill to initialize the ML pipeline environment. Load and validate all data files, inspect shapes and types, confirm required columns exist, and produce a structured data report for downstream agents. Trigger this skill at the start of any ML task before designing or coding models.
---

# Setup Agent

Your job is to load all competition data, validate it, and produce a clear report so downstream agents know exactly what they are working with.

## Steps

1. Load all data files from `./data/`:
   - `train.csv` — 57 RTs with sequence, labels, family, and 66 biophysical features
   - `test.csv` — test set RTs (no labels)
   - `esm2_embeddings.npz` — ESM2 mean-pooled embeddings (1280 dims)
   - `family_splits.csv` — family membership for LOFO cross-validation
   - `sample_submission.csv` — submission format reference

2. Validate each file:
   - Check it exists and is readable
   - Print shape, columns, and dtypes
   - Check for missing values
   - Confirm key columns: `rt_name`, `active`, `pe_efficiency_pct`, `rt_family`

3. Summarize the task:
   - Number of samples, class distribution (active vs inactive)
   - Family breakdown
   - Available features: 66 biophysical features already in train.csv + ESM2 embeddings
   - Evaluation metric: CLS (harmonic mean of PR-AUC and Weighted Spearman)

4. Write setup report to `./outputs/setup_report.json`

## Output Format

```json
{
  "status": "done",
  "summary": "Loaded 57 RTs across 7 families. 21 active, 36 inactive. Features: 66 biophysical + 1280 ESM2.",
  "outputs": {
    "n_samples": 57,
    "n_active": 21,
    "n_inactive": 36,
    "families": ["Retroviral", "Retron", "..."],
    "feature_cols": ["perplexity", "instability_index", "..."],
    "esm2_dims": 1280,
    "files_confirmed": ["train.csv", "esm2_embeddings.npz", "family_splits.csv"]
  }
}
```