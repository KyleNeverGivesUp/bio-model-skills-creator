---
name: aggregator
description: Use this skill to ensemble all model predictions and produce the final submission file. Read all prediction files, combine them, and write submission.csv. Trigger this skill after tuner is done, or when enough plans have been evaluated to produce a final submission.
---

# Aggregator Agent

Your job is to combine all available predictions into a final submission file.

## Steps

1. **Collect all prediction files**:
   - `./outputs/preds_plan_*.csv` — individual plan predictions
   - `./outputs/preds_tuned.csv` — tuned predictions (if available)

2. **Ensemble predictions**:
   - Rank-average all prediction sets (convert to ranks, then average)
   - Rank averaging is more robust than score averaging for this metric

   ```python
   from scipy.stats import rankdata
   
   all_preds = [pd.read_csv(f)["predicted_score"].values for f in pred_files]
   ranks = [rankdata(p) for p in all_preds]
   ensemble = np.mean(ranks, axis=0)
   ```

3. **Normalize** final scores to [0, 1] range

4. **Write submission.csv**:
   ```
   rt_name,predicted_score
   MMLV-RT,0.95
   BLV-RT,0.12
   ...
   ```
   All 57 RTs must be present. Missing predictions filled with 0.0.

5. **Compute final CLS** on the ensembled predictions and report it

## Output Format

```json
{
  "status": "done",
  "summary": "Ensembled 3 prediction sets. Final CLS=0.54. submission.csv written.",
  "outputs": {
    "n_models_ensembled": 3,
    "final_cls": 0.54,
    "submission_file": "./submission.csv"
  }
}
```

## Notes

- submission.csv must be in the project root, not in outputs/
- Every RT in train.csv must have a prediction
- Higher score = more likely active and efficient