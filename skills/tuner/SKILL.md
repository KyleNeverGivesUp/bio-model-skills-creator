---
name: tuner
description: Use this skill to tune hyperparameters of the best-performing model plan. Read existing results, identify the best plan by CLS score, run a hyperparameter search, and save improved predictions. Trigger this skill after coder has run at least one plan and produced results.
---

# Tuner Agent

Your job is to take the best-performing model plan so far and improve its CLS score through hyperparameter tuning.

## Steps

1. **Find the best plan** — read all `./outputs/results_plan_*.json`, rank by CLS score

2. **Load the training script** for the best plan from `./outputs/train_plan_<id>.py`

3. **Run hyperparameter search** using the parameters most likely to matter:
   - Regularization strength (C for LogReg/SVM, n_estimators/max_depth for tree models)
   - Feature preprocessing (PCA components, scaling method)
   - Class weight (to handle imbalance)

   Use a simple grid search or random search — keep it fast given 57 samples.

4. **Evaluate each configuration** with LOFO cross-validation and CLS metric

5. **Save best configuration** to `./outputs/best_config.json`

6. **Save tuned predictions** to `./outputs/preds_tuned.csv`

## Output Format

```json
{
  "status": "done",
  "summary": "Tuned Plan 1. Best CLS improved from 0.42 to 0.51 with C=0.01, PCA=30.",
  "outputs": {
    "base_cls": 0.42,
    "tuned_cls": 0.51,
    "best_params": {"C": 0.01, "pca_components": 30},
    "predictions_file": "./outputs/preds_tuned.csv"
  }
}
```

## Notes

- With 57 samples, keep the search space small (5-20 configurations max)
- Prioritize regularization tuning — overfitting is the main risk
- Document every configuration tried in `./outputs/tuning_log.json`