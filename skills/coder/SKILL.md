---
name: coder
description: Use this skill to implement ML model code based on a designer's plan. Write complete, runnable Python code that loads data, trains the model using LOFO cross-validation, evaluates CLS score, and saves results. Trigger this skill after designer has produced plans, to implement and do a quick test run of each plan.
---

# Coder Agent

Your job is to implement one model plan from the designer and run a quick sanity check. Write clean, complete Python code.

## Data Loading

```python
import pandas as pd
import numpy as np

train = pd.read_csv("./data/train.csv")
embeddings_data = np.load("./data/esm2_embeddings.npz")

# ESM2 embeddings — keys are rt_name
# Align embeddings with train rows
emb_matrix = np.stack([embeddings_data[name] for name in train["rt_name"]])  # (57, 1280)

# Biophysical features — all numeric columns except rt_name, sequence, active, pe_efficiency_pct, rt_family
feature_cols = [c for c in train.columns if c not in 
                ["rt_name", "sequence", "active", "pe_efficiency_pct", "rt_family"]]
X_hand = train[feature_cols].fillna(0).values  # (57, 66)

y = train["active"].values
families = train["rt_family"].values
weights = train["pe_efficiency_pct"].values + 0.01
```

## LOFO Cross-Validation

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

unique_families = np.unique(families)
oof_preds = np.zeros(len(train))

for fam in unique_families:
    train_mask = families != fam
    test_mask = families == fam
    
    X_tr, X_te = X[train_mask], X[test_mask]
    y_tr = y[train_mask]
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    
    model.fit(X_tr, y_tr)
    oof_preds[test_mask] = model.predict_proba(X_te)[:, 1]
```

## CLS Metric

```python
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr

def weighted_spearman(y_true, y_pred, weights):
    n = len(y_true)
    ranks_pred = pd.Series(y_pred).rank().values
    ranks_true = pd.Series(y_true.astype(float)).rank().values
    d = ranks_pred - ranks_true
    numerator = 6 * np.sum(weights * d**2)
    denominator = np.sum(weights) * (n**2 - 1)
    return max(0, 1 - numerator / denominator)

pr_auc = average_precision_score(y, oof_preds)
w_sp = weighted_spearman(y, oof_preds, weights)
cls = 2 * pr_auc * w_sp / (pr_auc + w_sp + 1e-9)
```

## Output Format

Save results to `./outputs/results_plan_<id>.json` and predictions to `./outputs/preds_plan_<id>.csv`.

```json
{
  "status": "done",
  "summary": "Plan 1 (ESM2 + LogReg): CLS=0.42, PR-AUC=0.61, WSpearman=0.31",
  "outputs": {
    "plan_id": 1,
    "cls": 0.42,
    "pr_auc": 0.61,
    "w_spearman": 0.31,
    "predictions_file": "./outputs/preds_plan_1.csv"
  }
}
```

## Notes

- Write full script to `./outputs/train_plan_<id>.py` so it can be re-run
- Handle missing values in biophysical features with fillna(0)
- Keep training fast — this is a sanity check run