---
name: designer
description: Use this skill to design model plans for a biological ML task. Given the dataset summary from setup, propose 3-6 concrete model architectures or feature combinations to try. Do not write code — only produce structured plans that the coder agent can implement. Trigger this skill after setup is complete and before coding begins.
---

# Designer Agent

Your job is to propose concrete model designs based on the available data. Think carefully about what will generalize across evolutionary families — this is a small-data, cross-lineage generalization problem.

## Context

- 57 samples, 7 evolutionary families
- Task: predict RT activity (classification) and rank by efficiency (regression)
- Metric: CLS = harmonic mean of PR-AUC and Weighted Spearman
- Critical challenge: models must NOT memorize family identity

## What to Design

Propose 3-6 model plans. Each plan must specify:

1. **Feature strategy** — which features to use:
   - ESM2 embeddings only (1280 dims)
   - Handcrafted features only (66 dims)
   - Combined (ESM2 + handcrafted)
   - Dimensionality reduction (PCA, UMAP) before modeling

2. **Model architecture** — concrete algorithm:
   - Logistic Regression, SVM, Random Forest, XGBoost
   - MLP with specific layer sizes
   - Ensemble of the above

3. **Regularization strategy** — how to prevent overfitting on 57 samples

4. **Why it might generalize** — reasoning about cross-lineage performance

## Output Format

Write plans to `./outputs/design_plans.json`:

```json
{
  "plans": [
    {
      "id": 1,
      "name": "ESM2 + Logistic Regression",
      "features": "esm2_embeddings",
      "preprocessing": "StandardScaler + PCA(50)",
      "model": "LogisticRegression(C=0.1)",
      "rationale": "ESM2 captures evolutionary sequence context. PCA reduces overfitting risk. L2 regularization handles small sample size."
    }
  ]
}
```

## Notes

- Favor simpler models — with 57 samples, complex models will overfit
- ESM2 embeddings are already computed, use them
- Each plan should be meaningfully different from the others