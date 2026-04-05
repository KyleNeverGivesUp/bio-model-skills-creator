---
name: rnafm
description: A fill-mask foundation model for RNA sequences trained on RNAcentral. Use this model to predict masked nucleotides in RNA sequences and generate contextualized embeddings for downstream RNA analysis tasks.
---

# rnafm

## Overview
rnafm is a bidirectional transformer-based foundation model pre-trained on RNA sequences from RNAcentral using masked language modeling. The model learns to predict masked nucleotides in RNA sequences by attending to surrounding context, enabling it to capture both local and global sequence patterns relevant to RNA structure and function.

## When to Use
This model is best suited for:
- Predicting missing or masked nucleotides in RNA sequences
- Generating contextualized embeddings for RNA sequences for downstream classification or regression tasks
- Analyzing non-coding RNA (ncRNA) sequences including rRNA, tRNA, miRNA, and lncRNA
- Transfer learning for RNA structure prediction, function prediction, or interaction analysis
- Quantifying sequence similarity based on learned representations

## How to Use
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load model and tokenizer
model_name = "multimolecule/rnafm"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Prepare input with masked token
sequence = "AUGCUGCUA[MASK]CGUA"
inputs = tokenizer(sequence, return_tensors="pt")

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get embeddings (contextualized representations)
hidden_states = outputs.hidden_states
```

## Input Format
Expects RNA sequences as strings using standard IUPAC nucleotide codes (A, U, G, C, N, and others). Use the `[MASK]` token to indicate positions to be predicted. Sequences are tokenized into individual nucleotides by the provided tokenizer.

## Output Format
For masked token prediction, the model outputs logits over the vocabulary (nucleotides). Hidden states provide contextualized embeddings of shape `(batch_size, sequence_length, hidden_dim)` for each position in the sequence.

## Example
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("multimolecule/rnafm")
model = AutoModelForMaskedLM.from_pretrained("multimolecule/rnafm")

# Predict masked position
sequence = "AUGCUGCUA[MASK]CGUA"
inputs = tokenizer(sequence, return_tensors="pt")
outputs = model(**inputs)

# Get top predicted nucleotide
mask_token_logits = outputs.logits[0, 9]  # Position of [MASK]
top_nucleotide = tokenizer.decode(mask_token_logits.argmax(axis=-1))
print(f"Predicted nucleotide: {top_nucleotide}")
```

## Notes
- Model uses 12 Transformer layers with 768 hidden dimensions
- Pre-trained on RNAcentral dataset of diverse RNA sequences
- Requires the `transformers` library (version 4.36.0+) and PyTorch
- Model license is AGPL-3.0; check licensing requirements for commercial use
- Input sequences should be cleaned and validated before masking
- For best results on specific RNA types (rRNA, tRNA, etc.), consider fine-tuning on domain-specific datasets