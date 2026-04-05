---
name: rnaernie
description: A masked language model for RNA sequences that predicts masked nucleotides in non-coding RNA (ncRNA) sequences. Use this model for fill-mask tasks on RNA sequences to understand sequence patterns and relationships in genomic data.
---

# rnaernie

## Overview
rnaernie is a transformer-based masked language model trained on RNA sequences from RNAcentral. It learns to predict masked nucleotides in RNA sequences, enabling understanding of RNA sequence composition, structure-function relationships, and genomic patterns in non-coding RNAs. This model is useful for analyzing ncRNA sequences and generating predictions about sequence conservation and functional elements.

## When to Use
This model is best suited for:
- Predicting masked nucleotides in RNA sequences (fill-mask tasks)
- Analyzing non-coding RNA (ncRNA) sequence patterns
- Feature extraction from RNA sequences for downstream tasks
- Understanding sequence relationships in genomic RNA data
- Identifying conserved or functionally important RNA regions

## How to Use
Load and run the model using the Hugging Face transformers library:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "multimolecule/rnaernie"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example: predict masked nucleotides
sequence = "AUGCUAGCUAG[MASK]GCUAGCUA"
inputs = tokenizer(sequence, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
```

## Input Format
The model expects RNA sequences as strings containing the nucleotide characters A, U, G, and C. Use the `[MASK]` token to indicate positions where nucleotide predictions are desired. Sequences should be tokenized using the provided tokenizer.

## Output Format
The model outputs logits for each position in the sequence. For masked positions, the logits represent probability distributions over the four nucleotide tokens (A, U, G, C). The softmax of these logits gives normalized probabilities for each nucleotide at that position.

## Example
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("multimolecule/rnaernie")
model = AutoModelForMaskedLM.from_pretrained("multimolecule/rnaernie")

sequence = "AUGCUAG[MASK]UAGCUAGC"
inputs = tokenizer(sequence, return_tensors="pt")
outputs = model(**inputs)

# Get predictions for masked position
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
logits = outputs.logits[0, mask_token_index]
predicted_token = torch.argmax(logits, dim=-1)
predicted_nucleotide = tokenizer.decode(predicted_token)
print(f"Predicted nucleotide: {predicted_nucleotide}")
```

## Notes
- Model is trained specifically on RNA sequences from RNAcentral, optimized for ncRNA analysis
- Requires `transformers` library and compatible PyTorch/TensorFlow installation
- Model uses SafeTensors format for efficient loading
- Licensed under AGPL-3.0; review license terms before use
- Best performance on sequences similar to RNAcentral training data (non-coding RNAs)