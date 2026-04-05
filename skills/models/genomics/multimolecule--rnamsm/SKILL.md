---
name: rnamsm
description: A fill-mask foundation model for RNA sequences trained on non-coding RNA families from Rfam. Use this model for masked token prediction tasks on RNA sequences, including secondary structure prediction and functional annotation of ncRNA regions.
---

# rnamsm

## Overview
rnamsm is a transformer-based foundation model trained on non-coding RNA (ncRNA) sequences from the Rfam database. The model learns contextual representations of RNA sequences and is designed for masked language modeling tasks. It can predict masked tokens in RNA sequences, enabling downstream applications like functional annotation, structural inference, and variant effect prediction on RNA molecules.

## When to Use
This model is best suited for:
- Predicting masked nucleotides in RNA sequences (fill-mask tasks)
- Learning RNA sequence representations for transfer learning
- Analyzing non-coding RNA families and their variants
- Functional annotation of ncRNA regions
- RNA variant impact assessment
- Secondary structure-aware sequence analysis

## How to Use
Load and use the model with the Hugging Face transformers library:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "multimolecule/rnamsm"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example: mask and predict
sequence = "AUGCUGCUGCUGC"
inputs = tokenizer(sequence, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
```

## Input Format
Sequences should be:
- RNA sequences in standard IUPAC notation (A, U, G, C, and ambiguous bases N, R, Y, etc.)
- Tokenized using the model's associated tokenizer
- Individual tokens represent nucleotides or special tokens ([CLS], [SEP], [MASK])
- Variable length sequences (typically processed up to model's max sequence length)

## Output Format
The model outputs:
- **logits**: Token probabilities for each position in the sequence (shape: batch_size × sequence_length × vocab_size)
- For masked positions ([MASK] tokens), the logits represent probability distributions over all possible nucleotides
- Higher logit values indicate higher predicted probability for that nucleotide at that position

## Example
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("multimolecule/rnamsm")
model = AutoModelForMaskedLM.from_pretrained("multimolecule/rnamsm")

# RNA sequence with masked position
sequence = "AUGCUGC[MASK]GCUGC"
inputs = tokenizer(sequence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    
# Get predicted nucleotide at masked position
masked_logits = outputs.logits[0, 7, :]  # position 7 is [MASK]
predicted_token = torch.argmax(masked_logits).item()
predicted_nucleotide = tokenizer.decode(predicted_token)
```

## Notes
- Model is trained exclusively on non-coding RNA sequences; performance on coding sequences may be limited
- Requires transformers library version compatible with the model checkpoint
- Distributed under AGPL-3.0 license
- Best performance on sequences similar to Rfam ncRNA families
- Tokenizer and model architecture details are available in the model repository
- Consider fine-tuning on task-specific RNA data for optimal performance on specialized applications