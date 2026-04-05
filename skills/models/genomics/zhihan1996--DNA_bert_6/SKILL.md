---
name: dna-bert-6
description: A BERT-based foundation model pre-trained on DNA sequences for genomic understanding and fill-mask prediction tasks. Use this model for predicting masked nucleotides in DNA sequences and understanding genomic patterns.
---

# DNA_bert_6

## Overview
DNA_bert_6 is a BERT transformer model pre-trained on DNA sequences to learn genomic language representations. It performs masked language modeling on DNA, enabling it to predict missing or masked nucleotides within genomic contexts. This model captures sequence-level patterns and dependencies in DNA that are useful for various genomics tasks including sequence analysis, variant interpretation, and genomic understanding.

## When to Use
This model is best suited for:
- Predicting masked nucleotides in DNA sequences (fill-mask task)
- Learning contextualized DNA sequence embeddings for downstream genomics tasks
- Analyzing DNA sequence patterns and motifs
- Feature extraction for genomic machine learning pipelines
- Understanding nucleotide relationships in genomic contexts

## How to Use
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load model and tokenizer
model_name = "zhihan1996/DNA_bert_6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example: predict masked nucleotide
sequence = "ACGTACGT[MASK]ACGTACGT"
inputs = tokenizer(sequence, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits
```

## Input Format
The model expects DNA sequences as strings containing the standard nucleotides (A, C, G, T) and the special [MASK] token to indicate positions to predict. Sequences should be tokenized using the provided tokenizer, which breaks sequences into 6-mer (6 nucleotide) tokens with stride-1 overlapping.

## Output Format
The model outputs logits for each position, representing the model's confidence scores across the vocabulary (nucleotide tokens). For fill-mask tasks, the output is a probability distribution over possible nucleotides at masked positions. Use softmax to convert logits to probabilities.

## Example
```python
from transformers import pipeline

# Use pipeline for fill-mask
nlp = pipeline("fill-mask", model="zhihan1996/DNA_bert_6")
results = nlp("ACGTACGT[MASK]ACGTACGT")

# Returns top predictions:
# [{'token_str': 'A', 'score': 0.45, ...}, 
#  {'token_str': 'C', 'score': 0.30, ...}, ...]
```

## Notes
- The model uses 6-mer tokenization of DNA sequences
- Requires PyTorch and Hugging Face transformers library
- Custom code may be required for some operations; ensure `trust_remote_code=True` if needed
- Best performance on genomic sequences similar to training data distribution
- Supports GPU acceleration for faster inference