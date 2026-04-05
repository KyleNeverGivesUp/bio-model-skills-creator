---
name: aido-rna-1-6b-cds
description: A 1.6B parameter RNA language model fine-tuned for coding sequence (CDS) understanding and analysis. Use this model for RNA sequence representation learning, CDS prediction, and genomic feature extraction tasks.
---

# AIDO.RNA-1.6B-CDS

## Overview
AIDO.RNA-1.6B-CDS is a fine-tuned variant of the AIDO.RNA-1.6B foundation model specialized for coding sequence (CDS) analysis. The model leverages transformer-based architecture to learn contextualized representations of RNA sequences, enabling accurate understanding of coding regions and their biological properties. It solves the problem of extracting meaningful embeddings and predictions from RNA sequences with emphasis on protein-coding regions.

## When to Use
- RNA coding sequence classification and annotation
- CDS boundary detection and prediction
- Gene structure analysis and feature extraction
- RNA sequence representation learning for downstream tasks
- Genomic sequence embedding generation for similarity searches
- Transfer learning for RNA-related computational biology tasks

## How to Use
```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load model and tokenizer
model_name = "genbio-ai/AIDO.RNA-1.6B-CDS"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Prepare RNA sequence
rna_sequence = "AUGCUGAUGUAAUGA"
inputs = tokenizer(rna_sequence, return_tensors="pt")

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

print(embeddings.shape)
```

## Input Format
RNA nucleotide sequences in standard IUPAC notation (A, U, G, C). Sequences should be provided as strings. The tokenizer converts sequences into k-mer tokens for model processing. Input sequences can range from short motifs to full-length transcripts.

## Output Format
The model outputs contextualized token embeddings of shape (batch_size, sequence_length, 768). Each position in the RNA sequence receives a 768-dimensional dense vector representation. These embeddings capture both local and global sequence context for downstream analysis.

## Example
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("genbio-ai/AIDO.RNA-1.6B-CDS")
model = AutoModel.from_pretrained("genbio-ai/AIDO.RNA-1.6B-CDS", trust_remote_code=True)

# Analyze a coding sequence
cds = "AUGCUGCCGAUGUAAAUGAAA"
tokens = tokenizer(cds, return_tensors="pt")
embeddings = model(**tokens).last_hidden_state

# Use embeddings for downstream tasks
print(f"Sequence length: {embeddings.shape[1]}")
print(f"Embedding dimension: {embeddings.shape[2]}")
```

## Notes
- Requires `transformers` library and PyTorch for inference
- Model is optimized for CDS regions; may perform differently on non-coding RNA
- Input sequences are typically tokenized as k-mers; check tokenizer configuration
- GPU recommended for processing long sequences or large batches
- Fine-tuned from genbio-ai/AIDO.RNA-1.6B base model
- License: Check model card for specific licensing terms