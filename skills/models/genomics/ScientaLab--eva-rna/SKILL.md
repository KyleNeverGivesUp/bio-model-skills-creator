---
name: eva-rna
description: A foundation model for RNA sequence analysis that generates contextual embeddings for transcriptomic data, enabling feature extraction from bulk RNA-seq and single-cell RNA datasets for gene expression analysis and immunological applications.
---

# eva-rna

## Overview
EVA-RNA is a transformer-based foundation model designed to extract meaningful biological features from RNA sequences and transcriptomic data. It learns representations of RNA sequences that capture functional and regulatory properties, making it suitable for downstream tasks in gene expression analysis, cell type classification, and immunological studies. The model is pre-trained on diverse RNA-seq datasets and provides general-purpose embeddings for both bulk and single-cell transcriptomics.

## When to Use
- Feature extraction from RNA sequences for downstream classification tasks
- Gene expression representation learning from bulk RNA-seq data
- Single-cell RNA-seq analysis and cell type characterization
- Immunological transcriptome analysis
- Transfer learning for specialized genomics applications
- Building contextualized embeddings for sequence-level or transcript-level predictions

## How to Use
```python
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel
import torch

# Download model
local_dir = snapshot_download(repo_id="ScientaLab/eva-rna")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ScientaLab/eva-rna", trust_remote_code=True)
model = AutoModel.from_pretrained("ScientaLab/eva-rna", trust_remote_code=True)

# Example: Get embeddings for an RNA sequence
rna_sequence = "AUGCUGCUGCUGCUGCUGCUGCUGCU"
inputs = tokenizer(rna_sequence, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
embeddings = outputs.last_hidden_state
```

## Input Format
The model accepts RNA or DNA sequences as strings. Sequences should be:
- Standard nucleotide characters (A, U/T, G, C)
- Token-length compatible with the model's vocabulary (typically 6-mers or nucleotide tokens)
- Can be individual sequences or batched sequences for efficiency
- Sequences are tokenized using the provided tokenizer before input to the model

## Output Format
The model outputs:
- **last_hidden_state**: Token-level embeddings with shape `(batch_size, sequence_length, hidden_dim)`, typically 768 dimensions
- **pooler_output**: Sequence-level embeddings (optional, depending on configuration)
- Embeddings capture biological semantics and can be used directly or as features for downstream models

## Example
```python
# Feature extraction for gene expression analysis
sequences = [
    "AUGCUGCUGCUGCUGCUGCUGCUGCU",
    "GCUGCUGCUGCUGCUGCUGCAUGCAU"
]

# Tokenize and embed
inputs = tokenizer(sequences, padding=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Extract sequence-level features (mean pooling)
embeddings = outputs.last_hidden_state.mean(dim=1)  # Shape: (2, 768)

# Use embeddings as features for downstream tasks
print(embeddings.shape)  # (2, 768)
```

## Notes
- The model requires `trust_remote_code=True` due to custom modeling code
- Designed for RNA/DNA sequence analysis; not suitable for protein sequences
- Compatible with standard transformers library workflows
- Supports GPU acceleration for faster inference on large datasets
- Best performance achieved with sequences up to model's maximum length (typically 512 tokens)
- Single-cell and bulk RNA-seq applications benefit from batch processing for efficiency
- Custom code dependencies must be trusted when loading from Hugging Face