---
name: dnabert-s
description: A BERT-based foundation model for DNA sequence analysis that generates contextualized embeddings for genomic feature extraction, enabling transfer learning across diverse genomics tasks.
---

# DNABERT-S

## Overview
DNABERT-S is a lightweight BERT model pre-trained on DNA sequences that produces contextualized embeddings for genomic data. It encodes biological information from DNA sequences into dense vector representations, enabling downstream applications in genomics without requiring task-specific training from scratch. The model is optimized for efficient feature extraction from DNA sequences of varying lengths.

## When to Use
This model is best suited for:
- Extracting features from DNA sequences for downstream genomics tasks
- Transfer learning in genomic prediction tasks (promoter detection, splice site prediction, etc.)
- Sequence similarity and clustering analysis
- Building embeddings for genome-wide association studies (GWAS)
- Fine-tuning on limited labeled genomic datasets

## How to Use
```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load model and tokenizer
model_name = "zhihan1996/DNABERT-S"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, output_hidden_states=True)

# Example DNA sequence (k-mers with space separation)
dna_sequence = "ACGTACGTACGT"
# Convert to k-mers (example: 3-mers with space)
kmers = " ".join([dna_sequence[i:i+3] for i in range(len(dna_sequence)-2)])

# Tokenize and encode
inputs = tokenizer(kmers, return_tensors="pt", padding=True, truncation=True)

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

print(embeddings.shape)  # [batch_size, sequence_length, 768]
```

## Input Format
- DNA sequences must be tokenized as k-mers (typically 3-mers or 6-mers) separated by spaces
- Sequences should contain only ACGT characters
- Maximum sequence length depends on model configuration (typically 512 tokens)
- Input format: space-separated k-mers (e.g., "ACG TGT ACG TAC")

## Output Format
- **last_hidden_state**: Tensor of shape `[batch_size, sequence_length, 768]` containing contextualized embeddings for each k-mer token
- **pooler_output**: Optional sequence-level representation (if available)
- Each embedding is a 768-dimensional dense vector capturing biological context

## Example
```python
# Extract sequence-level embedding by mean pooling
sequence_embedding = embeddings.mean(dim=1)  # [batch_size, 768]

# Use for classification task
classifier = torch.nn.Linear(768, num_classes)
logits = classifier(sequence_embedding)
```

## Notes
- Requires `trust_remote_code=True` due to custom tokenization
- Input sequences must be pre-processed into k-mers; using incorrect k-mer sizes may reduce performance
- The model was pre-trained with specific k-mer tokenization strategy; consistency in k-mer selection is critical
- Supports GPU acceleration for faster inference on large datasets
- Compatible with Hugging Face Transformers library and text-embeddings-inference
- Licensed under Apache 2.0