---
name: aido-rna-1-6b
description: A 1.6 billion parameter RNA language model trained on genomic RNA sequences for representation learning and downstream genomic tasks. Use this model for RNA sequence analysis, feature extraction, and transfer learning on RNA-based biological problems.
---

# AIDO.RNA-1.6B

## Overview
AIDO.RNA-1.6B is a large-scale transformer-based language model pre-trained on RNA sequences from genomic data. The model learns meaningful representations of RNA sequences and can be fine-tuned or used for feature extraction on downstream genomic tasks such as RNA function prediction, secondary structure analysis, and variant effect prediction.

## When to Use
This model is best suited for:
- RNA sequence embedding and representation learning
- Transfer learning on RNA classification tasks
- RNA-based genomic feature extraction
- Variant effect prediction on RNA sequences
- RNA function and interaction prediction
- Secondary structure prediction assistance

## How to Use
Load the model using Hugging Face transformers:

```python
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download

# Download model
model_dir = snapshot_download(repo_id="genbio-ai/AIDO.RNA-1.6B")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("genbio-ai/AIDO.RNA-1.6B")
model = AutoModel.from_pretrained("genbio-ai/AIDO.RNA-1.6B")

# Example RNA sequence
rna_sequence = "AUGCUAGCUAGCUAGCUAG"
inputs = tokenizer(rna_sequence, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state
```

## Input Format
RNA nucleotide sequences provided as strings containing the standard RNA alphabet (A, U, G, C). Sequences can be tokenized using the model's associated tokenizer which handles nucleotide k-merization. Input sequences should be raw RNA sequences without additional formatting.

## Output Format
The model outputs contextualized embeddings for each position in the input sequence. The output shape is (batch_size, sequence_length, hidden_size=1024). These embeddings can be used for downstream classification, clustering, or as features for other models.

## Example
```python
rna_seq = "AUGCUAGCUAGCUAGCUA"
inputs = tokenizer(rna_seq, return_tensors="pt")
outputs = model(**inputs)
# outputs.last_hidden_state shape: (1, seq_len, 1024)
# Use mean pooling for sequence-level representation
sequence_embedding = outputs.last_hidden_state.mean(dim=1)
```

## Notes
- Model requires PyTorch and Hugging Face transformers library
- Input sequences should be valid RNA (A, U, G, C nucleotides)
- Very long sequences may require gradient checkpointing or sequence truncation
- The model was trained on genomic RNA data; performance on non-standard RNA types may vary
- Fine-tuning on downstream tasks typically yields best results
- Refer to original paper and repository for detailed architecture and training methodology