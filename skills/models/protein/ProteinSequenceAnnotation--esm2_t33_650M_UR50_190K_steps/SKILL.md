---
name: esm2-t33-650m-ur50-190k-steps
description: ESM-2 protein language model with 33 transformer layers trained on 650M parameters for protein sequence understanding and representation learning. Use this model for protein annotation, function prediction, and sequence-based analysis tasks.
---

# esm2_t33_650M_UR50_190K_steps

## Overview
This is an ESM-2 (Evolutionary Scale Modeling) protein language model trained on protein sequences from UniRef50. The model learns contextual representations of amino acid sequences through unsupervised pre-training, enabling it to capture evolutionary and functional information embedded in protein sequences. It's designed for downstream protein analysis tasks including structure prediction, function annotation, and sequence-based property prediction.

## When to Use
- Protein sequence representation and embedding generation
- Protein function and annotation prediction
- Evolutionary relationship analysis
- Transfer learning for downstream protein classification tasks
- Sequence-based protein property prediction
- Protein interaction prediction

## How to Use
```python
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel
import torch

# Download model
local_dir = snapshot_download(repo_id="ProteinSequenceAnnotation/esm2_t33_650M_UR50_190K_steps")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ProteinSequenceAnnotation/esm2_t33_650M_UR50_190K_steps")
model = AutoModel.from_pretrained("ProteinSequenceAnnotation/esm2_t33_650M_UR50_190K_steps")

# Prepare protein sequence
sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
inputs = tokenizer(sequence, return_tensors="pt")

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

print(f"Embedding shape: {embeddings.shape}")
```

## Input Format
Protein sequences as strings containing single-letter amino acid codes (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y). The model uses a standard ESM tokenizer. Sequences can be single proteins or multiple sequences (batched processing supported).

## Output Format
The model outputs contextual embeddings of shape `(batch_size, sequence_length, 1280)` where 1280 is the hidden dimension. The `last_hidden_state` contains per-token representations that capture evolutionary and functional information. These embeddings can be used as features for downstream tasks or aggregated (e.g., mean pooling) for sequence-level representations.

## Example
```python
# Get sequence-level representation via mean pooling
sequence_embedding = embeddings.mean(dim=1)  # Shape: (batch_size, 1280)

# Use for downstream classification task
import torch.nn as nn
classifier = nn.Linear(1280, num_protein_classes)
logits = classifier(sequence_embedding)
```

## Notes
- This model produces embeddings; it is not designed for direct protein structure prediction without additional modules.
- Input sequences should be reasonable biological lengths (typically <4000 residues for memory efficiency).
- The model was trained on UniRef50, so performance may vary on sequences very different from natural proteins.
- Requires `transformers` and `torch` libraries.
- For optimal performance on GPU, consider batch processing multiple sequences.
- The 650M parameter variant offers a balance between computational cost and representation quality compared to larger ESM-2 models.