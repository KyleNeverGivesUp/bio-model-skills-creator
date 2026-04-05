---
name: openmed-ner-dnadetect-supermedical-125m
description: Token classification model for named entity recognition in biomedical and genomic texts, identifying genes, proteins, DNA/RNA sequences, and cell types. Use this model for extracting molecular biology entities from scientific literature and genomic data.
---

# OpenMed-NER-DNADetect-SuperMedical-125M

## Overview
OpenMed-NER-DNADetect-SuperMedical-125M is a RoBERTa-based token classification model fine-tuned for named entity recognition (NER) in biomedical and genomic domains. It identifies and extracts key molecular biology entities including genes, proteins, DNA sequences, RNA sequences, cell lines, and cell types from unstructured text. This model solves the critical problem of automated entity extraction from biomedical literature and genomic datasets, enabling downstream bioinformatics analysis and knowledge base construction.

## When to Use
- Extracting gene and protein names from scientific publications
- Identifying cell types and cell lines in biomedical texts
- Recognizing DNA and RNA sequences in genomic documents
- Performing entity linking for molecular biology knowledge graphs
- Annotating biomedical corpora for machine learning pipeline construction
- Processing clinical and research genomics reports

## How to Use
```python
from transformers import pipeline
from huggingface_hub import snapshot_download

# Load the model
ner_pipeline = pipeline(
    "token-classification",
    model="OpenMed/OpenMed-NER-DNADetect-SuperMedical-125M"
)

# Example usage
text = "The BRCA1 gene encodes a protein involved in DNA repair mechanisms."
results = ner_pipeline(text)

for entity in results:
    print(f"Entity: {entity['word']}, Type: {entity['entity']}, Score: {entity['score']:.3f}")
```

## Input Format
Plain text strings containing biomedical or genomic content. Text is tokenized automatically by the model's tokenizer. Optimal input lengths are typically under 512 tokens; longer texts should be split into sentences or paragraphs.

## Output Format
Token-level predictions with:
- `word`: The recognized entity token
- `entity`: The entity type (e.g., B-GENE, I-PROTEIN, B-CELL_TYPE)
- `score`: Confidence score (0-1) for the prediction
- `start`/`end`: Character offsets in original text

Entity types include: GENE, PROTEIN, DNA, RNA, CELL_LINE, CELL_TYPE, and others marked with B- (beginning) and I- (inside) prefixes following BIO tagging scheme.

## Example
```python
text = "HER2 overexpression in breast cancer cells promotes MAPK pathway activation."
results = ner_pipeline(text)

# Output might include:
# {'word': 'HER2', 'entity': 'B-GENE', 'score': 0.98}
# {'word': 'MAPK', 'entity': 'B-GENE', 'score': 0.95}
```

## Notes
- Model is optimized for English biomedical texts
- Performance depends on text quality and domain similarity to training data
- Requires transformers library (version 4.0+) and PyTorch/TensorFlow
- Model size: ~125M parameters; suitable for CPU inference but GPU recommended for batch processing
- Compatible with Hugging Face inference endpoints
- Apache 2.0 license allows commercial use
- For best results, preprocess text by removing special characters and normalizing whitespace