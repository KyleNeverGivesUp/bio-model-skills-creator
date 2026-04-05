---
name: openmed-ner-dnadetect-bigmed-560m
description: A 560M parameter token-classification model for biomedical named entity recognition, specializing in detecting DNA, RNA, proteins, genes, cell lines, and cell types in genomics and molecular biology texts.
---

# OpenMed-NER-DNADetect-BigMed-560M

## Overview
OpenMed-NER-DNADetect-BigMed-560M is a specialized named entity recognition (NER) model built on XLM-RoBERTa architecture for identifying and classifying biomedical entities in genomics and molecular biology literature. The model extracts mentions of DNA sequences, RNA molecules, proteins, genes, cell lines, and cell types from unstructured biomedical text, enabling automated knowledge extraction from scientific publications and genomic data.

## When to Use
This model is best suited for:
- Extracting gene and protein names from biomedical literature
- Identifying DNA and RNA sequences mentioned in text
- Recognizing cell line and cell type nomenclature in scientific documents
- Automatic annotation of genomic and molecular biology research papers
- Building knowledge bases from biomedical publications
- Supporting literature mining pipelines in genomics research

## How to Use
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load model and tokenizer
model_id = "OpenMed/OpenMed-NER-DNADetect-BigMed-560M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)

# Create NER pipeline
ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer)

# Run inference
text = "The BRCA1 gene encodes a protein that suppresses tumor growth in human cells."
results = ner_pipeline(text)
for entity in results:
    print(f"{entity['word']}: {entity['entity']} (score: {entity['score']:.3f})")
```

## Input Format
The model expects plain text input containing biomedical content. Text should be:
- Natural language sentences or paragraphs from scientific literature
- UTF-8 encoded
- Tokenized automatically by the XLM-RoBERTa tokenizer
- Up to the model's maximum sequence length (typically 512 tokens)

## Output Format
The model outputs token-level classifications with:
- **word**: Individual tokens or subword pieces
- **entity**: Entity type label (e.g., B-PROTEIN, I-GENE, B-DNA, B-CELL_LINE, B-CELL_TYPE)
- **score**: Confidence score (0-1) for the classification
- **start/end**: Character positions in the original text

Entity tags use BIO (Begin-Inside-Outside) notation where B- indicates the start of an entity and I- indicates continuation.

## Example
```python
text = "TP53 mutations in MCF-7 cells lead to p53 protein degradation."
results = ner_pipeline(text)

# Output example:
# TP53: B-GENE (0.987)
# MCF-7: B-CELL_LINE (0.994)
# p53: B-PROTEIN (0.989)
# protein: I-PROTEIN (0.991)
```

## Notes
- Model is trained on biomedical literature with focus on genomics domain
- Multilingual support through XLM-RoBERTa base, optimized for English biomedical text
- For optimal performance, use domain-specific biomedical text
- Long documents should be split into sentences before processing
- Output confidence scores should be thresholded (typically >0.7) for production use
- Model requires transformers library ≥4.30.0
- Apache 2.0 licensed and suitable for research and commercial applications