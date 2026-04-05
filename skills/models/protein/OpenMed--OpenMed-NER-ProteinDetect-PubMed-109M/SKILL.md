---
name: openmed-ner-proteindetect-pubmed-109m
description: A named entity recognition model fine-tuned on PubMed literature to detect and classify protein entities including proteins, protein complexes, protein families, and protein variants. Use this model for biomedical text mining and automated protein mention extraction from scientific literature.
---

# OpenMed-NER-ProteinDetect-PubMed-109M

## Overview
This is a token-classification model based on BERT (109M parameters) trained on PubMed biomedical literature for named entity recognition (NER) of protein-related entities. The model identifies and classifies mentions of proteins, protein complexes, protein families/groups, and protein variants in unstructured biomedical text. It solves the problem of automated protein entity extraction from scientific publications, enabling systematic knowledge extraction for systems biology, molecular biology, and biochemistry research.

## When to Use
- Extracting protein mentions from PubMed abstracts or full-text articles
- Identifying protein complexes and protein families in biomedical literature
- Detecting protein variants in research papers
- Building protein interaction databases from scientific text
- Mining knowledge for protein-centric bioinformatics pipelines
- Preprocessing biomedical text for downstream molecular biology analysis

## How to Use
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Load model and tokenizer
model_name = "OpenMed/OpenMed-NER-ProteinDetect-PubMed-109M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create NER pipeline
ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Example usage
text = "The TP53 protein forms a complex with MDM2 and regulates cell cycle checkpoints."
results = ner_pipeline(text)
print(results)
```

## Input Format
Plain English text from biomedical or scientific literature (PubMed abstracts, full-text articles, or similar scientific documents). The model accepts variable-length text sequences up to the BERT token limit (512 tokens). Input text should be preprocessed by removing excessive whitespace and special characters.

## Output Format
A list of token-level predictions with:
- `entity`: The predicted entity label (e.g., `B-protein`, `I-protein`, `B-protein_complex`, `B-protein_variant`, `B-protein_family_or_group`)
- `score`: Confidence score (0-1) for the prediction
- `word`: The token/subword
- `start`/`end`: Character positions in the original text

Using `aggregation_strategy="simple"` groups subword tokens into complete entities.

## Example
```python
text = "BRCA1 and BRCA2 proteins are critical tumor suppressors in the p53 pathway."
results = ner_pipeline(text)

# Output example:
# [
#   {'entity_group': 'protein', 'score': 0.98, 'word': 'BRCA1', 'start': 0, 'end': 5},
#   {'entity_group': 'protein', 'score': 0.97, 'word': 'BRCA2', 'start': 10, 'end': 15},
#   {'entity_group': 'protein', 'score': 0.96, 'word': 'p53', 'start': 60, 'end': 63}
# ]
```

## Notes
- Model is optimized for PubMed literature; performance may vary on other biomedical text domains
- Requires `transformers>=4.30.0` and `torch>=1.9.0`
- Best performance on well-formatted English text; handle domain-specific abbreviations carefully
- Nested or overlapping protein entities are not supported; model uses standard BIO tagging scheme
- Consider fine-tuning on domain-specific data if working with specialized protein terminology outside PubMed conventions
- Apache 2.0 license permits commercial and research use