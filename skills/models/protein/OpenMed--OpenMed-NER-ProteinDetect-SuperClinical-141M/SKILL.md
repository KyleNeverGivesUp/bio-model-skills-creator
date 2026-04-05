---
name: openmed-ner-proteindetect-superclinical-141m
description: A 141M parameter DeBERTa-v2 based token classification model for detecting and classifying protein entities in biomedical and clinical text, including proteins, protein complexes, protein families, and protein variants.
---

# OpenMed-NER-ProteinDetect-SuperClinical-141M

## Overview
OpenMed-NER-ProteinDetect-SuperClinical-141M is a specialized named entity recognition (NER) model designed to identify and classify protein-related entities in biomedical and clinical literature. Built on the DeBERTa-v2 architecture with 141M parameters, it detects various protein entity types including individual proteins, protein complexes, protein families/groups, and protein variants. This model is particularly suited for information extraction from clinical research, molecular biology studies, and systems biology applications.

## When to Use
This model is best suited for:
- Extracting protein mentions from biomedical research papers and clinical notes
- Identifying protein complexes and their interactions in molecular biology texts
- Detecting protein variants and mutations in clinical genomics literature
- Building protein-centric knowledge bases from unstructured biomedical text
- Annotating molecular biology datasets for downstream analysis
- Supporting drug discovery pipelines that require protein entity recognition

## How to Use
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_name = "OpenMed/OpenMed-NER-ProteinDetect-SuperClinical-141M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create a NER pipeline
ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer)

# Example usage
text = "The TP53 protein forms a complex with MDM2 and regulates cell cycle progression."
results = ner_pipeline(text)
print(results)
```

## Input Format
The model accepts raw biomedical text as input. Text should be:
- Tokenizable by the DeBERTa-v2 tokenizer
- English language biomedical or clinical content
- Typical input lengths: 128-512 tokens (adjust based on your use case)
- No special preprocessing required beyond standard text normalization

## Output Format
The model outputs token-level predictions with:
- **entity**: The type of protein entity (PROTEIN, PROTEIN_COMPLEX, PROTEIN_FAMILY_OR_GROUP, PROTEIN_VARIANT, or O for non-entities)
- **score**: Confidence score (0-1) for each token prediction
- **start/end**: Character positions of detected entities in the original text
- **word**: The recognized token

## Example
```python
text = "BRCA1 and BRCA2 mutations increase cancer risk in the TP53 pathway."
results = ner_pipeline(text)

# Output example:
# [
#   {'entity': 'B-PROTEIN', 'score': 0.98, 'index': 1, 'word': 'BRCA1'},
#   {'entity': 'B-PROTEIN', 'score': 0.97, 'index': 4, 'word': 'BRCA2'},
#   {'entity': 'O', 'score': 0.99, 'index': 5, 'word': 'mutations'},
#   {'entity': 'B-PROTEIN', 'score': 0.96, 'index': 10, 'word': 'TP53'}
# ]
```

## Notes
- The model uses BIO (Begin-Inside-Outside) tagging scheme for entity classification
- Requires transformers library version compatible with DeBERTa-v2
- Best performance on clinical and biomedical text; may have reduced accuracy on informal or non-English text
- Token-level predictions require post-processing to group sub-word tokens into complete entities
- Model size (~141M parameters) requires ~500MB of GPU memory for inference
- Licensed under Apache 2.0