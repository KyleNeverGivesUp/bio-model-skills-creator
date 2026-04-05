---
name: openmed-ner-dnadetect-pubmed-109m
description: A BERT-based token classification model for named entity recognition of genomic and molecular entities (genes, proteins, DNA, RNA, cell lines, cell types) in biomedical literature. Use this model to automatically identify and extract molecular biology entities from PubMed abstracts and biomedical texts.
---

# OpenMed-NER-DNADetect-PubMed-109M

## Overview
OpenMed-NER-DNADetect-PubMed-109M is a specialized named entity recognition (NER) model trained on PubMed biomedical literature to identify and classify genomic and molecular entities. The model recognizes six key entity types: DNA sequences, RNA molecules, proteins, genes, cell lines, and cell types. It is built on a 109M parameter BERT architecture fine-tuned for token-level classification in the genomics and molecular biology domain.

## When to Use
This model is ideal for:
- Extracting gene and protein names from biomedical research papers
- Identifying DNA and RNA sequences mentioned in scientific text
- Recognizing cell line and cell type mentions in experimental protocols
- Building biomedical knowledge graphs from literature
- Automated annotation of molecular biology entities in PubMed abstracts
- Supporting literature mining for genomics research

## How to Use
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Load the model and tokenizer
model_name = "OpenMed/OpenMed-NER-DNADetect-PubMed-109M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create a NER pipeline
ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Example usage
text = "The BRCA1 gene encodes a protein that plays a crucial role in DNA repair."
results = ner_pipeline(text)
for entity in results:
    print(f"{entity['word']}: {entity['entity_group']}")
```

## Input Format
The model expects plain text input in English, typically biomedical or genomics-related sentences or abstracts from scientific literature. Text should be tokenizable by the BERT tokenizer and can range from single sentences to short paragraphs.

## Output Format
The model outputs token-level predictions with the following entity types:
- **B-DNA / I-DNA**: DNA sequences
- **B-RNA / I-RNA**: RNA molecules
- **B-Protein / I-Protein**: Protein entities
- **B-Gene / I-Gene**: Gene names
- **B-CellLine / I-CellLine**: Cell line designations
- **B-CellType / I-CellType**: Cell type identifiers
- **O**: Non-entity tokens

The `aggregation_strategy="simple"` merges subword tokens into complete entities.

## Example
```python
text = "HeLa cells were transfected with a plasmid expressing the EGFP protein under control of the TP53 promoter."

results = ner_pipeline(text)
# Output:
# HeLa: B-CellLine
# EGFP: B-Protein
# TP53: B-Gene
```

## Notes
- Model is specifically trained on PubMed abstracts; performance may vary on other biomedical text types
- Requires transformers library (>=4.30.0) and PyTorch/TensorFlow
- Best performance on standard biomedical English text
- Entity boundaries follow BIO tagging scheme (Beginning, Inside, Outside)
- Consider preprocessing text by sentence for optimal entity recognition
- Model size is 109M parameters; suitable for GPU or CPU inference