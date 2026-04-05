# Bio Model Skills Creator

This project automates the discovery, evaluation, and skill definition generation for biological foundation models available on Hugging Face.

## Bio Model Skills Generator

The core component is located in `scripts/model_search.py`, which serves as the Bio Model Skills Generator. This script performs the following functions:

1. **Search Biological Models**: Scans Hugging Face for foundation models related to biomedical, bioinformatics, protein analysis, genomics, DNA/RNA processing, drug discovery, and medical imaging.

2. **Quality Filtering**: Uses Claude AI to score and filter models based on quality criteria, ensuring only high-quality models (above a configurable threshold) are selected.

3. **Skill Generation**: Automatically generates SKILL.md files for approved models, creating standardized skill definitions that can be used in various applications.

### Configuration

The generator can be configured through environment variables:
- `ANTHROPIC_MODEL`: The Claude model to use (default: claude-sonnet-4-20250514)
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `ANTHROPIC_BASE_URL`: Custom base URL for Anthropic API (optional)

### Usage

Run the generator with:
```bash
python scripts/model_search.py
```

## Next Steps

1. Optimize skills optimization
2. Other skills, such as setup, designer