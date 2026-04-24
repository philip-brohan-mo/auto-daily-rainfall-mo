# Weather Document Extraction

Tools for extracting structured weather observations from document images,
running multimodal mini-LLMs over those images, and fine-tuning improved
extractors with Hugging Face TRL.

## Scope

This repository is organized around three stages:

1. Ingest document images and annotations.
2. Run baseline extraction with a vision-capable mini-LLM.
3. Fine-tune the model with TRL to improve extraction quality.

## Getting started

```bash
conda env create -f environment.yml
conda activate weather-doc-extractor
python -m src.main info
python -m unittest discover -s tests
```

All project code should be run from the `weather-doc-extractor` Conda
environment. This keeps the model stack, training libraries, and Python
version isolated from the system interpreter.

## Planned workflow

```text
document images -> dataset records -> baseline inference -> evaluation -> TRL fine-tuning
```

## Project structure

- `src/weather_doc_extractor/cli.py`: entrypoints for local workflows
- `src/weather_doc_extractor/config.py`: project configuration dataclasses
- `src/weather_doc_extractor/schemas.py`: extraction schemas
- `src/weather_doc_extractor/pipeline.py`: orchestration stubs for ingest, infer, and training
- `tests/`: starter tests for the project contract

## Notes on models

The codebase is intentionally model-agnostic for now. We can plug in a specific
vision-language model family once we choose:

- a first mini-LLM checkpoint,
- the annotation format for weather fields,
- the fine-tuning method (full fine-tune vs LoRA/QLoRA),
- the evaluation metrics we care about most.

## Environment

- `environment.yml` defines the standard local runtime for this repo.
- The intended Python version is `3.11` inside Conda.
- The environment installs the package in editable mode so local source changes
  are picked up immediately.

## Next steps

1. Choose the first document type to support.
2. Define the exact weather schema we want to extract.
3. Add a concrete Hugging Face model adapter and dataset builder.
