---
description: 'Instructions for building AI pipelines to extract weather data from historical documents, including fine-tuning and checkpoint management.'
applyTo: '**'
---

# Compute Resources

This computer has limited capability (6Gb RAM, no GPU), so the main modelling tasks are designed to run on Azure ML with GPU acceleration for training and inference. Use the local machine for data preparation, validation, orchestration, and code changes only.

Do not assume local caches or local artifacts persist between Azure jobs. Anything that must survive runs should live on the Azure datastore.

# Environments

All work is done in a Conda environment. Environment definitions live in the `azureml/` directory. There may be multiple environment files for different Azure ML compute targets, for example `conda.yml` for V100 and `conda-a100.yml` for A100.

When changing model support or dependencies:
- update the relevant Azure ML environment YAML files,
- re-register the environment if needed,
- keep the local `environment.yml` consistent with the Azure environments where applicable.

# Azure Workflow

Azure ML is the execution target for extraction, evaluation, and fine-tuning.

Use the repository scripts rather than ad hoc commands:
- `scripts/aml_submit.sh` for Azure ML job submission
- `scripts/aml_upload.sh` for uploading datasets to Azure
- `scripts/run_extract.sh` as the Azure extraction entrypoint
- `scripts/create_model_registry_entry.py` and `scripts/list_checkpoints.sh` for checkpoint tracking

Prefer datastore-backed paths for all persistent inputs, outputs, caches, and checkpoints. HF caches should live on the datastore mount, not in `/tmp`.

# Data and Checkpoints

The repository may contain multiple datasets and multiple fine-tuned checkpoints. Treat them as first-class, Azure-hosted artifacts.

When adding or using a new training dataset:
- keep image and transcription directories paired,
- preserve the existing stem naming convention,
- make sure the dataset can be consumed by the ingest and evaluation pipeline.

When adding or using a new checkpoint:
- store it on Azure datastore,
- use the checkpoint workflow and registry rather than local copies,
- make it selectable for extraction through the Azure submission scripts.

# Model Support

This repository supports multiple vision-language model families and versions. When changing model handling:
- preserve backward compatibility for existing presets,
- update model preset configuration, inference routing, fine-tuning logic, and CLI help together,
- add or update tests for detection, message formatting, and training/example construction,
- update documentation alongside code changes.

Be careful with model family-specific behavior. Different model families may require different message formatting, processor handling, or environment variants.

# Editing Expectations

Prefer small, focused changes that fix the root cause.

Do not:
- make unrelated refactors,
- change model behavior without updating tests,
- introduce new dependencies unless they are required and documented,
- move artifacts back to local storage if they are expected to be Azure-hosted.

If a change touches extraction or fine-tuning, verify both paths when practical.

# Documentation

If you change CLI behavior, model presets, checkpoint handling, or Azure job submission:
- update the README or docs as needed,
- keep usage examples aligned with the scripts,
- make sure the docs reflect the current Azure workflow.

# General Guidance

Follow the existing code style and keep changes consistent with the repository structure. If a requested change depends on Azure ML behavior, prefer a repository-native solution over a one-off workaround.

