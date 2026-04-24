# Configuration Reference

All configuration is managed through Python dataclasses defined in
`src/weather_doc_extractor/config.py`.  There is no external config file;
values are changed by passing CLI flags or by editing the dataclasses directly
for programmatic use.

## `AppConfig`

The top-level configuration object, composed of the sections below.

## `IngestConfig`

Controls where the ingest stage reads data from.

| Field | Default | Description |
|-------|---------|-------------|
| `images_dir` | `Daily_rainfall_sample/images` | Directory of document images |
| `transcriptions_dir` | `Daily_rainfall_sample/transcriptions` | Directory of JSON transcriptions |
| `output_dir` | `data/dataset` | Where to write processed dataset records |

## `ModelConfig`

Controls which model is used for inference.

| Field | Default | Description |
|-------|---------|-------------|
| `model_name` | `HuggingFaceTB/SmolVLM-500M-Instruct` | HuggingFace model ID or local adapter path |
| `max_new_tokens` | `2048` | Maximum tokens the model may generate |
| `temperature` | `0.0` | Sampling temperature (0 = greedy) |
| `device` | `auto` | Device placement (`auto`, `cpu`, `cuda`) |

## `TrainingConfig`

Controls the fine-tuning process.

| Field | Default | Description |
|-------|---------|-------------|
| `output_dir` | `outputs/checkpoints` | Root directory for saved adapters |
| `learning_rate` | `2e-4` | AdamW learning rate |
| `epochs` | `3` | Number of training epochs |
| `batch_size` | `1` | Per-device training batch size |
| `gradient_accumulation_steps` | `8` | Effective batch = batch_size × this |
| `eval_split` | `0.1` | Fraction of data used for validation |
| `lora_r` | `8` | LoRA rank |
| `lora_alpha` | `16` | LoRA alpha scaling factor |
| `lora_dropout` | `0.05` | Dropout applied to LoRA layers |
| `lora_target_modules` | `None` | Linear layers to adapt; `None` = all |

## `ProjectPaths`

General-purpose path configuration (used internally by some pipeline functions).

| Field | Default |
|-------|---------|
| `data_dir` | `data` |
| `raw_images_dir` | `data/raw_images` |
| `annotations_dir` | `data/annotations` |
| `outputs_dir` | `outputs` |
| `models_dir` | `models` |
