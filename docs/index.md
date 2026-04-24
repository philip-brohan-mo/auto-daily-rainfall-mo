# Auto Daily Rainfall

**Automated extraction of daily rainfall data from scanned historical documents
using small, locally-run vision-language models.**

---

## What this project does

Meteorological archives contain millions of handwritten and typed daily rainfall
registers stretching back over a century.  This project uses small
[vision-language models (VLMs)](https://huggingface.co/models?pipeline_tag=image-text-to-text)
from Hugging Face to read those document images and output structured JSON — no
cloud API required.

The pipeline has three stages:

```
Scanned document images
       │
       ▼
  1. Ingest          — pair images with human transcriptions
       │
       ▼
  2. Extract         — run a VLM over each image → JSON grid
       │
       ▼
  3. Evaluate / Fine-tune — measure accuracy; improve the model with LoRA
```

## Sample document

The repository ships with a small set of sample images from the
`Daily_rainfall_sample/` directory.  Each image is a monthly grid of daily
rainfall totals (mm) for a single UK station:

| Column | Contents |
|--------|----------|
| Rows | Day 1 – Day 31 |
| Columns | Jan – Dec |
| Extra row | Monthly totals |

## Supported models

| Short name | HuggingFace ID | Size |
|------------|---------------|------|
| `smolvlm`  | `HuggingFaceTB/SmolVLM-500M-Instruct` | 500 M params |
| `granite`  | `ibm-granite/granite-vision-3.2-2b`   | 2 B params |

Any HuggingFace model ID can be passed directly with `--model`.

## Key features

- **Runs locally** — models are downloaded once and cached by HuggingFace.
- **LoRA fine-tuning** — improve extraction quality with your own transcribed data in a few hours on a single GPU.
- **Adapter loading** — pass a fine-tuned adapter directory as `--model` and it loads automatically.
- **Diagnostic figures** — side-by-side image / extracted-table visualisation with blue/red cell colouring against ground truth.

## Quick links

- [Installation](installation.md)
- [Quick start](quickstart.md)
- [CLI reference](reference/cli.md)
