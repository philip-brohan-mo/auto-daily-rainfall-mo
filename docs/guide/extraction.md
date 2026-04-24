# Extraction

The extraction stage runs a vision-language model over a document image and
returns a structured JSON rainfall grid.

## Running extraction

```bash
weather-extract extract --model smolvlm <image_path>
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `smolvlm` | Model preset or HuggingFace ID |

### Example

```bash
weather-extract extract \
  --model smolvlm \
  Daily_rainfall_sample/images/DRain_1871-1880_Cornwall-59.jpg
```

Output:

```json
{
  "days": {
    "Day 1": [null, 0.5, 1.2, ...],
    ...
  },
  "totals": [12.4, 8.1, ...]
}
```

## How extraction works

1. The image is loaded and passed to the model along with a prompt asking it
   to return the rainfall data as JSON.
2. The raw text response is parsed with a lenient JSON extractor that strips
   markdown fences and partial responses.
3. If parsing succeeds, a `DailyRainfallGrid` object is returned.  On failure,
   the raw model output is printed to stderr.

## Using a fine-tuned adapter

Pass the adapter directory in place of a model name:

```bash
weather-extract extract \
  --model outputs/checkpoints/HuggingFaceTB--SmolVLM-500M-Instruct \
  Daily_rainfall_sample/images/DRain_1871-1880_Cornwall-59.jpg
```

The code detects that the path is a local LoRA adapter (by checking for
`adapter_config.json`) and loads the base model automatically before applying
the adapter weights.

## Model presets

| Preset | HuggingFace ID |
|--------|---------------|
| `smolvlm` | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| `granite` | `ibm-granite/granite-vision-3.2-2b` |

Any full HuggingFace model ID (e.g. `Qwen/Qwen2-VL-2B-Instruct`) can also be
passed directly.
