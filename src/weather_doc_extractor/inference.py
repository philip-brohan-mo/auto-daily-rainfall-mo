"""Vision-LLM inference for daily-rainfall grid extraction.

Public API
----------
detect_model_family(model_name)
    Return the model family tag ("smolvlm", "granite", or "generic").

build_messages(image_path, model_family)
    Build the chat-message list (with embedded image) for a VLM.

extract_grid(image_path, config)
    Load the configured model and extract a DailyRainfallGrid from an image.

parse_extraction_response(text)
    Parse a raw LLM text response into a DailyRainfallGrid (or None).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from weather_doc_extractor.config import ModelConfig
from weather_doc_extractor.ingest import _coerce_value
from weather_doc_extractor.schemas import DailyRainfallGrid

# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------

#: Known model families and the substrings that identify them.
_FAMILY_PATTERNS: list[tuple[str, list[str]]] = [
    ("smolvlm", ["smolvlm", "idefics"]),
    ("granite", ["granite"]),
]


def detect_model_family(model_name: str) -> str:
    """Return the model family for *model_name*.

    Returns one of ``"smolvlm"``, ``"granite"``, or ``"generic"``.
    The family governs how messages are built and how the processor is called.

    If *model_name* is a local adapter directory, the base model name is read
    from ``adapter_config.json`` and used for family detection.
    """
    name = model_name
    adapter_cfg = Path(model_name) / "adapter_config.json"
    if adapter_cfg.exists():
        import json as _json

        name = _json.loads(adapter_cfg.read_text()).get(
            "base_model_name_or_path", model_name
        )

    lower = name.lower()
    for family, patterns in _FAMILY_PATTERNS:
        if any(p in lower for p in patterns):
            return family
    return "generic"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a data-entry assistant for historical meteorological documents.
The image shows a scanned daily-rainfall register page.
The table has:
  - Rows labelled "Day 1" through "Day 31" plus a "Totals" row.
  - 12 columns, one for each month in order:
    January, February, March, April, May, June,
    July, August, September, October, November, December.
  - Values are rainfall amounts in inches. Missing or illegible cells are blank.

Return ONLY a JSON object with exactly these 32 keys: "Day 1", "Day 2", ..., "Day 31", "Totals"
Each value is a JSON array of EXACTLY 12 numbers, one per month in the order above.
Use JSON null for any blank or illegible cell. Do not use strings for numbers.

Example row with all 12 months present:
"Day 1": [0.12, null, 0.05, 0.0, null, null, 0.38, null, null, 0.22, null, 0.07]
          Jan   Feb   Mar   Apr  May   Jun   Jul   Aug   Sep   Oct   Nov   Dec

Output ONLY the JSON object, no commentary, no markdown fences:"""


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------


def build_messages(
    image_path: Path | None = None,
    model_family: str = "smolvlm",
) -> list[dict[str, Any]]:
    """Return a chat-message list appropriate for *model_family*.

    SmolVLM / Idefics3
        Uses ``{"type": "image"}`` as a placeholder; the PIL image is passed
        separately to ``processor(images=[...])``.

    Granite
        Embeds the image URL/path directly in the message content as
        ``{"type": "image", "url": str(image_path)}`` so that
        ``processor.apply_chat_template`` can resolve it.

    Parameters
    ----------
    image_path:
        Path to the image file.  Required for the ``"granite"`` family;
        ignored for ``"smolvlm"`` (the placeholder carries no path).
    model_family:
        One of ``"smolvlm"``, ``"granite"``, or ``"generic"``.
        ``"generic"`` falls back to the SmolVLM placeholder convention.
    """
    if model_family == "granite":
        image_item: dict[str, Any] = {
            "type": "image",
            "url": str(image_path) if image_path is not None else "",
        }
    else:
        # SmolVLM / generic: placeholder only; PIL image passed separately
        image_item = {"type": "image"}

    return [
        {
            "role": "user",
            "content": [
                image_item,
                {"type": "text", "text": EXTRACTION_PROMPT},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_DAY_KEYS = {f"Day {i}" for i in range(1, 32)}
_MONTHS = 12


def _pad_or_trim(values: list[Any], length: int = _MONTHS) -> list[float | None]:
    """Normalise a list to exactly *length* coerced floats-or-None."""
    result = [_coerce_value(v) for v in values[:length]]
    result += [None] * max(0, length - len(result))
    return result


def _repair_truncated_json(text: str) -> str:
    """Attempt to close a truncated JSON object.

    If the model generation was cut off before the closing ``}``, this
    function closes any open array, then closes the object.  Returns the
    original text unchanged if it already ends with ``}``.
    """
    text = text.rstrip()
    if text.endswith("}"):
        return text

    # Close an open array if present, then close the object
    if not text.endswith("]"):
        # Drop any trailing partial token (e.g. a half-written number)
        text = re.sub(r",\s*$", "", text)  # trailing comma
        text = re.sub(r"[\d.]+\s*$", "", text)  # trailing partial number
        text = text.rstrip().rstrip(",")
        text += "]"
    text += "\n}"
    return text


def _extract_object(text: str) -> str | None:
    """Return the first JSON object literal found in *text*.

    Handles three common model output patterns:
    - Plain object: ``{...}``
    - Array-wrapped: ``[{...}]`` (Granite sometimes wraps in an array)
    - Prose + object: ``"Here is the data: {...}"``

    Returns the object string, or ``None`` if no ``{`` is found.
    """
    start = text.find("{")
    if start == -1:
        return None
    fragment = text[start:]
    # Strip a trailing array bracket that some models (Granite) append
    # e.g. "{ ... }\n]" → "{ ... }"
    fragment = re.sub(r"\}\s*\]\s*$", "}", fragment.rstrip())
    return fragment


def parse_extraction_response(text: str) -> DailyRainfallGrid | None:
    """Attempt to parse *text* into a :class:`DailyRainfallGrid`.

    Tolerances applied:
    - Strips markdown code fences.
    - Handles array-wrapped JSON output ``[{...}]`` (Granite convention).
    - Searches for the first ``{…}`` block if prose surrounds the JSON.
    - Attempts to repair truncated JSON (missing closing ``}``).
    - Coerces string numbers to float.
    - Returns ``None`` only if no parseable JSON structure can be found.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip()

    fragment = _extract_object(cleaned)
    if fragment is None:
        return None

    # Try to parse as-is first; if that fails, attempt repair
    raw: dict[str, list[Any]] | None = None
    for candidate in (fragment, _repair_truncated_json(fragment)):
        try:
            raw = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    if raw is None:
        return None

    days: dict[str, list[float | None]] = {}
    totals: list[float | None] = [None] * _MONTHS

    for key, values in raw.items():
        if not isinstance(values, list):
            continue
        if key in _DAY_KEYS:
            days[key] = _pad_or_trim(values)
        elif key.lower() == "totals":
            totals = _pad_or_trim(values)

    if not days:
        return None

    # Fill in any missing day keys with all-None rows
    for i in range(1, 32):
        days.setdefault(f"Day {i}", [None] * _MONTHS)

    return DailyRainfallGrid(days=days, totals=totals)


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------


def _is_adapter_path(name: str) -> bool:
    """Return True if *name* is a local directory containing a LoRA adapter."""
    p = Path(name)
    return p.is_dir() and (p / "adapter_config.json").exists()


def _gpu_dtype() -> "torch.dtype":
    """Return the best floating-point dtype for the available GPU.

    * ``bfloat16`` — Ampere and newer (compute capability >= 8.0, e.g. A100)
    * ``float16``  — older CUDA GPUs (e.g. V100, T4) that lack bfloat16 support
    * ``float32``  — CPU fallback (also used when CUDA is unavailable)
    """
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if major >= 8 else torch.float16


def _log_device_info() -> None:
    """Print GPU/CUDA diagnostics to stdout for AML job logs."""
    import torch

    print("[device] torch version:", torch.__version__)
    print("[device] CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"[device] CUDA device count: {n}")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            print(
                f"[device]   GPU {i}: {props.name}  "
                f"CC={props.major}.{props.minor}  "
                f"VRAM={props.total_memory // 1024**3}GB"
            )
        print("[device] dtype selected:", _gpu_dtype())
    else:
        print("[device] WARNING: no CUDA device found — running on CPU")
        # Print CUDA version the build was compiled against (may differ from driver)
        print("[device] CUDA build version:", torch.version.cuda)
    print(flush=True)

def _load_model_and_processor(config: ModelConfig):  # type: ignore[return]
    """Load the processor and model from HuggingFace (or a local LoRA adapter).

    If *config.model_name* points to a local directory that contains
    ``adapter_config.json``, the base model is read from the adapter config
    and the LoRA weights are applied via ``peft.PeftModel``.

    Raises ``ImportError`` if the ``train`` extras are not installed.
    """
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "Install the 'train' extras to run inference: " "pip install -e '.[train]'"
        ) from exc

    _log_device_info()

    if _is_adapter_path(config.model_name):
        import json as _json

        from peft import PeftModel

        adapter_dir = Path(config.model_name)
        adapter_cfg = _json.loads((adapter_dir / "adapter_config.json").read_text())
        base_model_name = adapter_cfg["base_model_name_or_path"]

        processor = AutoProcessor.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        base = AutoModelForImageTextToText.from_pretrained(
            base_model_name,
            torch_dtype=_gpu_dtype(),
            device_map=config.device,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(adapter_dir))
    else:
        processor = AutoProcessor.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            config.model_name,
            torch_dtype=_gpu_dtype(),
            device_map=config.device,
            trust_remote_code=True,
        )

    model.eval()
    return processor, model


def extract_grid(
    image_path: Path,
    config: ModelConfig,
) -> tuple[DailyRainfallGrid | None, str]:
    """Run the configured VLM over *image_path* and return ``(grid, raw_text)``.

    Loads the model and processor on every call.  When processing many images,
    use :func:`extract_grid_with_model` to load once and reuse.

    *grid* is ``None`` when the model response cannot be parsed.

    Parameters
    ----------
    image_path:
        Path to the document image (JPEG, PNG, …).
    config:
        :class:`~weather_doc_extractor.config.ModelConfig` controlling which
        model to load and its generation parameters.
    """
    processor, model = _load_model_and_processor(config)
    family = detect_model_family(config.model_name)
    return extract_grid_with_model(image_path, config, processor, model, family)


def extract_grid_with_model(
    image_path: Path,
    config: ModelConfig,
    processor: Any,
    model: Any,
    family: str,
) -> tuple[DailyRainfallGrid | None, str]:
    """Run inference using an already-loaded *model* and *processor*.

    Use this in batch loops to avoid reloading weights for every image.
    Obtain *processor*, *model*, and *family* once via::

        processor, model = _load_model_and_processor(config)
        family = detect_model_family(config.model_name)

    Parameters
    ----------
    image_path:
        Path to the document image.
    config:
        Model generation parameters (``max_new_tokens``, ``temperature``).
    processor:
        HuggingFace processor already loaded for this model.
    model:
        HuggingFace model already loaded and on the target device.
    family:
        Model family string from :func:`detect_model_family`.
    """
    try:
        import torch
        from PIL import Image as PILImage
    except ImportError as exc:
        raise ImportError(
            "Install the 'train' extras to run inference: " "pip install -e '.[train]'"
        ) from exc

    image = PILImage.open(image_path).convert("RGB")
    messages = build_messages(image_path, model_family=family)

    do_sample = config.temperature > 0.0
    generate_kwargs: dict[str, Any] = dict(
        max_new_tokens=config.max_new_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        generate_kwargs["temperature"] = config.temperature

    if family == "granite":
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        text_prompt: str = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)

    input_len = inputs["input_ids"].shape[1]
    raw_text: str = processor.batch_decode(
        output_ids[:, input_len:],
        skip_special_tokens=True,
    )[0]

    grid = parse_extraction_response(raw_text)
    return grid, raw_text
