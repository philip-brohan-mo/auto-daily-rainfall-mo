"""Vision-LLM inference for daily-rainfall grid extraction.

Public API
----------
detect_model_family(model_name)
    Return the model family tag ("smolvlm", "granite", "granite4", or
    "generic").

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
#: Order matters — more specific prefixes must come before broader ones.
_FAMILY_PATTERNS: list[tuple[str, list[str]]] = [
    ("gemma4", ["gemma-4"]),
    ("gemma3", ["gemma-3"]),
    ("smolvlm2", ["smolvlm2"]),
    ("smolvlm", ["smolvlm", "idefics"]),
    ("granite4", ["granite-vision-4.1"]),
    ("granite", ["granite"]),
    ("ministral", ["mistral-small-3", "pixtral"]),
]


def detect_model_family(model_name: str) -> str:
    """Return the model family for *model_name*.

    Returns one of ``"smolvlm"``, ``"smolvlm2"``, ``"granite"``,
    ``"granite4"``, or ``"generic"``.
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
    pil_image: Any = None,
) -> list[dict[str, Any]]:
    """Return a chat-message list appropriate for *model_family*.

    SmolVLM / SmolVLM2 / Idefics3 / Gemma 4 / generic
        Uses ``{"type": "image"}`` as a placeholder; the PIL image is passed
        separately to ``processor(images=[...])``.

    Granite 3.2 / Granite 4.1 / Gemma 3
        Embeds the PIL image directly in the message content as
        ``{"type": "image", "image": pil_image}`` so that
        ``processor.apply_chat_template`` can resolve it.

    Granite (path-based fallback)
        When *pil_image* is ``None`` for the granite family, embeds the image
        URL/path as ``{"type": "image", "url": str(image_path)}``.

    Parameters
    ----------
    image_path:
        Path to the image file.  Required for Granite model families when
        *pil_image* is not provided; ignored for ``"smolvlm"``.
    model_family:
        One of ``"smolvlm"``, ``"smolvlm2"``, ``"granite"``, ``"granite4"``,
        ``"gemma3"``, ``"gemma4"``, or ``"generic"``.
    pil_image:
        Pre-loaded ``PIL.Image.Image``.  Required for the ``"gemma3"`` family;
        used instead of the URL for Granite families if provided.
    """
    if model_family == "gemma3":
        image_item: dict[str, Any] = {"type": "image", "image": pil_image}
        return [
            {
                "role": "user",
                "content": [
                    image_item,
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            },
        ]
    if model_family.startswith("granite"):
        if pil_image is not None:
            image_item = {"type": "image", "image": pil_image}
        else:
            image_item = {
                "type": "image",
                "url": str(image_path) if image_path is not None else "",
            }
        return [
            {
                "role": "user",
                "content": [
                    image_item,
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            },
        ]
    # SmolVLM / Gemma 4 / generic: placeholder only; PIL image passed separately
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
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


def _local_hf_home() -> Path | None:
    """Return a writable local HF_HOME path, creating it if necessary.

    When HF_HOME points to a blob-storage mount (slow random reads), we do NOT
    copy the whole cache locally — that would duplicate every previously-cached
    model and quickly exhaust local disk.  Instead we return a fresh local
    directory and rely on ``_resolve_model_path`` to copy only the one model
    snapshot that is actually needed for this job.

    Returns None if HF_HOME is not set (HuggingFace uses its default cache).
    """
    import os

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return None

    local = Path("/tmp/hf_cache")
    if not local.exists():
        local.mkdir(parents=True, exist_ok=True)
        print(f"[cache] Created local HF cache dir at {local}", flush=True)
    else:
        print(f"[cache] Reusing local HF cache dir at {local}", flush=True)
    return local


def _resolve_model_path(model_name: str) -> str:
    """Return the local snapshot path for *model_name*, copying from blob store if needed.

    Strategy (avoids copying the entire HF cache to local disk):

    1. Check whether the model is already in the *local* HF cache
       (``/tmp/hf_cache``).  If yes, return that path directly.
    2. If not cached locally, check whether it is in the *remote* blob-store
       cache (the original ``HF_HOME``).  If yes, copy only that model's
       snapshot directory to the local cache, then return the local path.
    3. If not cached anywhere, return *model_name* so that transformers
       downloads it from HuggingFace (and writes to the local cache).
    """
    import os
    import shutil
    import time

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    # Step 1: already in local cache?
    try:
        local_path = snapshot_download(model_name, local_files_only=True)
        size_mb = (
            sum(f.stat().st_size for f in Path(local_path).rglob("*") if f.is_file())
            / 1e6
        )
        print(
            f"[cache] CACHE HIT (local) — {model_name}\n"
            f"        {local_path}  ({size_mb:.0f} MB)",
            flush=True,
        )
        return local_path
    except LocalEntryNotFoundError:
        pass

    # Step 2: in the remote blob-store cache?
    original_hf_home = os.environ.get("_ORIGINAL_HF_HOME")
    if original_hf_home:
        try:
            remote_path = snapshot_download(
                model_name,
                local_files_only=True,
                cache_dir=original_hf_home,
            )
            # Validate the snapshot is complete before using it.  A previous job
            # that crashed mid-download can leave a partial snapshot directory that
            # causes confusing errors later.
            snapshot_p = Path(remote_path)
            broken = [
                p for p in snapshot_p.rglob("*") if p.is_symlink() and not p.exists()
            ]
            if not (snapshot_p / "config.json").exists() or broken:
                print(
                    f"[cache] WARNING — blob snapshot for {model_name} is incomplete "
                    f"({len(broken)} broken symlink(s)); treating as a cache miss "
                    f"and re-downloading.",
                    flush=True,
                )
                raise LocalEntryNotFoundError(remote_path)
            # Copy only this model's snapshot to local cache.
            local_cache_dir = Path("/tmp/hf_cache")
            # Preserve the relative sub-path (hub/models--org--name/snapshots/...)
            rel = Path(remote_path).relative_to(Path(original_hf_home))
            local_dest = local_cache_dir / rel
            if not local_dest.exists():
                print(
                    f"[cache] CACHE HIT (blob) — copying {model_name} snapshot "
                    f"to local disk…",
                    flush=True,
                )
                t0 = time.monotonic()
                try:
                    shutil.copytree(remote_path, local_dest)
                except Exception as exc:
                    # Partial/corrupt snapshot on the blob store: some blob files
                    # referenced by symlinks in the snapshot dir are missing.
                    # Clean up the incomplete local copy and fall through to a
                    # fresh download from HuggingFace.
                    print(
                        f"[cache] WARNING — blob snapshot copy failed ({exc}); "
                        f"discarding partial copy and re-downloading.",
                        flush=True,
                    )
                    if local_dest.exists():
                        shutil.rmtree(local_dest, ignore_errors=True)
                    raise LocalEntryNotFoundError(remote_path) from exc
                elapsed = time.monotonic() - t0
                size_mb = (
                    sum(f.stat().st_size for f in local_dest.rglob("*") if f.is_file())
                    / 1e6
                )
                print(
                    f"[cache] Copied {size_mb:.0f} MB in {elapsed:.1f}s → {local_dest}",
                    flush=True,
                )
            else:
                print(
                    f"[cache] CACHE HIT (local, already copied) — {model_name}",
                    flush=True,
                )
            return str(local_dest)
        except LocalEntryNotFoundError:
            pass

    # Step 3: not cached anywhere — download from HuggingFace.
    print(
        f"[cache] CACHE MISS — {model_name} not cached; downloading from HuggingFace.",
        flush=True,
    )
    return model_name


def _uses_causal_lm(family: str) -> bool:
    """Return True for families that require ``AutoModelForCausalLM``.

    Gemma 4 uses ``AutoModelForCausalLM`` rather than
    ``AutoModelForImageTextToText``.
    """
    return family == "gemma4"


def _load_model_and_processor(config: ModelConfig):  # type: ignore[return]
    """Load the processor and model from HuggingFace (or a local LoRA adapter).

    If *config.model_name* points to a local directory that contains
    ``adapter_config.json``, the base model is read from the adapter config
    and the LoRA weights are applied via ``peft.PeftModel``.

    Raises ``ImportError`` if the ``train`` extras are not installed.
    """
    import os

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
        )
    except ImportError as exc:
        raise ImportError(
            "Install the 'train' extras to run inference: " "pip install -e '.[train]'"
        ) from exc

    _log_device_info()

    # Point HF_HOME at a local writable directory to avoid slow blob-mount reads.
    # Save the original blob-store path as _ORIGINAL_HF_HOME so _resolve_model_path
    # can copy only the required model snapshot across (not the whole cache).
    original_hf_home = os.environ.get("HF_HOME")
    local_cache = _local_hf_home()
    if local_cache is not None:
        os.environ["HF_HOME"] = str(local_cache)
        # huggingface_hub reads HF_HOME once at import time into the module-level
        # constant HUGGINGFACE_HUB_CACHE.  Setting os.environ afterwards has no
        # effect on that constant.  Patch it directly so snapshot_download and
        # from_pretrained use the local path even if the library was imported
        # before this function ran.
        try:
            import huggingface_hub.constants as _hfc

            _hfc.HF_HOME = str(local_cache)
            _hfc.HUGGINGFACE_HUB_CACHE = str(local_cache / "hub")
            _hfc.HF_HUB_CACHE = str(local_cache / "hub")
        except Exception:
            pass
        if original_hf_home and Path(original_hf_home) != local_cache:
            os.environ["_ORIGINAL_HF_HOME"] = original_hf_home
        print(f"[cache] HF_HOME set to local cache: {local_cache}", flush=True)

    # Cache dir to pass explicitly to from_pretrained — bypasses any stale
    # module-level constant that slipped through the patch above.
    hf_cache_dir: str | None = str(local_cache / "hub") if local_cache else None

    family = detect_model_family(config.model_name)
    use_causal = _uses_causal_lm(family)
    model_cls = AutoModelForCausalLM if use_causal else AutoModelForImageTextToText

    extra_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if hf_cache_dir:
        extra_kwargs["cache_dir"] = hf_cache_dir

    model_device_map: str | None = config.device

    if _is_adapter_path(config.model_name):
        import json as _json

        from peft import PeftModel

        adapter_dir = Path(config.model_name)
        adapter_cfg = _json.loads((adapter_dir / "adapter_config.json").read_text())
        base_model_name = adapter_cfg["base_model_name_or_path"]
        resolved_name = _resolve_model_path(base_model_name)

        proc_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if hf_cache_dir:
            proc_kwargs["cache_dir"] = hf_cache_dir
        processor = AutoProcessor.from_pretrained(resolved_name, **proc_kwargs)
        base = model_cls.from_pretrained(
            resolved_name,
            torch_dtype=_gpu_dtype(),
            device_map=model_device_map,
            **extra_kwargs,
        )
        model = PeftModel.from_pretrained(base, str(adapter_dir))
    else:
        resolved_name = _resolve_model_path(config.model_name)
        proc_kwargs = {"trust_remote_code": True}
        if hf_cache_dir:
            proc_kwargs["cache_dir"] = hf_cache_dir
        processor = AutoProcessor.from_pretrained(resolved_name, **proc_kwargs)
        model = model_cls.from_pretrained(
            resolved_name,
            torch_dtype=_gpu_dtype(),
            device_map=model_device_map,
            **extra_kwargs,
        )

    model.eval()

    # Write any newly-downloaded files back to the persistent blob-store cache.
    _sync_cache_to_remote(local_cache)

    return processor, model


def _sync_cache_to_remote(local_cache: "Path | None") -> None:
    """Copy new files from *local_cache* back to the original HF_HOME mount.

    Only runs if HF_HOME was originally a different path (i.e. a blob mount).
    Skips files that already exist at the destination to avoid redundant I/O.
    """
    import os
    import shutil

    original_hf_home = os.environ.get("_ORIGINAL_HF_HOME")
    if local_cache is None or not original_hf_home:
        return

    dst = Path(original_hf_home)
    new_files = 0
    for src_file in local_cache.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(local_cache)
        dst_file = dst / rel
        if not dst_file.exists():
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            new_files += 1
    if new_files:
        print(f"[cache] Wrote {new_files} new file(s) back to {dst}", flush=True)
    else:
        print("[cache] No new files to sync back to remote cache.", flush=True)


def extract_grid_batch_with_model(
    image_paths: list[Path],
    config: ModelConfig,
    processor: Any,
    model: Any,
    family: str,
) -> list[tuple[DailyRainfallGrid | None, str]]:
    """Run inference on a batch of images in a single forward pass.

    Processing images together makes better use of GPU parallelism compared
    to calling :func:`extract_grid_with_model` in a loop.

    Parameters
    ----------
    image_paths:
        List of paths to document images.  All are processed together.
    config:
        Model generation parameters.
    processor:
        HuggingFace processor already loaded for this model.
    model:
        HuggingFace model already loaded and on the target device.
    family:
        Model family string from :func:`detect_model_family`.

    Returns
    -------
    list of (grid, raw_text) pairs, one per input image, in order.
    """
    try:
        import torch
        from PIL import Image as PILImage
    except ImportError as exc:
        raise ImportError(
            "Install the 'train' extras to run inference: pip install -e '.[train]'"
        ) from exc

    images = [PILImage.open(p).convert("RGB") for p in image_paths]

    do_sample = config.temperature > 0.0
    generate_kwargs: dict[str, Any] = dict(
        max_new_tokens=config.max_new_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        generate_kwargs["temperature"] = config.temperature

    if family.startswith("granite") or family in ("gemma3", "gemma4"):
        # These families require per-image processing: Granite and Gemma 3 embed
        # the image in the message, Gemma 4 has a processor API that is
        # not straightforwardly batched.
        return [
            extract_grid_with_model(p, config, processor, model, family)
            for p in image_paths
        ]

    # SmolVLM / generic: build one prompt per image, batch-tokenise with padding
    text_prompts = [
        processor.apply_chat_template(
            build_messages(p, model_family=family),
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in image_paths
    ]
    inputs = processor(
        text=text_prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)

    # Inputs are left-padded to a uniform length; strip that prefix from outputs
    input_len = inputs["input_ids"].shape[1]
    results: list[tuple[DailyRainfallGrid | None, str]] = []
    for i in range(len(image_paths)):
        raw_text: str = processor.decode(
            output_ids[i, input_len:],
            skip_special_tokens=True,
        )
        results.append((parse_extraction_response(raw_text), raw_text))
    return results


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
    messages = build_messages(image_path, model_family=family, pil_image=image)

    do_sample = config.temperature > 0.0
    generate_kwargs: dict[str, Any] = dict(
        max_new_tokens=config.max_new_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        generate_kwargs["temperature"] = config.temperature

    if family.startswith("granite") or family == "gemma3":
        # These families embed the image directly in the message; the processor
        # tokenises everything in one apply_chat_template call.
        # Gemma 3: pass do_pan_and_scan=True so the processor tiles wide/tall
        # scans into patches, preventing detail loss at fixed 896×896 encoding.
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if family == "gemma3":
            template_kwargs["do_pan_and_scan"] = True
        inputs = processor.apply_chat_template(messages, **template_kwargs).to(
            model.device
        )
    elif family == "gemma4":
        # Gemma 4: two-step — tokenise text, then pass PIL image separately.
        # disable_thinking avoids outputting reasoning tokens.
        text_prompt: str = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        # SmolVLM / generic
        text_prompt = processor.apply_chat_template(
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
