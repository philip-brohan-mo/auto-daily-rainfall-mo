#!/usr/bin/env bash
# run_extract.sh — entrypoint called by azureml/extract_job.yml
# Environment variables set by the AML job:
#   WEATHER_IMAGES_DIR, SHARD, TOTAL_SHARDS, EXTRACT_LIMIT (optional)
#   WEATHER_MODEL (default model), WEATHER_CHECKPOINT (optional — overrides WEATHER_MODEL)
#   CLEAR_HF_MODULES (optional — set to "1" to delete stale transformers_modules cache)
set -euo pipefail

# Clear stale HF custom model code if requested (fixes create_causal_mask errors after
# transformers version upgrades — the cached modules/ dir gets rebuilt on next download)
if [[ "${CLEAR_HF_MODULES:-0}" == "1" ]]; then
    # Clear both the modules cache and the granite model snapshots to force re-download
    MODULES_DIR="${HF_HOME:-}/modules/transformers_modules"
    GRANITE_DIR="${HF_HOME:-}/hub/models--ibm-granite--granite-vision-4.1-4b"
    
    if [[ -d "$MODULES_DIR" ]]; then
        echo "[cache] Clearing stale HF modules cache: $MODULES_DIR"
        rm -rf "$MODULES_DIR"
    fi
    if [[ -d "$GRANITE_DIR" ]]; then
        echo "[cache] Clearing Granite 4.1 model snapshots (will re-download): $GRANITE_DIR"
        rm -rf "$GRANITE_DIR"
    fi
    echo "[cache] Done — will re-download fresh model and custom code."
fi

LIMIT_FLAG=""
if [[ -n "${EXTRACT_LIMIT:-}" ]]; then
    LIMIT_FLAG="--limit $EXTRACT_LIMIT"
fi

BATCH_SIZE_FLAG=""
if [[ -n "${BATCH_SIZE:-}" ]]; then
    BATCH_SIZE_FLAG="--batch-size $BATCH_SIZE"
fi

RETRY_BATCH_SIZE_FLAG=""
if [[ -n "${RETRY_BATCH_SIZE:-}" ]]; then
    RETRY_BATCH_SIZE_FLAG="--retry-batch-size $RETRY_BATCH_SIZE"
fi

# Use checkpoint when adapter files exist at the mounted path (directly or one
# level deeper in a model-slug subdirectory).  Fail hard if a checkpoint is
# specified but no adapter_config.json can be found — never fall back silently.
MODEL=""
CHECKPOINT_REQUIRED="${WEATHER_CHECKPOINT_REQUIRED:-0}"
find_adapter_dir() {
    local root="$1"
    find "$root" -type f -name adapter_config.json 2>/dev/null | head -n1 | sed 's|/adapter_config.json$||'
}
if [[ -n "${WEATHER_CHECKPOINT:-}" ]] && [[ "$CHECKPOINT_REQUIRED" == "1" ]]; then
    echo "[run_extract] Searching for adapter in checkpoint: $WEATHER_CHECKPOINT" >&2
    echo "[run_extract] Contents (first 30 lines):" >&2
    find "$WEATHER_CHECKPOINT" -maxdepth 3 -type f -o -type d 2>/dev/null | head -30 >&2 || echo "[run_extract] ERROR: checkpoint path does not exist or not readable" >&2
    MODEL="$(find_adapter_dir "$WEATHER_CHECKPOINT")"
    if [[ -z "$MODEL" ]]; then
        echo "[run_extract] ERROR: WEATHER_CHECKPOINT is set but no adapter_config.json found at $WEATHER_CHECKPOINT (recursive search)." >&2
        exit 1
    fi
    echo "[run_extract] Using checkpoint: $MODEL" >&2
elif [[ -n "${WEATHER_CHECKPOINT:-}" ]] && [[ "$CHECKPOINT_REQUIRED" != "1" ]]; then
    echo "[run_extract] Ignoring checkpoint input mount (not explicitly requested)." >&2
    MODEL="$(find_adapter_dir "$WEATHER_CHECKPOINT")"
    if [[ -n "$MODEL" ]]; then
        echo "[run_extract] Found adapter in checkpoint mount; using: $MODEL" >&2
    fi
elif [[ -n "${WEATHER_MODEL:-}" ]]; then
    MODEL="$WEATHER_MODEL"
else
    echo "[run_extract] ERROR: neither WEATHER_CHECKPOINT nor WEATHER_MODEL is set — refusing to use a default model." >&2
    exit 1
fi
MODEL_FLAG="--model $MODEL"

echo "[run_extract] shard=$SHARD/$TOTAL_SHARDS images=$WEATHER_IMAGES_DIR output=$OUTPUT_DIR model=$MODEL ${LIMIT_FLAG:+limit=$EXTRACT_LIMIT} ${BATCH_SIZE_FLAG:+batch=$BATCH_SIZE} ${RETRY_BATCH_SIZE_FLAG:+retry_batch=$RETRY_BATCH_SIZE}"
python -c "import torch, transformers; print(f'[env] torch={torch.__version__} transformers={transformers.__version__} cuda={torch.version.cuda}')"

python -m weather_doc_extractor.cli batch-extract \
    --shard "$SHARD" \
    --total-shards "$TOTAL_SHARDS" \
    --output-dir "$OUTPUT_DIR" \
    $MODEL_FLAG \
    $LIMIT_FLAG \
    $BATCH_SIZE_FLAG \
    $RETRY_BATCH_SIZE_FLAG
