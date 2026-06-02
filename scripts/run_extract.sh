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

# Use checkpoint only when adapter files exist at the mounted path.
# This avoids treating the default placeholder input mount as a real checkpoint.
if [[ -n "${WEATHER_CHECKPOINT:-}" ]] && [[ -f "$WEATHER_CHECKPOINT/adapter_config.json" ]]; then
    MODEL="$WEATHER_CHECKPOINT"
    echo "[run_extract] Using checkpoint: $WEATHER_CHECKPOINT" >&2
else
    MODEL="${WEATHER_MODEL:-smolvlm}"
    if [[ -n "${WEATHER_CHECKPOINT:-}" ]]; then
        echo "[run_extract] No adapter_config.json at checkpoint mount; using base model: $MODEL" >&2
    fi
fi
MODEL_FLAG="--model $MODEL"

echo "[run_extract] shard=$SHARD/$TOTAL_SHARDS images=$WEATHER_IMAGES_DIR output=$OUTPUT_DIR model=$MODEL ${LIMIT_FLAG:+limit=$EXTRACT_LIMIT} ${BATCH_SIZE_FLAG:+batch=$BATCH_SIZE}"
python -c "import torch, transformers; print(f'[env] torch={torch.__version__} transformers={transformers.__version__} cuda={torch.version.cuda}')"

python -m weather_doc_extractor.cli batch-extract \
    --shard "$SHARD" \
    --total-shards "$TOTAL_SHARDS" \
    --output-dir "$OUTPUT_DIR" \
    $MODEL_FLAG \
    $LIMIT_FLAG \
    $BATCH_SIZE_FLAG
