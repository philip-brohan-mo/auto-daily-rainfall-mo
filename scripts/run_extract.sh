#!/usr/bin/env bash
# run_extract.sh — entrypoint called by azureml/extract_job.yml
# Environment variables set by the AML job:
#   WEATHER_IMAGES_DIR, SHARD, TOTAL_SHARDS, EXTRACT_LIMIT (optional)
set -euo pipefail

LIMIT_FLAG=""
if [[ -n "${EXTRACT_LIMIT:-}" ]]; then
    LIMIT_FLAG="--limit $EXTRACT_LIMIT"
fi

echo "[run_extract] shard=$SHARD/$TOTAL_SHARDS images=$WEATHER_IMAGES_DIR output=$OUTPUT_DIR ${LIMIT_FLAG:+limit=$EXTRACT_LIMIT}"
python -c "import torch, transformers; print(f'[env] torch={torch.__version__} transformers={transformers.__version__} cuda={torch.version.cuda}')"

python -m weather_doc_extractor.cli batch-extract \
    --shard "$SHARD" \
    --total-shards "$TOTAL_SHARDS" \
    --output-dir "$OUTPUT_DIR" \
    $LIMIT_FLAG
