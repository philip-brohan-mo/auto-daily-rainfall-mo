#!/usr/bin/env bash
# azure_validate_granite4.sh — Submit required Azure validation jobs for a VLM model.
#
# This script submits:
#  1) A smoke extraction run with the specified model
#  2) A smoke fine-tune run with the specified model
#
# It uses scripts/aml_submit.sh, so workspace settings are read from
# azureml/config.env (or CLI flags that override it).
#
# Usage:
#   bash scripts/azure_validate_granite4.sh [--model MODEL] [--limit N] [--compute CLUSTER] [--env-variant v100|a100]
#
# Example:
#   bash scripts/azure_validate_granite4.sh --model granite4 --limit 2 --env-variant a100
#   bash scripts/azure_validate_granite4.sh --model smolvlm2 --limit 2 --env-variant a100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

MODEL="granite4"
LIMIT=2
COMPUTE=""
ENV_VARIANT=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --compute)
            COMPUTE="$2"
            shift 2
            ;;
        --env-variant)
            ENV_VARIANT="$2"
            shift 2
            ;;
        --help|-h)
            sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

[[ -n "$COMPUTE" ]] && EXTRA_ARGS+=(--compute "$COMPUTE")
[[ -n "$ENV_VARIANT" ]] && EXTRA_ARGS+=(--env-variant "$ENV_VARIANT")

echo "[azure-validate] Submitting extraction smoke test for $MODEL"
bash "$REPO_DIR/scripts/aml_submit.sh" \
    --model "$MODEL" \
    --total-shards 1 \
    --limit "$LIMIT" \
    "${EXTRA_ARGS[@]}" \
    extract

echo
echo "[azure-validate] Submitting fine-tune smoke test for $MODEL"
bash "$REPO_DIR/scripts/aml_submit.sh" \
    --model "$MODEL" \
    --limit "$LIMIT" \
    "${EXTRA_ARGS[@]}" \
    finetune

echo
echo "[azure-validate] Submitted required $MODEL validation jobs."
echo "[azure-validate] Required follow-up checks:" 
echo "  1. Confirm extract job status is Completed and output JSON files parse." 
echo "  2. Confirm finetune job status is Completed and adapter checkpoints are written." 
echo "  3. Run post-finetune extraction with the produced adapter path as --model." 
echo "  4. Run one extraction smoke with --model granite to confirm 3.2 compatibility."
