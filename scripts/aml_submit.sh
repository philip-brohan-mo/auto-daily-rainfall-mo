#!/usr/bin/env bash
# aml_submit.sh — Submit weather-extract jobs to an Azure ML workspace.
#
# ── Configuration file ────────────────────────────────────────────────────────
#   Copy azureml/config.env.example to azureml/config.env and fill in your
#   workspace coordinates and data paths.  The script sources it automatically.
#   CLI flags and environment variables override any value in config.env.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash scripts/aml_submit.sh {extract|evaluate|finetune|env} [options]
#
# ── Options ───────────────────────────────────────────────────────────────────
#   --subscription SUB      Azure subscription ID or name
#   --resource-group RG     Azure resource group containing the workspace
#   --workspace WS          Azure ML workspace name
#   --compute CLUSTER       Compute cluster name (default: gpu-cluster)
#   --total-shards N        Parallel shards for extract/evaluate (default: 8)
#   --limit N               Process only N images per shard (for smoke tests);
#                           also defaults --total-shards to 1
#   --help                  Show this message
#
# ── Examples ──────────────────────────────────────────────────────────────────
#   # First-time setup: copy and edit the config, then register the environment:
#   cp azureml/config.env.example azureml/config.env
#   # edit azureml/config.env
#   bash scripts/aml_submit.sh env
#
#   # Submit 8 extract shards (uses azureml/config.env):
#   bash scripts/aml_submit.sh --total-shards 8 extract
#
#   # Quick smoke test — 1 shard, 5 images:
#   bash scripts/aml_submit.sh --limit 5 extract
#
#   # Override workspace on the command line:
#   bash scripts/aml_submit.sh --workspace other-ws evaluate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$REPO_DIR/azureml/config.env"

# ── Load config.env (if present) before applying CLI overrides ───────────────
if [[ -f "$CONFIG_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$CONFIG_FILE"
fi

# Apply defaults *after* sourcing so config.env values become the baseline.
COMMAND=""
TOTAL_SHARDS=8
EXTRACT_LIMIT=""
WEATHER_MODEL="${WEATHER_MODEL:-smolvlm}"
AML_COMPUTE="${AML_COMPUTE:-gpu-cluster}"
AML_SUBSCRIPTION="${AML_SUBSCRIPTION:-}"
AML_RESOURCE_GROUP="${AML_RESOURCE_GROUP:-}"
AML_WORKSPACE="${AML_WORKSPACE:-}"
AML_DATASTORE_BASE="${AML_DATASTORE_BASE:-azureml://datastores/workspaceblobstore/paths}"
AML_IMAGES_PATH="${AML_IMAGES_PATH:-Daily_rainfall_sample/images}"
AML_TRANSCRIPTIONS_PATH="${AML_TRANSCRIPTIONS_PATH:-Daily_rainfall_sample/transcriptions}"
AML_OUTPUTS_PATH="${AML_OUTPUTS_PATH:-outputs}"

usage() {
    sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subscription)    AML_SUBSCRIPTION="$2";    shift 2 ;;
        --resource-group)  AML_RESOURCE_GROUP="$2";  shift 2 ;;
        --workspace)       AML_WORKSPACE="$2";        shift 2 ;;
        --total-shards)    TOTAL_SHARDS="$2";         shift 2 ;;
        --limit)           EXTRACT_LIMIT="$2";        shift 2 ;;
        --model)           WEATHER_MODEL="$2";        shift 2 ;;
        --compute)         AML_COMPUTE="$2";          shift 2 ;;
        --help|-h)         usage 0 ;;
        extract|evaluate|finetune|env) COMMAND="$1"; shift ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

[[ -z "$COMMAND" ]] && { echo "Error: command required (extract|evaluate|finetune|env)" >&2; usage 1; }
[[ -z "$AML_SUBSCRIPTION" ]]   && { echo "Error: AML_SUBSCRIPTION not set (use --subscription or azureml/config.env)"   >&2; exit 1; }
[[ -z "$AML_RESOURCE_GROUP" ]] && { echo "Error: AML_RESOURCE_GROUP not set (use --resource-group or azureml/config.env)" >&2; exit 1; }
[[ -z "$AML_WORKSPACE" ]]      && { echo "Error: AML_WORKSPACE not set (use --workspace or azureml/config.env)"           >&2; exit 1; }

# When --limit is set and --total-shards was not explicitly specified, default to 1 shard.
[[ -n "$EXTRACT_LIMIT" && "$TOTAL_SHARDS" -eq 8 ]] && TOTAL_SHARDS=1

# Output subdirectory: extractions/<model-slug>/<YYYYMMDD-HHMMSS>
# This ensures runs from different models and repeated runs never overwrite each other.
MODEL_SLUG="${WEATHER_MODEL##*/}"
RUN_TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"

# Resolved datastore URIs
IMAGES_URI="$AML_DATASTORE_BASE/$AML_IMAGES_PATH"
TRANSCRIPTIONS_URI="$AML_DATASTORE_BASE/$AML_TRANSCRIPTIONS_PATH"
OUTPUTS_URI="$AML_DATASTORE_BASE/$AML_OUTPUTS_PATH"
EXTRACTIONS_URI="$OUTPUTS_URI/extractions/$MODEL_SLUG/$RUN_TIMESTAMP"

AML_ARGS=(
    --workspace-name "$AML_WORKSPACE"
    --resource-group "$AML_RESOURCE_GROUP"
    --subscription   "$AML_SUBSCRIPTION"
)

echo "Workspace:  $AML_WORKSPACE  ($AML_RESOURCE_GROUP / $AML_SUBSCRIPTION)"
echo "Compute:    $AML_COMPUTE"
echo "Model:      $WEATHER_MODEL"
echo "Images:     $IMAGES_URI"
[[ "$COMMAND" != "extract" ]] && echo "Transcript: $TRANSCRIPTIONS_URI"
echo "Outputs:    $OUTPUTS_URI"
echo

case "$COMMAND" in
    env)
        echo "Registering Azure ML environment in workspace '$AML_WORKSPACE'..."
        az ml environment create \
            --file "$REPO_DIR/azureml/environment.yml" \
            "${AML_ARGS[@]}"
        echo "Environment registered."
        ;;

    extract)
        echo "Submitting $TOTAL_SHARDS extract shard(s)${EXTRACT_LIMIT:+ (limit: $EXTRACT_LIMIT images each)}..."
        echo "  Output: $EXTRACTIONS_URI"
        for i in $(seq 1 "$TOTAL_SHARDS"); do
            echo "  Shard $i / $TOTAL_SHARDS ..."
            az ml job create \
                --file "$REPO_DIR/azureml/extract_job.yml" \
                "${AML_ARGS[@]}" \
                --set compute="azureml:$AML_COMPUTE" \
                --set inputs.images_dir.path="$IMAGES_URI" \
                --set outputs.extractions.path="$EXTRACTIONS_URI" \
                --set environment_variables.SHARD="$i" \
                --set environment_variables.TOTAL_SHARDS="$TOTAL_SHARDS" \
                --set environment_variables.WEATHER_MODEL="$WEATHER_MODEL" \
                ${EXTRACT_LIMIT:+--set environment_variables.EXTRACT_LIMIT="$EXTRACT_LIMIT"} \
                --set display_name="batch-extract-${MODEL_SLUG}-${i}-of-${TOTAL_SHARDS}" \
                --query name --output tsv
        done
        echo "Submitted $TOTAL_SHARDS extract job(s)."
        ;;

    evaluate)
        echo "Submitting $TOTAL_SHARDS evaluate shards..."
        for i in $(seq 1 "$TOTAL_SHARDS"); do
            echo "  Shard $i / $TOTAL_SHARDS ..."
            az ml job create \
                --file "$REPO_DIR/azureml/evaluate_job.yml" \
                "${AML_ARGS[@]}" \
                --set compute="azureml:$AML_COMPUTE" \
                --set inputs.images_dir.path="$IMAGES_URI" \
                --set inputs.transcriptions_dir.path="$TRANSCRIPTIONS_URI" \
                --set outputs.results.path="$OUTPUTS_URI/eval" \
                --set environment_variables.SHARD="$i" \
                --set environment_variables.TOTAL_SHARDS="$TOTAL_SHARDS" \
                --set display_name="evaluate-${i}-of-${TOTAL_SHARDS}" \
                --query name --output tsv
        done
        echo "Submitted $TOTAL_SHARDS evaluate jobs."
        ;;

    finetune)
        echo "Submitting finetune job..."
        az ml job create \
            --file "$REPO_DIR/azureml/finetune_job.yml" \
            "${AML_ARGS[@]}" \
            --set compute="azureml:$AML_COMPUTE" \
            --set inputs.images_dir.path="$IMAGES_URI" \
            --set inputs.transcriptions_dir.path="$TRANSCRIPTIONS_URI" \
            --set outputs.checkpoints.path="$OUTPUTS_URI/checkpoints" \
            --query name --output tsv
        echo "Finetune job submitted."
        ;;
esac
