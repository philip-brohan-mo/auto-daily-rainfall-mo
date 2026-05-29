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
#   --model MODEL           Model preset (smolvlm, smolvlm2, granite, granite4) or HF ID
#   --env-variant v100|a100 Azure ML environment to use (default: v100).
#                           v100 uses weather-doc-extractor  (PyTorch 2.4, CUDA 12.1)
#                           a100 uses weather-doc-extractor-a100 (PyTorch 2.8, CUDA 12.6)
#                           Auto-selected for granite4, gemma3, gemma4, ministral which need torch>=2.6.
#   --dataset real|fake|test_real|test_fake
#                           Quick dataset selector:
#                             real      → Daily_rainfall_sample
#                             fake      → fake_daily_rainfall
#                             test_real → test_data/real  (committed test split)
#                             test_fake → test_data/fake  (committed test split)
#   --images-path PATH      Override AML_IMAGES_PATH
#   --transcriptions-path PATH  Override AML_TRANSCRIPTIONS_PATH
#   --checkpoint PATH       Use a saved fine-tuned checkpoint for extraction
#                           (overrides --model); path relative to AML_DATASTORE_BASE
#   --clear-modules         Delete stale HF transformers_modules cache before running.
#                           Use after upgrading transformers to fix custom-model errors.
#   --help                  Show this message
#
# ── Examples ──────────────────────────────────────────────────────────────────
#   # First-time setup: copy and edit the config, then register the environment:
#   cp azureml/config.env.example azureml/config.env
#   # edit azureml/config.env
#   bash scripts/aml_submit.sh env
#
#   # Extract from real data (default, 8 shards):
#   bash scripts/aml_submit.sh --total-shards 8 extract
#
#   # Extract from fake data (quick test):
#   bash scripts/aml_submit.sh --dataset fake --limit 10 extract
#
#   # Extract from fake data for full fine-tuning:
#   bash scripts/aml_submit.sh --dataset fake --model granite4 finetune
#
#   # Extract using a fine-tuned checkpoint:
#   bash scripts/aml_submit.sh --checkpoint outputs/checkpoints/granite-fake-20260526-143000 --limit 50 extract
#
#   # Quick smoke test — real data, 1 shard, 5 images:
#   bash scripts/aml_submit.sh --limit 5 extract

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
TOTAL_SHARDS="${TOTAL_SHARDS:-1}"
EXTRACT_LIMIT=""
BATCH_SIZE="${BATCH_SIZE:-1}"
WEATHER_MODEL="${WEATHER_MODEL:-smolvlm}"
WEATHER_CHECKPOINT=""
CLEAR_HF_MODULES="0"
HF_TOKEN="${HF_TOKEN:-}"
AML_ENV_VARIANT="${AML_ENV_VARIANT:-v100}"
AML_COMPUTE="${AML_COMPUTE:-gpu-cluster}"
AML_SUBSCRIPTION="${AML_SUBSCRIPTION:-}"
AML_RESOURCE_GROUP="${AML_RESOURCE_GROUP:-}"
AML_WORKSPACE="${AML_WORKSPACE:-}"
AML_DATASTORE_BASE="${AML_DATASTORE_BASE:-azureml://datastores/workspaceblobstore/paths}"
AML_IMAGES_PATH="${AML_IMAGES_PATH:-Daily_rainfall_sample/images}"
AML_TRANSCRIPTIONS_PATH="${AML_TRANSCRIPTIONS_PATH:-Daily_rainfall_sample/transcriptions}"
AML_OUTPUTS_PATH="${AML_OUTPUTS_PATH:-outputs}"
AML_HF_CACHE_PATH="${AML_HF_CACHE_PATH:-hf_cache}"

usage() {
    sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subscription)      AML_SUBSCRIPTION="$2";    shift 2 ;;
        --resource-group)    AML_RESOURCE_GROUP="$2";  shift 2 ;;
        --workspace)         AML_WORKSPACE="$2";        shift 2 ;;
        --total-shards)      TOTAL_SHARDS="$2";         shift 2 ;;
        --limit)             EXTRACT_LIMIT="$2";        shift 2 ;;
        --batch-size)        BATCH_SIZE="$2";            shift 2 ;;
        --model)             WEATHER_MODEL="$2";        shift 2 ;;
        --compute)           AML_COMPUTE="$2";          shift 2 ;;
        --env-variant)       AML_ENV_VARIANT="$2";      shift 2 ;;
        --dataset)
            case "$2" in
                real)      AML_IMAGES_PATH="Daily_rainfall_sample/images"; AML_TRANSCRIPTIONS_PATH="Daily_rainfall_sample/transcriptions" ;;
                fake)      AML_IMAGES_PATH="fake_daily_rainfall/images"; AML_TRANSCRIPTIONS_PATH="fake_daily_rainfall/transcriptions" ;;
                test_real) AML_IMAGES_PATH="test_data/real/images"; AML_TRANSCRIPTIONS_PATH="test_data/real/transcriptions" ;;
                test_fake) AML_IMAGES_PATH="test_data/fake/images"; AML_TRANSCRIPTIONS_PATH="test_data/fake/transcriptions" ;;
                *) echo "Unknown dataset: $2 (use 'real', 'fake', 'test_real', or 'test_fake')" >&2; exit 1 ;;
            esac
            shift 2 ;;
        --images-path)       AML_IMAGES_PATH="$2";       shift 2 ;;
        --transcriptions-path) AML_TRANSCRIPTIONS_PATH="$2"; shift 2 ;;
        --checkpoint)        WEATHER_CHECKPOINT="$2";    shift 2 ;;
        --clear-modules)     CLEAR_HF_MODULES="1";         shift ;;
        --help|-h)           usage 0 ;;
        extract|evaluate|finetune|env) COMMAND="$1"; shift ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

[[ -z "$COMMAND" ]] && { echo "Error: command required (extract|evaluate|finetune|env)" >&2; usage 1; }
[[ -z "$AML_SUBSCRIPTION" ]]   && { echo "Error: AML_SUBSCRIPTION not set (use --subscription or azureml/config.env)"   >&2; exit 1; }
[[ -z "$AML_RESOURCE_GROUP" ]] && { echo "Error: AML_RESOURCE_GROUP not set (use --resource-group or azureml/config.env)" >&2; exit 1; }
[[ -z "$AML_WORKSPACE" ]]      && { echo "Error: AML_WORKSPACE not set (use --workspace or azureml/config.env)"           >&2; exit 1; }

# Select environment file and registered name based on variant.
# Auto-upgrade to a100 if the model requires torch >= 2.6 (v100 has 2.4)
case "$WEATHER_MODEL" in
    granite4|granite-vision-4.1*|gemma3|gemma4|ministral|mistral*)
        if [[ "$AML_ENV_VARIANT" == "v100" ]]; then
            echo "⚠️  Model $WEATHER_MODEL requires torch >= 2.6. Auto-switching to a100 environment."
            AML_ENV_VARIANT="a100"
        fi
        ;;
esac

if [[ "$AML_ENV_VARIANT" == "a100" ]]; then
    AML_ENV_FILE="$REPO_DIR/azureml/environment-a100.yml"
    AML_ENV_OVERRIDE="--set environment=azureml:weather-doc-extractor-a100@latest"
else
    AML_ENV_FILE="$REPO_DIR/azureml/environment.yml"
    AML_ENV_OVERRIDE=""
fi

# When --limit is set and --total-shards was not explicitly specified, default to 1 shard.
[[ -n "$EXTRACT_LIMIT" && "$TOTAL_SHARDS" -eq 8 ]] && TOTAL_SHARDS=1

# Output subdirectory: extractions/<model-slug>/<YYYYMMDD-HHMMSS>
# This ensures runs from different models and repeated runs never overwrite each other.
# If checkpoint is specified, use it for extraction; otherwise use the model preset.
if [[ -n "$WEATHER_CHECKPOINT" ]]; then
    # Checkpoint overrides model; extract the checkpoint name for the output directory
    CHECKPOINT_NAME="${WEATHER_CHECKPOINT##*/}"
    CHECKPOINT_URI="$AML_DATASTORE_BASE/$WEATHER_CHECKPOINT"
    MODEL_SLUG="$CHECKPOINT_NAME"
else
    CHECKPOINT_URI=""
    MODEL_SLUG="${WEATHER_MODEL##*/}"
fi
RUN_TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"

# Resolved datastore URIs
IMAGES_URI="$AML_DATASTORE_BASE/$AML_IMAGES_PATH"
TRANSCRIPTIONS_URI="$AML_DATASTORE_BASE/$AML_TRANSCRIPTIONS_PATH"
OUTPUTS_URI="$AML_DATASTORE_BASE/$AML_OUTPUTS_PATH"
EXTRACTIONS_URI="$OUTPUTS_URI/extractions/$MODEL_SLUG/$RUN_TIMESTAMP"
HF_CACHE_URI="$AML_DATASTORE_BASE/$AML_HF_CACHE_PATH"

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
        # Auto-increment the version so re-registration never fails on duplicate.
        ENV_NAME=$(grep '^name:' "$AML_ENV_FILE" | awk '{print $2}')
        CURRENT_VERSION=$(az ml environment list \
            --name "$ENV_NAME" "${AML_ARGS[@]}" \
            --query "[].version" --output tsv 2>/dev/null \
            | sort -n | tail -1) || true
        [[ -z "$CURRENT_VERSION" ]] && CURRENT_VERSION=0
        NEXT_VERSION=$(( CURRENT_VERSION + 1 ))
        echo "Registering environment '$ENV_NAME' version $NEXT_VERSION (variant: $AML_ENV_VARIANT)..."
        PATCHED_YML="$REPO_DIR/azureml/.environment_tmp.yml"
        sed "s/^version:.*/version: $NEXT_VERSION/" "$AML_ENV_FILE" > "$PATCHED_YML"
        az ml environment create --file "$PATCHED_YML" "${AML_ARGS[@]}"
        rm -f "$PATCHED_YML"
        sed -i "s/^version:.*/version: $NEXT_VERSION/" "$AML_ENV_FILE"
        echo "Environment registered as version $NEXT_VERSION."
        echo "$AML_ENV_FILE updated — commit the version bump if desired."
        ;;

    extract)
        echo "Submitting $TOTAL_SHARDS extract shard(s)${EXTRACT_LIMIT:+ (limit: $EXTRACT_LIMIT images each)}..."
        [[ -n "$WEATHER_CHECKPOINT" ]] && echo "  Checkpoint: $CHECKPOINT_URI"
        echo "  Output:   $EXTRACTIONS_URI"
        echo "  HF cache: $HF_CACHE_URI"
        for i in $(seq 1 "$TOTAL_SHARDS"); do
            echo "  Shard $i / $TOTAL_SHARDS ..."
            az ml job create \
                --file "$REPO_DIR/azureml/extract_job.yml" \
                "${AML_ARGS[@]}" \
                --set compute="azureml:$AML_COMPUTE" \
                ${AML_ENV_OVERRIDE:+$AML_ENV_OVERRIDE} \
                --set inputs.images_dir.path="$IMAGES_URI" \
                --set outputs.extractions.path="$EXTRACTIONS_URI" \
                --set outputs.hf_cache.path="$HF_CACHE_URI" \
                --set environment_variables.SHARD="$i" \
                --set environment_variables.TOTAL_SHARDS="$TOTAL_SHARDS" \
                --set environment_variables.WEATHER_MODEL="$WEATHER_MODEL" \
                --set environment_variables.BATCH_SIZE="$BATCH_SIZE" \
                ${CHECKPOINT_URI:+--set environment_variables.WEATHER_CHECKPOINT="$CHECKPOINT_URI"} \
                ${EXTRACT_LIMIT:+--set environment_variables.EXTRACT_LIMIT="$EXTRACT_LIMIT"} \
                --set environment_variables.CLEAR_HF_MODULES="$CLEAR_HF_MODULES" \
                ${HF_TOKEN:+--set environment_variables.HF_TOKEN="$HF_TOKEN"} \
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
                ${AML_ENV_OVERRIDE:+$AML_ENV_OVERRIDE} \
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
        JOB_ID=$(az ml job create \
            --file "$REPO_DIR/azureml/finetune_job.yml" \
            "${AML_ARGS[@]}" \
            --set compute="azureml:$AML_COMPUTE" \
            ${AML_ENV_OVERRIDE:+$AML_ENV_OVERRIDE} \
            --set inputs.images_dir.path="$IMAGES_URI" \
            --set inputs.transcriptions_dir.path="$TRANSCRIPTIONS_URI" \
            --set outputs.checkpoints.path="$OUTPUTS_URI/checkpoints" \
            --set environment_variables.WEATHER_MODEL="$WEATHER_MODEL" \
            ${EXTRACT_LIMIT:+--set environment_variables.EXTRACT_LIMIT="$EXTRACT_LIMIT"} \
            ${HF_TOKEN:+--set environment_variables.HF_TOKEN="$HF_TOKEN"} \
            --set display_name="finetune-${MODEL_SLUG}" \
            --query name --output tsv)
        echo "Finetune job submitted: $JOB_ID"
        echo
        echo "After the job completes, register the checkpoint:"
        echo "  python scripts/create_model_registry_entry.py \\"
        echo "    --checkpoint-path outputs/checkpoints/CHECKPOINT_NAME \\"
        echo "    --base-model $WEATHER_MODEL \\"
        echo "    --dataset $AML_IMAGES_PATH \\"
        echo "    --registry-file outputs/model_registry.json"
        echo
        echo "Then extract using the checkpoint:"
        echo "  bash scripts/aml_submit.sh --checkpoint outputs/checkpoints/CHECKPOINT_NAME extract"
        ;;
esac
