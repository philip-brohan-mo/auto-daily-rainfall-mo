#!/usr/bin/env bash
# aml_submit.sh — Submit weather-extract jobs to an Azure ML workspace.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash scripts/aml_submit.sh --subscription SUB --resource-group RG \
#                              --workspace WS {extract|evaluate|finetune}
#
# ── Required ──────────────────────────────────────────────────────────────────
#   --subscription SUB      Azure subscription ID or name
#   --resource-group RG     Azure resource group containing the workspace
#   --workspace WS          Azure ML workspace name
#
# ── Options ───────────────────────────────────────────────────────────────────
#   --total-shards N        Parallel shards for extract/evaluate (default: 8)
#   --compute CLUSTER       Compute cluster name (default: gpu-cluster)
#   --help                  Show this message
#
# ── Environment variable alternatives ─────────────────────────────────────────
#   Set AML_SUBSCRIPTION, AML_RESOURCE_GROUP, AML_WORKSPACE to avoid typing
#   them on every invocation.
#
# ── Examples ──────────────────────────────────────────────────────────────────
#   # Register the environment (once):
#   bash scripts/aml_submit.sh --subscription abc123 \
#       --resource-group my-rg --workspace my-ws env
#
#   # Submit 8 extract shards:
#   bash scripts/aml_submit.sh --subscription abc123 \
#       --resource-group my-rg --workspace my-ws \
#       --total-shards 8 extract
#
#   # Submit 8 evaluate shards:
#   bash scripts/aml_submit.sh --subscription abc123 \
#       --resource-group my-rg --workspace my-ws \
#       --total-shards 8 evaluate
#
#   # Submit a fine-tuning job:
#   bash scripts/aml_submit.sh --subscription abc123 \
#       --resource-group my-rg --workspace my-ws finetune

set -euo pipefail

COMMAND=""
TOTAL_SHARDS=8
AML_COMPUTE="${AML_COMPUTE:-gpu-cluster}"
AML_SUBSCRIPTION="${AML_SUBSCRIPTION:-}"
AML_RESOURCE_GROUP="${AML_RESOURCE_GROUP:-}"
AML_WORKSPACE="${AML_WORKSPACE:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

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
        --compute)         AML_COMPUTE="$2";          shift 2 ;;
        --help|-h)         usage 0 ;;
        extract|evaluate|finetune|env) COMMAND="$1"; shift ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

[[ -z "$COMMAND" ]] && { echo "Error: command required (extract|evaluate|finetune|env)" >&2; usage 1; }
[[ -z "$AML_SUBSCRIPTION" ]]   && { echo "Error: --subscription required (or set AML_SUBSCRIPTION)"   >&2; exit 1; }
[[ -z "$AML_RESOURCE_GROUP" ]] && { echo "Error: --resource-group required (or set AML_RESOURCE_GROUP)" >&2; exit 1; }
[[ -z "$AML_WORKSPACE" ]]      && { echo "Error: --workspace required (or set AML_WORKSPACE)"           >&2; exit 1; }

AML_ARGS=(
    --workspace-name "$AML_WORKSPACE"
    --resource-group "$AML_RESOURCE_GROUP"
    --subscription   "$AML_SUBSCRIPTION"
)

case "$COMMAND" in
    env)
        echo "Registering Azure ML environment in workspace '$AML_WORKSPACE'..."
        az ml environment create \
            --file "$REPO_DIR/azureml/environment.yml" \
            "${AML_ARGS[@]}"
        echo "Environment registered."
        ;;

    extract)
        echo "Submitting $TOTAL_SHARDS extract shards to workspace '$AML_WORKSPACE'..."
        for i in $(seq 1 "$TOTAL_SHARDS"); do
            echo "  Shard $i / $TOTAL_SHARDS ..."
            az ml job create \
                --file "$REPO_DIR/azureml/extract_job.yml" \
                "${AML_ARGS[@]}" \
                --set compute="azureml:$AML_COMPUTE" \
                --set environment_variables.SHARD="$i" \
                --set environment_variables.TOTAL_SHARDS="$TOTAL_SHARDS" \
                --set display_name="batch-extract-${i}-of-${TOTAL_SHARDS}" \
                --query name --output tsv
        done
        echo "Submitted $TOTAL_SHARDS extract jobs."
        ;;

    evaluate)
        echo "Submitting $TOTAL_SHARDS evaluate shards to workspace '$AML_WORKSPACE'..."
        for i in $(seq 1 "$TOTAL_SHARDS"); do
            echo "  Shard $i / $TOTAL_SHARDS ..."
            az ml job create \
                --file "$REPO_DIR/azureml/evaluate_job.yml" \
                "${AML_ARGS[@]}" \
                --set compute="azureml:$AML_COMPUTE" \
                --set environment_variables.SHARD="$i" \
                --set environment_variables.TOTAL_SHARDS="$TOTAL_SHARDS" \
                --set display_name="evaluate-${i}-of-${TOTAL_SHARDS}" \
                --query name --output tsv
        done
        echo "Submitted $TOTAL_SHARDS evaluate jobs."
        ;;

    finetune)
        echo "Submitting finetune job to workspace '$AML_WORKSPACE'..."
        az ml job create \
            --file "$REPO_DIR/azureml/finetune_job.yml" \
            "${AML_ARGS[@]}" \
            --set compute="azureml:$AML_COMPUTE" \
            --query name --output tsv
        echo "Finetune job submitted."
        ;;
esac
