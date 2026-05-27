#!/usr/bin/env bash
# azure_register_environments.sh — Register or update curated environments in Azure ML.
#
# Registers/updates the weather-doc-extractor environments in Azure ML workspace:
#   1. weather-doc-extractor (V100 variant, PyTorch 2.4 + CUDA 12.1)
#   2. weather-doc-extractor-a100 (A100 variant, PyTorch 2.8 + CUDA 12.6)
#
# Uses azureml/config.env for workspace coordinates.
#
# Usage:
#   bash scripts/azure_register_environments.sh [--variant v100|a100|both] [--force]
#
# Options:
#   --variant VARIANT   Register only v100, a100, or both (default: both)
#   --version-mode MODE  Version strategy: auto or fixed (default: auto)
#   --dry-run          Print az ml environment create commands without executing them
#
# Examples:
#   bash scripts/azure_register_environments.sh
#   bash scripts/azure_register_environments.sh --variant a100
#   bash scripts/azure_register_environments.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$REPO_DIR/azureml/config.env"

# Source config file for workspace credentials
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: $CONFIG_FILE not found" >&2
    exit 1
fi
source "$CONFIG_FILE"

AML_SUBSCRIPTION="${AML_SUBSCRIPTION:-}"
AML_RESOURCE_GROUP="${AML_RESOURCE_GROUP:-}"
AML_WORKSPACE="${AML_WORKSPACE:-}"

[[ -z "$AML_SUBSCRIPTION" ]]   && { echo "Error: AML_SUBSCRIPTION not set in $CONFIG_FILE" >&2; exit 1; }
[[ -z "$AML_RESOURCE_GROUP" ]] && { echo "Error: AML_RESOURCE_GROUP not set in $CONFIG_FILE" >&2; exit 1; }
[[ -z "$AML_WORKSPACE" ]]      && { echo "Error: AML_WORKSPACE not set in $CONFIG_FILE" >&2; exit 1; }

VARIANT="both"
VERSION_MODE="auto"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            VARIANT="$2"
            [[ "$VARIANT" =~ ^(v100|a100|both)$ ]] || { echo "Error: --variant must be v100, a100, or both" >&2; exit 1; }
            shift 2
            ;;
        --version-mode)
            VERSION_MODE="$2"
            [[ "$VERSION_MODE" =~ ^(auto|fixed)$ ]] || { echo "Error: --version-mode must be auto or fixed" >&2; exit 1; }
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
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

register_environment() {
    local env_name="$1"
    local env_file="$2"
    local env_dir
    local use_file="$env_file"
    local base_version
    local next_version

    if [[ "$VERSION_MODE" == "auto" ]]; then
        base_version="$(awk -F': ' '/^version:/ {print $2; exit}' "$env_file")"
        if [[ -z "$base_version" ]]; then
            echo "Error: could not read version from $env_file" >&2
            exit 1
        fi

        if $DRY_RUN; then
            next_version="$((base_version + 1))"
        else
            local current_latest
            current_latest="$(az ml environment list \
                --name "$env_name" \
                --workspace-name "$AML_WORKSPACE" \
                --resource-group "$AML_RESOURCE_GROUP" \
                --subscription "$AML_SUBSCRIPTION" \
                --query 'max_by([?name==`'"$env_name"'`], &to_number(version)).version' \
                -o tsv 2>/dev/null || true)"

            if [[ -z "$current_latest" || "$current_latest" == "null" ]]; then
                next_version="$base_version"
            else
                next_version="$((current_latest + 1))"
            fi
        fi

        env_dir="$(dirname "$env_file")"
        use_file="$(mktemp "$env_dir/.tmp_${env_name//-/_}_XXXXXX.yml")"
        awk -v v="$next_version" '{ if ($1=="version:") { print "version: " v } else { print } }' "$env_file" > "$use_file"
        echo "[azure-register] Using auto version $next_version for $env_name"
    fi
    
    echo "[azure-register] Registering environment: $env_name"
    
    local cmd=(
        az ml environment create
        --name "$env_name"
        --file "$use_file"
        --workspace-name "$AML_WORKSPACE"
        --resource-group "$AML_RESOURCE_GROUP"
        --subscription "$AML_SUBSCRIPTION"
    )
    cmd+=(--no-wait)
    
    if $DRY_RUN; then
        echo "[dry-run] ${cmd[*]}"
    else
        "${cmd[@]}"
    fi

    if [[ "$use_file" != "$env_file" ]]; then
        rm -f "$use_file"
    fi
}

if [[ "$VARIANT" == "v100" || "$VARIANT" == "both" ]]; then
    register_environment \
        "weather-doc-extractor" \
        "$REPO_DIR/azureml/environment.yml"
    echo
fi

if [[ "$VARIANT" == "a100" || "$VARIANT" == "both" ]]; then
    register_environment \
        "weather-doc-extractor-a100" \
        "$REPO_DIR/azureml/environment-a100.yml"
    echo
fi

echo "[azure-register] Environment registration submitted."
echo "[azure-register] (Using --no-wait; jobs will build asynchronously.)"
echo "[azure-register] Check environment build status in Azure ML Studio or with:"
echo "                 az ml environment show --name weather-doc-extractor ..."
