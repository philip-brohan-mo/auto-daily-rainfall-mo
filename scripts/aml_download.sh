#!/usr/bin/env bash
# aml_download.sh — Download data from the Azure ML workspace datastore.
#
# Sources azureml/config.env for workspace coordinates and default paths.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash scripts/aml_download.sh [what] [options]
#
# ── What to download (pick one or more) ──────────────────────────────────────
#   extractions           Download $AML_OUTPUTS_PATH/extractions → outputs/extractions/
#   eval                  Download $AML_OUTPUTS_PATH/eval        → outputs/eval/
#   checkpoints           Download $AML_OUTPUTS_PATH/checkpoints → outputs/checkpoints/
#   all                   Download all three output directories
#   --src PATH --dst DIR  Download any datastore path to a custom local directory
#
# ── Options ───────────────────────────────────────────────────────────────────
#   --output-dir DIR      Root local output directory (default: outputs/)
#   --dry-run             Print az storage commands without executing them
#   --help
#
# ── Examples ──────────────────────────────────────────────────────────────────
#   bash scripts/aml_download.sh extractions
#   bash scripts/aml_download.sh all
#   bash scripts/aml_download.sh --src my_project/outputs/eval --dst /tmp/eval

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$REPO_DIR/azureml/config.env"

[[ -f "$CONFIG_FILE" ]] && source "$CONFIG_FILE"

AML_SUBSCRIPTION="${AML_SUBSCRIPTION:-}"
AML_RESOURCE_GROUP="${AML_RESOURCE_GROUP:-}"
AML_WORKSPACE="${AML_WORKSPACE:-}"
AML_DATASTORE_BASE="${AML_DATASTORE_BASE:-azureml://datastores/workspaceblobstore/paths}"
AML_OUTPUTS_PATH="${AML_OUTPUTS_PATH:-outputs}"

OUTPUT_DIR="${REPO_DIR}/outputs"
CUSTOM_SRC=""
CUSTOM_DST=""
DRY_RUN=false
TARGETS=()

usage() {
    sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        extractions|eval|checkpoints|all) TARGETS+=("$1"); shift ;;
        --src)             CUSTOM_SRC="$2"; shift 2 ;;
        --dst)             CUSTOM_DST="$2"; shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --help|-h)         usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

if [[ ${#TARGETS[@]} -eq 0 && -z "$CUSTOM_SRC" ]]; then
    echo "Error: specify what to download (extractions|eval|checkpoints|all) or --src/--dst" >&2
    usage 1
fi

[[ -z "$AML_SUBSCRIPTION" ]]   && { echo "Error: AML_SUBSCRIPTION not set" >&2; exit 1; }
[[ -z "$AML_RESOURCE_GROUP" ]] && { echo "Error: AML_RESOURCE_GROUP not set" >&2; exit 1; }
[[ -z "$AML_WORKSPACE" ]]      && { echo "Error: AML_WORKSPACE not set" >&2; exit 1; }

# ── Resolve storage account and container from the datastore ─────────────────
DATASTORE_NAME="$(echo "$AML_DATASTORE_BASE" | sed 's|azureml://datastores/||;s|/paths.*||')"
echo "Resolving datastore '$DATASTORE_NAME' in workspace '$AML_WORKSPACE'..."
DATASTORE_JSON="$(az ml datastore show \
    --name "$DATASTORE_NAME" \
    --workspace-name "$AML_WORKSPACE" \
    --resource-group "$AML_RESOURCE_GROUP" \
    --subscription "$AML_SUBSCRIPTION" \
    --output json)"

STORAGE_ACCOUNT="$(echo "$DATASTORE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['account_name'])")"
CONTAINER="$(echo "$DATASTORE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['container_name'])")"
echo "Storage account: $STORAGE_ACCOUNT  container: $CONTAINER"
echo

# ── Download helper ───────────────────────────────────────────────────────────
do_download() {
    local src_path="$1"
    local dst="$2"
    mkdir -p "$dst"
    echo "Downloading https://${STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER}/${src_path}"
    echo "         → $dst"
    local cmd=(
        az storage blob download-batch
        --account-name "$STORAGE_ACCOUNT"
        --auth-mode login
        --source "$CONTAINER"
        --source-path "$src_path/*"
        --destination "$dst"
        --overwrite true
    )
    if $DRY_RUN; then
        echo "[dry-run] ${cmd[*]}"
    else
        "${cmd[@]}"
        echo "Done."
    fi
    echo
}

# ── Run downloads ─────────────────────────────────────────────────────────────
for target in "${TARGETS[@]}"; do
    case "$target" in
        extractions)
            do_download "$AML_OUTPUTS_PATH/extractions" "$OUTPUT_DIR/extractions"
            ;;
        eval)
            do_download "$AML_OUTPUTS_PATH/eval" "$OUTPUT_DIR/eval"
            ;;
        checkpoints)
            do_download "$AML_OUTPUTS_PATH/checkpoints" "$OUTPUT_DIR/checkpoints"
            ;;
        all)
            do_download "$AML_OUTPUTS_PATH/extractions" "$OUTPUT_DIR/extractions"
            do_download "$AML_OUTPUTS_PATH/eval"        "$OUTPUT_DIR/eval"
            do_download "$AML_OUTPUTS_PATH/checkpoints" "$OUTPUT_DIR/checkpoints"
            ;;
    esac
done

if [[ -n "$CUSTOM_SRC" ]]; then
    [[ -z "$CUSTOM_DST" ]] && { echo "Error: --dst required with --src" >&2; exit 1; }
    do_download "$CUSTOM_SRC" "$CUSTOM_DST"
fi
