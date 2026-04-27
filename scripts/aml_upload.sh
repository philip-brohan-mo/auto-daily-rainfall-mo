#!/usr/bin/env bash
# aml_upload.sh — Upload local data to the Azure ML workspace datastore.
#
# Sources azureml/config.env for workspace coordinates and default paths.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash scripts/aml_upload.sh [what] [options]
#
# ── What to upload (pick one or more) ────────────────────────────────────────
#   images                Upload local images dir  → $AML_IMAGES_PATH
#   transcriptions        Upload local transcriptions dir → $AML_TRANSCRIPTIONS_PATH
#   all                   Upload both images and transcriptions
#   --src DIR --dst PATH  Upload any local directory to a custom datastore path
#
# ── Options ───────────────────────────────────────────────────────────────────
#   --local-images DIR    Override local images directory   (default: Daily_rainfall_sample/images)
#   --local-transcriptions DIR  Override local transcriptions dir
#   --dry-run             Print az storage commands without executing them
#   --help
#
# ── Examples ──────────────────────────────────────────────────────────────────
#   bash scripts/aml_upload.sh all
#   bash scripts/aml_upload.sh images
#   bash scripts/aml_upload.sh --src /data/my_images --dst my_project/images

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$REPO_DIR/azureml/config.env"

[[ -f "$CONFIG_FILE" ]] && source "$CONFIG_FILE"

AML_SUBSCRIPTION="${AML_SUBSCRIPTION:-}"
AML_RESOURCE_GROUP="${AML_RESOURCE_GROUP:-}"
AML_WORKSPACE="${AML_WORKSPACE:-}"
AML_DATASTORE_BASE="${AML_DATASTORE_BASE:-azureml://datastores/workspaceblobstore/paths}"
AML_IMAGES_PATH="${AML_IMAGES_PATH:-Daily_rainfall_sample/images}"
AML_TRANSCRIPTIONS_PATH="${AML_TRANSCRIPTIONS_PATH:-Daily_rainfall_sample/transcriptions}"

LOCAL_IMAGES="${REPO_DIR}/Daily_rainfall_sample/images"
LOCAL_TRANSCRIPTIONS="${REPO_DIR}/Daily_rainfall_sample/transcriptions"
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
        images|transcriptions|all) TARGETS+=("$1"); shift ;;
        --src)             CUSTOM_SRC="$2"; shift 2 ;;
        --dst)             CUSTOM_DST="$2"; shift 2 ;;
        --local-images)    LOCAL_IMAGES="$2"; shift 2 ;;
        --local-transcriptions) LOCAL_TRANSCRIPTIONS="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --help|-h)         usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

if [[ -z "${TARGETS[*]:-}" && -z "$CUSTOM_SRC" ]]; then
    echo "Error: specify what to upload (images|transcriptions|all) or --src/--dst" >&2
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

# ── Upload helper ─────────────────────────────────────────────────────────────
do_upload() {
    local src="$1"
    local dst_path="$2"
    echo "Uploading  $src"
    echo "        → https://${STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER}/${dst_path}"
    local cmd=(
        az storage blob upload-batch
        --account-name "$STORAGE_ACCOUNT"
        --auth-mode login
        --source "$src"
        --destination "$CONTAINER"
        --destination-path "$dst_path"
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

# ── Run uploads ───────────────────────────────────────────────────────────────
for target in "${TARGETS[@]:-}"; do
    case "$target" in
        images)
            do_upload "$LOCAL_IMAGES" "$AML_IMAGES_PATH"
            ;;
        transcriptions)
            do_upload "$LOCAL_TRANSCRIPTIONS" "$AML_TRANSCRIPTIONS_PATH"
            ;;
        all)
            do_upload "$LOCAL_IMAGES" "$AML_IMAGES_PATH"
            do_upload "$LOCAL_TRANSCRIPTIONS" "$AML_TRANSCRIPTIONS_PATH"
            ;;
    esac
done

if [[ -n "$CUSTOM_SRC" ]]; then
    [[ -z "$CUSTOM_DST" ]] && { echo "Error: --dst required with --src" >&2; exit 1; }
    do_upload "$CUSTOM_SRC" "$CUSTOM_DST"
fi
