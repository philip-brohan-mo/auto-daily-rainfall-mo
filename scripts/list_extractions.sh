#!/usr/bin/env bash
# list_extractions.sh — List registered extraction runs and their output paths.

set -euo pipefail

usage() {
    cat <<EOF
list_extractions.sh — List submitted extraction runs from a local registry

Usage:
  bash scripts/list_extractions.sh [OPTIONS]

Options:
  --registry FILE         Path to extraction_registry.json (default: outputs/extraction_registry.json)
  --format json|table     Output format (default: table)
  --help                  Show this message

Examples:
  # List all registered extraction runs
  bash scripts/list_extractions.sh

  # List as JSON
  bash scripts/list_extractions.sh --format json

  # Use custom registry file
  bash scripts/list_extractions.sh --registry /path/to/extraction_registry.json
EOF
    exit "${1:-0}"
}

REGISTRY_FILE="outputs/extraction_registry.json"
FORMAT="table"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry) REGISTRY_FILE="$2"; shift 2 ;;
        --format)   FORMAT="$2";        shift 2 ;;
        --help|-h)  usage 0 ;;
        *)          echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

if [[ ! -f "$REGISTRY_FILE" ]]; then
    echo "Error: Registry file not found: $REGISTRY_FILE"
    echo "No extractions registered yet. Run 'bash scripts/aml_submit.sh extract' first."
    exit 1
fi

case "$FORMAT" in
    json)
        cat "$REGISTRY_FILE"
        ;;
    table)
        echo "Registered Extraction Runs:"
        echo "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────"
        python3 << PYTHON
import json
from pathlib import Path

registry_file = Path("$REGISTRY_FILE")
if not registry_file.exists():
    print("Registry file not found: $REGISTRY_FILE")
    raise SystemExit(1)

with registry_file.open("r", encoding="utf-8") as f:
    registry = json.load(f)

runs = registry.get("extractions", [])
if not runs:
    print("No extraction runs registered.")
    raise SystemExit(0)

print(f"{'Run':<18} {'Model':<22} {'Dataset':<35} {'Created':<12} {'Checkpoint':<10}")
print("─" * 110)

for run in runs:
    run_name = run.get("run_name", "unknown")
    model = run.get("model_slug") or run.get("model", "unknown")
    dataset = run.get("dataset", "unknown")
    created = run.get("created_at", "unknown")[:10]
    has_ckpt = "yes" if run.get("checkpoint_path") else "no"
    print(f"{run_name:<18} {model:<22} {dataset:<35} {created:<12} {has_ckpt:<10}")

print()
print("Extraction output paths:")
for i, run in enumerate(runs, 1):
    print(f"  {i}. {run.get('extractions_path', 'unknown')}")
PYTHON
        ;;
    *)
        echo "Unknown format: $FORMAT" >&2
        exit 1
        ;;
esac
