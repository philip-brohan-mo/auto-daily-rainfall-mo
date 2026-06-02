#!/usr/bin/env python3
"""
Create or update the model registry JSON file after a fine-tuning job completes.

Usage:
    python scripts/create_model_registry_entry.py \
        --checkpoint-path outputs/checkpoints/granite-fake-20260526-143000 \
        --base-model granite \
        --dataset fake \
        --registry-file outputs/model_registry.json

The registry tracks all trained models for easy reference and comparison.
"""

import json
import argparse
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create model registry entry")
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to the checkpoint (relative to datastore root)",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model used (e.g., granite, smolvlm2)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset used for training (e.g., real, fake)",
    )
    parser.add_argument(
        "--registry-file",
        required=True,
        help="Path to the registry JSON file",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional notes about this training run",
    )
    parser.add_argument(
        "--job-id",
        default="",
        help="Optional Azure ML job ID for this training run",
    )
    parser.add_argument(
        "--status",
        default="completed",
        choices=["submitted", "running", "completed", "failed"],
        help="Lifecycle status for this registry entry",
    )
    args = parser.parse_args()

    # Parse checkpoint path to extract timestamp
    checkpoint_name = Path(args.checkpoint_path).name
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    # Create registry entry
    entry = {
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_name": checkpoint_name,
        "base_model": args.base_model,
        "dataset": args.dataset,
        "job_id": args.job_id,
        "status": args.status,
        "created_at": timestamp,
        "notes": args.notes,
    }

    # Load existing registry or create new
    registry_path = Path(args.registry_file)
    if registry_path.exists():
        if registry_path.stat().st_size == 0:
            registry = {"models": []}
        else:
            with open(registry_path) as f:
                try:
                    registry = json.load(f)
                except JSONDecodeError as exc:
                    raise ValueError(
                        f"Registry file is not valid JSON: {registry_path}. "
                        "Fix or remove it and retry."
                    ) from exc
    else:
        registry = {"models": []}

    # Upsert by checkpoint path so repeated registration updates metadata
    replaced = False
    for idx, existing in enumerate(registry["models"]):
        if existing.get("checkpoint_path") == args.checkpoint_path:
            registry["models"][idx] = entry
            replaced = True
            break
    if not replaced:
        registry["models"].append(entry)

    registry["updated_at"] = timestamp

    # Write back
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Registry updated: {args.registry_file}")
    print(f"  Checkpoint: {checkpoint_name}")
    print(f"  Base model: {args.base_model}")
    print(f"  Dataset: {args.dataset}")
    if args.job_id:
        print(f"  Job ID: {args.job_id}")
    print(f"  Status: {args.status}")


if __name__ == "__main__":
    main()
