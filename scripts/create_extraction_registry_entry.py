#!/usr/bin/env python3
"""Create or update a local registry entry for extraction submissions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create extraction registry entry")
    parser.add_argument(
        "--extractions-path",
        required=True,
        help="Path to extraction outputs relative to datastore root",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model preset or model id used for extraction",
    )
    parser.add_argument(
        "--model-slug",
        required=True,
        help="Model slug used in extraction output directory",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset/images path used for extraction",
    )
    parser.add_argument(
        "--images-path",
        required=True,
        help="Images path used for extraction",
    )
    parser.add_argument(
        "--transcriptions-path",
        default="",
        help="Transcriptions path for the selected dataset, if available",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="Checkpoint path used for extraction, if any",
    )
    parser.add_argument(
        "--total-shards",
        type=int,
        required=True,
        help="Number of shards submitted",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional extraction limit per shard",
    )
    parser.add_argument(
        "--job-ids",
        default="",
        help="Comma-separated Azure ML job IDs for this extraction run",
    )
    parser.add_argument(
        "--registry-file",
        required=True,
        help="Path to extraction registry JSON file",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional notes about this extraction run",
    )
    args = parser.parse_args()

    timestamp = _utc_now_iso()
    extraction_dir = Path(args.extractions_path)

    job_ids = [j.strip() for j in args.job_ids.split(",") if j.strip()]

    entry = {
        "extractions_path": args.extractions_path,
        "run_name": extraction_dir.name,
        "model_slug": args.model_slug,
        "model": args.model,
        "dataset": args.dataset,
        "images_path": args.images_path,
        "transcriptions_path": args.transcriptions_path,
        "checkpoint_path": args.checkpoint_path,
        "total_shards": args.total_shards,
        "limit": args.limit,
        "job_ids": job_ids,
        "created_at": timestamp,
        "notes": args.notes,
    }

    registry_path = Path(args.registry_file)
    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {"extractions": []}

    extractions = registry.setdefault("extractions", [])

    # Replace existing entry for the same extraction path to keep reruns idempotent.
    replaced = False
    for idx, existing in enumerate(extractions):
        if existing.get("extractions_path") == args.extractions_path:
            extractions[idx] = entry
            replaced = True
            break
    if not replaced:
        extractions.append(entry)

    registry["updated_at"] = timestamp

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    action = "Updated" if replaced else "Added"
    print(f"{action} extraction registry: {registry_path}")
    print(f"  Run: {entry['run_name']}")
    print(f"  Model: {entry['model']}")
    print(f"  Dataset: {entry['dataset']}")


if __name__ == "__main__":
    main()
