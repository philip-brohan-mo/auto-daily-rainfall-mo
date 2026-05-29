#!/usr/bin/env python3
"""Create deterministic test dataset splits from the real and fake datasets.

Selects 10 paired image/transcription records from each of
  - Daily_rainfall_sample/
  - fake_daily_rainfall/
and copies them into
  - test_data/real/images/      test_data/real/transcriptions/
  - test_data/fake/images/      test_data/fake/transcriptions/

Selection is deterministic: every N-th stem in sorted order, where N is chosen
so that exactly 10 pairs are selected and they are spread evenly across the
alphabet and decade range.  The same stems are always selected unless the
source datasets change.

Running this script more than once is safe: it overwrites existing test data
files with identical content.

Usage
-----
    python scripts/create_test_splits.py [--n-test 10] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SOURCES: dict[str, dict[str, Path]] = {
    "real": {
        "images": REPO_ROOT / "Daily_rainfall_sample" / "images",
        "transcriptions": REPO_ROOT / "Daily_rainfall_sample" / "transcriptions",
    },
    "fake": {
        "images": REPO_ROOT / "fake_daily_rainfall" / "images",
        "transcriptions": REPO_ROOT / "fake_daily_rainfall" / "transcriptions",
    },
}

TEST_DATA_ROOT = REPO_ROOT / "test_data"


def _find_paired_stems(src: dict[str, Path]) -> list[str]:
    """Return sorted list of stems that have both an image and a transcription."""
    image_stems = {p.stem for p in src["images"].glob("*.jpg")}
    transcription_stems = {p.stem for p in src["transcriptions"].glob("*.json")}
    paired = sorted(image_stems & transcription_stems)
    return paired


def _select_evenly(stems: list[str], n: int) -> list[str]:
    """Select *n* stems spread evenly across the sorted list."""
    if len(stems) <= n:
        return stems
    # Compute step so that we get exactly n indices spanning the full list.
    # Use integer arithmetic to avoid floating-point skew.
    step = len(stems) / n
    return [stems[int(i * step)] for i in range(n)]


def create_split(
    dataset_name: str,
    n_test: int = 10,
    dry_run: bool = False,
) -> list[str]:
    """Create test split for one dataset.  Returns the list of selected stems."""
    src = SOURCES[dataset_name]
    dst_images = TEST_DATA_ROOT / dataset_name / "images"
    dst_transcriptions = TEST_DATA_ROOT / dataset_name / "transcriptions"

    paired_stems = _find_paired_stems(src)
    if not paired_stems:
        print(f"  [ERROR] No paired records found in {src['images']}", file=sys.stderr)
        return []

    selected = _select_evenly(paired_stems, n_test)

    print(
        f"\n{dataset_name}: selecting {len(selected)} of {len(paired_stems)} paired records"
    )
    for stem in selected:
        print(f"  {stem}")

    if dry_run:
        print("  [dry-run] Skipping file copies.")
        return selected

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_transcriptions.mkdir(parents=True, exist_ok=True)

    for stem in selected:
        img_src = src["images"] / f"{stem}.jpg"
        img_dst = dst_images / f"{stem}.jpg"
        shutil.copy2(img_src, img_dst)

        json_src = src["transcriptions"] / f"{stem}.json"
        json_dst = dst_transcriptions / f"{stem}.json"
        shutil.copy2(json_src, json_dst)

    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=10,
        help="Number of test pairs to select from each dataset (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without copying any files",
    )
    args = parser.parse_args(argv)

    all_stems: dict[str, list[str]] = {}
    for dataset_name in ("real", "fake"):
        stems = create_split(dataset_name, n_test=args.n_test, dry_run=args.dry_run)
        all_stems[dataset_name] = stems

    # Write a manifest so downstream tools know which stems are held out.
    if not args.dry_run:
        manifest_path = TEST_DATA_ROOT / "test_stems.json"
        manifest_path.write_text(json.dumps(all_stems, indent=2))
        print(f"\nTest stem manifest written to: {manifest_path}")

    total = sum(len(v) for v in all_stems.values())
    print(f"\nDone: {total} test pairs created across {len(all_stems)} datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
