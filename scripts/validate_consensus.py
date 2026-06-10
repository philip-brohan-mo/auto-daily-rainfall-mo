#!/usr/bin/env python3
"""Generate validation plots for consensus transcriptions.

This script mirrors the original 2-panel visualize output by reusing
``weather_doc_extractor.visualize.make_figure``.

Panels are:
1. Source image.
2. Consensus output table.

The only intentional difference from the original is table text colour:
green for consensus-correct cells and red for consensus-incorrect cells.

Output files follow the existing validation naming convention:
``<stem>_comparison.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make the src package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from weather_doc_extractor.schemas import DailyRainfallGrid
from weather_doc_extractor.visualize import make_figure

_ROW_LABELS = [f"Day {i}" for i in range(1, 32)] + ["Totals"]
_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate consensus validation figures (image + consensus table)"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/consensus_dataset_1000"),
        help="Root of consensus dataset (default: outputs/consensus_dataset_1000)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory containing source images (default: <dataset-root>/images)",
    )
    parser.add_argument(
        "--consensus-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing consensus JSON files "
            "(default: <dataset-root>/consensus_transcriptions)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation/consensus/figures"),
        help="Directory for output figures (default: outputs/validation/consensus/figures)",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Only process one stem (default: process all stems in consensus-dir)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of stems to process (after sorting)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG DPI (default: 150)",
    )
    return parser.parse_args()


def _load_consensus(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _consensus_to_grid(
    consensus: dict[str, Any],
) -> tuple[DailyRainfallGrid, list[list[bool]]]:
    days: dict[str, list[float | None]] = {}
    correct_mask: list[list[bool]] = []

    for row_label in _ROW_LABELS[:-1]:
        row_data = consensus.get(row_label, [])
        values: list[float | None] = []
        row_mask: list[bool] = []
        for month_idx in range(12):
            cell = (
                row_data[month_idx]
                if isinstance(row_data, list) and month_idx < len(row_data)
                else {}
            )
            if not isinstance(cell, dict):
                cell = {}
            values.append(_coerce_float(cell.get("value")))
            row_mask.append(bool(cell.get("correct", False)))
        days[row_label] = values
        correct_mask.append(row_mask)

    totals_data = consensus.get("Totals", [])
    totals: list[float | None] = []
    totals_mask: list[bool] = []
    for month_idx in range(12):
        cell = (
            totals_data[month_idx]
            if isinstance(totals_data, list) and month_idx < len(totals_data)
            else {}
        )
        if not isinstance(cell, dict):
            cell = {}
        totals.append(_coerce_float(cell.get("value")))
        totals_mask.append(bool(cell.get("correct", False)))
    correct_mask.append(totals_mask)

    return DailyRainfallGrid(days=days, totals=totals), correct_mask


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in _IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _get_table_artist(figure):
    # Axis 1 is the right-hand table panel in make_figure.
    if len(figure.axes) < 2:
        return None
    table_axis = figure.axes[1]
    for artist in table_axis.get_children():
        if hasattr(artist, "get_celld"):
            return artist
    return None


def _colour_consensus_panel(figure, correct_mask: list[list[bool]]) -> bool:
    table_artist = _get_table_artist(figure)
    if table_artist is None:
        return False

    cells = table_artist.get_celld()
    for (row, col), cell in cells.items():
        if row <= 0 or col < 0:
            continue
        is_correct = correct_mask[row - 1][col]
        cell.get_text().set_color("#1b7837" if is_correct else "#b2182b")
    return True


def _iter_stems(
    consensus_dir: Path, one_stem: str | None, limit: int | None
) -> list[str]:
    if one_stem is not None:
        stems = [one_stem]
    else:
        stems = sorted(path.stem for path in consensus_dir.glob("*.json"))
    if limit is not None:
        stems = stems[:limit]
    return stems


def main() -> int:
    args = _parse_args()

    dataset_root = args.dataset_root.resolve()
    images_dir = (
        args.images_dir if args.images_dir is not None else dataset_root / "images"
    ).resolve()
    consensus_dir = (
        args.consensus_dir
        if args.consensus_dir is not None
        else dataset_root / "consensus_transcriptions"
    ).resolve()
    output_dir = args.output_dir.resolve()

    if not images_dir.exists():
        print(f"Error: images directory not found: {images_dir}", file=sys.stderr)
        return 1
    if not consensus_dir.exists():
        print(f"Error: consensus directory not found: {consensus_dir}", file=sys.stderr)
        return 1

    stems = _iter_stems(consensus_dir, args.stem, args.limit)
    if not stems:
        print("No stems to process.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved = 0
    skipped = 0

    for stem in stems:
        consensus_path = consensus_dir / f"{stem}.json"
        consensus = _load_consensus(consensus_path)
        if consensus is None:
            print(f"Skipping {stem}: invalid consensus JSON ({consensus_path})")
            skipped += 1
            continue

        image_path = _find_image(images_dir, stem)
        if image_path is None:
            print(f"Skipping {stem}: image not found in {images_dir}")
            skipped += 1
            continue

        grid, correct_mask = _consensus_to_grid(consensus)
        title = stem

        fig = make_figure(
            image_path=image_path,
            grid=grid,
            title=title,
            model_name="Consensus",
        )

        if not _colour_consensus_panel(fig, correct_mask):
            print(
                f"Warning: could not locate consensus table artist for {stem}; using default colours"
            )

        out_path = output_dir / f"{stem}_comparison.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        saved += 1
        print(f"Figure: {out_path}")

    print(f"Done. Saved {saved} figure(s); skipped {skipped}.")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
