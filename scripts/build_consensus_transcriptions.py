#!/usr/bin/env python3
"""Build consensus transcriptions (Option A format) from 5 extraction directories.

Each output cell is:
  {"value": <float|null>, "correct": <bool>}

A cell is marked correct when at least `--agreement-threshold` models produce the
same normalized value.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

DAYS = [f"Day {i}" for i in range(1, 32)]
ALL_KEYS = DAYS + ["Totals"]


def _normalize_value(v: Any, precision: int) -> float | None:
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() == "null":
        return None
    try:
        return round(float(v), precision)
    except (TypeError, ValueError):
        return None


def _empty_consensus() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for key in ALL_KEYS:
        out[key] = [{"value": None, "correct": False} for _ in range(12)]
    return out


def _load_extraction(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if data.get("parse_failed"):
        return None
    grid = data.get("grid")
    if not isinstance(grid, dict):
        return None
    return grid


def _get_row(grid: dict[str, Any], key: str) -> list[Any] | None:
    """Return a 12-value row for ``key`` from either supported grid schema.

    Supports both:
    - flat schema: {"Day 1": [...], ..., "Totals": [...]}
    - nested schema: {"days": {"Day 1": [...]}, "totals": [...]}
    """
    # Flat schema support
    flat_row = grid.get(key)
    if isinstance(flat_row, list):
        return flat_row

    # Nested schema support used by extraction outputs
    if key == "Totals":
        totals = grid.get("totals")
        if isinstance(totals, list):
            return totals
    else:
        days = grid.get("days")
        if isinstance(days, dict):
            day_row = days.get(key)
            if isinstance(day_row, list):
                return day_row

    return None


def _collect_stems(input_dirs: list[Path]) -> set[str]:
    stems: set[str] = set()
    for d in input_dirs:
        for p in d.glob("*.json"):
            stems.add(p.stem)
    return stems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build consensus transcriptions")
    parser.add_argument(
        "--input-dir",
        action="append",
        dest="input_dirs",
        required=True,
        help="Model extraction directory (repeat 5 times)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for consensus transcription JSON files",
    )
    parser.add_argument(
        "--agreement-threshold",
        type=int,
        default=2,
        help="Minimum agreeing models to mark correct=true (default: 2)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal precision for value normalization before voting",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional path to write summary JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = [Path(p).resolve() for p in args.input_dirs]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for d in input_dirs:
        if not d.exists():
            raise SystemExit(f"Input dir not found: {d}")

    stems = sorted(_collect_stems(input_dirs))
    if not stems:
        raise SystemExit("No extraction JSON files found in input directories")

    stats = {
        "total_stems": len(stems),
        "total_cells": 0,
        "correct_cells": 0,
        "incorrect_cells": 0,
        "missing_model_files": 0,
        "parse_failed_or_invalid": 0,
        "input_dirs": [str(d) for d in input_dirs],
        "agreement_threshold": args.agreement_threshold,
        "precision": args.precision,
    }

    for stem in stems:
        grids: list[dict[str, Any] | None] = []
        for d in input_dirs:
            p = d / f"{stem}.json"
            if not p.exists():
                stats["missing_model_files"] += 1
                grids.append(None)
                continue
            grid = _load_extraction(p)
            if grid is None:
                stats["parse_failed_or_invalid"] += 1
            grids.append(grid)

        consensus = _empty_consensus()

        for key in ALL_KEYS:
            for month_idx in range(12):
                votes: list[float | None] = []
                for grid in grids:
                    if grid is None:
                        continue
                    row = _get_row(grid, key)
                    if not isinstance(row, list) or len(row) <= month_idx:
                        continue
                    votes.append(_normalize_value(row[month_idx], args.precision))

                if not votes:
                    chosen = None
                    top_count = 0
                else:
                    counts = Counter(votes)
                    top_count = max(counts.values())
                    # Deterministic tie-break: first value in encounter order among top counts.
                    top_values = {v for v, c in counts.items() if c == top_count}
                    chosen = next(v for v in votes if v in top_values)

                is_correct = top_count >= args.agreement_threshold
                consensus[key][month_idx] = {"value": chosen, "correct": is_correct}

                stats["total_cells"] += 1
                if is_correct:
                    stats["correct_cells"] += 1
                else:
                    stats["incorrect_cells"] += 1

        out_file = output_dir / f"{stem}.json"
        out_file.write_text(json.dumps(consensus, indent=2), encoding="utf-8")

    coverage = 0.0
    if stats["total_cells"] > 0:
        coverage = stats["correct_cells"] / stats["total_cells"]
    stats["agreement_coverage"] = round(coverage, 6)

    print(json.dumps(stats, indent=2), flush=True)
    if args.summary_file is not None:
        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        args.summary_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
