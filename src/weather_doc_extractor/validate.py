"""Validation utilities: compare downloaded extraction results against test data.

This module is the local counterpart to the Azure ML evaluation pipeline.
It loads per-stem JSON files produced by the ``batch-extract`` Azure command,
matches them to ground-truth transcriptions and source images in a test
dataset directory, and produces accuracy statistics and diagnostic figures.

Typical workflow
----------------
1. Run ``batch-extract`` on a test dataset via Azure ML (see ``aml_submit.sh``).
2. Download the per-stem JSON extraction results (see ``aml_download.sh``).
3. Run the ``validate`` CLI command (or call these functions directly)::

       weather-extract validate \\
           --extractions outputs/extractions/<model>/<run>/ \\
           --test-data test_data/real/ \\
           --output-dir outputs/validation/my-run/

Public API
----------
load_extraction_results(extractions_dir, test_data_dir)
    Pair per-stem extraction JSONs with ground-truth transcriptions and images.

run_validation(records, tolerance)
    Compare all predictions against ground truth; return an EvaluationReport.

print_summary(report)
    Print a formatted per-record table and aggregate statistics to stdout.

save_report(report, path)
    Write the full JSON report to *path*.

generate_figures(records, report, output_dir, model_name, tolerance)
    Save one three-panel comparison figure per test record.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from weather_doc_extractor.evaluate import (
    EvaluationReport,
    GridComparison,
    compare_grids,
)
from weather_doc_extractor.ingest import load_grid
from weather_doc_extractor.schemas import DailyRainfallGrid

if TYPE_CHECKING:
    pass  # only for forward references


@dataclass
class ValidationRecord:
    """One test record pairing a prediction with its ground truth."""

    stem: str
    predicted: DailyRainfallGrid | None  # None when parse_failed
    ground_truth: DailyRainfallGrid
    image_path: Path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_extraction_results(
    extractions_dir: Path,
    test_data_dir: Path,
) -> list[ValidationRecord]:
    """Load and pair extraction results with ground-truth transcriptions.

    Parameters
    ----------
    extractions_dir:
        Directory containing per-stem JSON files produced by ``batch-extract``
        (``<stem>.json`` with keys ``stem``, ``parse_failed``, ``grid``).
    test_data_dir:
        Local test dataset directory containing ``images/`` and
        ``transcriptions/`` subdirectories (e.g. ``test_data/real/``).

    Returns
    -------
    list[ValidationRecord]
        Records sorted by stem.  Stems that have a ground-truth transcription
        but no matching extraction JSON are skipped with a warning.
    """
    images_dir = test_data_dir / "images"
    transcriptions_dir = test_data_dir / "transcriptions"

    # Discover stems with a ground-truth transcription.
    gt_stems = {p.stem: p for p in sorted(transcriptions_dir.glob("*.json"))}
    if not gt_stems:
        print(
            f"WARNING: no transcription files found in {transcriptions_dir}",
            file=sys.stderr,
        )
        return []

    records: list[ValidationRecord] = []
    for stem, gt_path in gt_stems.items():
        extraction_path = extractions_dir / f"{stem}.json"
        if not extraction_path.exists():
            print(
                f"WARNING: no extraction result for {stem} (expected {extraction_path})",
                file=sys.stderr,
            )
            continue

        # Find the corresponding image (prefer .jpg, fall back to .png)
        image_path: Path | None = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            print(f"WARNING: image not found for {stem}", file=sys.stderr)
            continue

        # Load ground truth
        ground_truth = load_grid(gt_path)

        # Load prediction
        raw = json.loads(extraction_path.read_text())
        if raw.get("parse_failed", False):
            predicted = None
        else:
            grid_dict = raw["grid"]
            predicted = DailyRainfallGrid(
                days=grid_dict["days"],
                totals=grid_dict["totals"],
            )

        records.append(
            ValidationRecord(
                stem=stem,
                predicted=predicted,
                ground_truth=ground_truth,
                image_path=image_path,
            )
        )

    return records


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def run_validation(
    records: list[ValidationRecord],
    tolerance: float = 0.005,
) -> EvaluationReport:
    """Compare every prediction against its ground truth.

    Reuses :func:`weather_doc_extractor.evaluate.compare_grids` so that
    metrics are identical to those from the Azure evaluation pipeline.

    Parameters
    ----------
    records:
        Validation records as returned by :func:`load_extraction_results`.
    tolerance:
        Absolute tolerance in inches for a numeric match.

    Returns
    -------
    EvaluationReport
        Aggregate and per-record accuracy statistics.
    """
    comparisons: list[GridComparison] = []
    for rec in records:
        cmp = compare_grids(rec.stem, rec.predicted, rec.ground_truth, tolerance)
        comparisons.append(cmp)

    report = EvaluationReport(comparisons=comparisons)
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COL_WIDTH = {
    "stem": 48,
    "accuracy": 10,
    "fp": 5,
    "fn": 5,
    "wrong": 6,
    "failed": 8,
}


def print_summary(report: EvaluationReport) -> None:
    """Print a formatted per-record table and aggregate statistics to stdout."""
    header = (
        f"{'Stem':<{_COL_WIDTH['stem']}}"
        f"{'Accuracy':>{_COL_WIDTH['accuracy']}}"
        f"{'FP':>{_COL_WIDTH['fp']}}"
        f"{'FN':>{_COL_WIDTH['fn']}}"
        f"{'Wrong':>{_COL_WIDTH['wrong']}}"
        f"{'Failed':>{_COL_WIDTH['failed']}}"
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    for c in report.comparisons:
        if c.parse_failed:
            accuracy_str = "FAILED"
        else:
            accuracy_str = f"{c.accuracy:.1%}"

        print(
            f"{c.stem:<{_COL_WIDTH['stem']}}"
            f"{accuracy_str:>{_COL_WIDTH['accuracy']}}"
            f"{c.false_positive:>{_COL_WIDTH['fp']}}"
            f"{c.false_negative:>{_COL_WIDTH['fn']}}"
            f"{c.value_wrong:>{_COL_WIDTH['wrong']}}"
            f"{'yes' if c.parse_failed else '':>{_COL_WIDTH['failed']}}"
        )

    print(separator)
    s = report.summary()
    macro_acc_str = f"{s['macro_accuracy']:.1%}"
    print(
        f"{'AGGREGATE':<{_COL_WIDTH['stem']}}"
        f"{macro_acc_str:>{_COL_WIDTH['accuracy']}}"
        f"{s['total_false_positive']:>{_COL_WIDTH['fp']}}"
        f"{s['total_false_negative']:>{_COL_WIDTH['fn']}}"
        f"{s['total_value_wrong']:>{_COL_WIDTH['wrong']}}"
        f"{s['n_failed_parse']:>{_COL_WIDTH['failed']}}"
    )
    print(separator)
    print(
        f"\nImages: {s['n_images']}  |  "
        f"Macro accuracy: {s['macro_accuracy']:.1%}  |  "
        f"Micro accuracy: {s['micro_accuracy']:.1%}  |  "
        f"Parse failures: {s['n_failed_parse']}"
    )


def save_report(report: EvaluationReport, path: Path) -> Path:
    """Write the full JSON validation report to *path*.

    Returns the resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": report.summary(),
        "comparisons": [c.summary() for c in report.comparisons],
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def generate_figures(
    records: list[ValidationRecord],
    report: EvaluationReport,
    output_dir: Path,
    model_name: str | None = None,
    tolerance: float = 0.005,
) -> list[Path]:
    """Save one three-panel comparison figure per validation record.

    Parameters
    ----------
    records:
        Validation records (must match the comparisons in *report* by order).
    report:
        Evaluation report returned by :func:`run_validation`.
    output_dir:
        Directory to write PNG figures.  Created if it does not exist.
    model_name:
        Shown as the predicted-panel title in each figure.
    tolerance:
        Absolute tolerance in inches for the blue/red colour coding.

    Returns
    -------
    list[Path]
        Paths of the saved figure files.
    """
    from weather_doc_extractor.visualize import save_comparison_figure

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a quick lookup from stem → GridComparison for the title annotation.
    comparison_by_stem = {c.stem: c for c in report.comparisons}

    saved: list[Path] = []
    for rec in records:
        cmp = comparison_by_stem.get(rec.stem)
        if cmp is not None and not cmp.parse_failed:
            acc_str = f"  |  acc {cmp.accuracy:.1%}"
        elif cmp is not None and cmp.parse_failed:
            acc_str = "  |  [parse failed]"
        else:
            acc_str = ""

        title = rec.stem + acc_str
        out_path = output_dir / f"{rec.stem}_comparison.png"
        saved_path = save_comparison_figure(
            rec.image_path,
            rec.predicted,
            rec.ground_truth,
            output_path=out_path,
            title=title,
            tolerance=tolerance,
            model_name=model_name,
        )
        saved.append(saved_path)
        print(f"  Figure: {saved_path}")

    return saved
