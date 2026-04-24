"""Evaluation utilities for comparing extracted grids against ground truth.

Public API
----------
compare_grids(predicted, ground_truth, tolerance)
    Compare two DailyRainfallGrid objects cell by cell.

score_record(record, config, tolerance)
    Run inference on one record and return a GridComparison.

evaluate_dataset(records, config, tolerance)
    Run inference on all paired records and aggregate metrics.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Sequence

from weather_doc_extractor.schemas import DailyRainfallGrid, DailyRainfallRecord

_MONTHS = 12
_DAY_KEYS = [f"Day {i}" for i in range(1, 32)]

# Default tolerance for numeric comparison (inches)
DEFAULT_TOLERANCE = 0.005


@dataclass
class CellResult:
    key: str       # e.g. "Day 3" or "Totals"
    month: int     # 1-based
    truth: float | None
    predicted: float | None
    match: bool


@dataclass
class GridComparison:
    """Full comparison between a predicted and a ground-truth grid."""

    stem: str
    cells: list[CellResult] = field(default_factory=list)
    parse_failed: bool = False

    # ---- derived counts (populated by compare_grids) ----
    total_cells: int = 0
    exact_matches: int = 0       # both None, or numeric within tolerance
    null_correct: int = 0        # truth None AND predicted None
    false_positive: int = 0      # truth None, predicted non-None
    false_negative: int = 0      # truth non-None, predicted None
    value_close: int = 0         # both non-None and within tolerance
    value_wrong: int = 0         # both non-None but outside tolerance

    @property
    def accuracy(self) -> float:
        if self.total_cells == 0:
            return 0.0
        return self.exact_matches / self.total_cells

    def summary(self) -> dict[str, object]:
        return {
            "stem": self.stem,
            "parse_failed": self.parse_failed,
            "total_cells": self.total_cells,
            "accuracy": round(self.accuracy, 4),
            "exact_matches": self.exact_matches,
            "null_correct": self.null_correct,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "value_close": self.value_close,
            "value_wrong": self.value_wrong,
        }


@dataclass
class EvaluationReport:
    """Aggregated metrics across multiple GridComparisons."""

    comparisons: list[GridComparison] = field(default_factory=list)

    @property
    def n_images(self) -> int:
        return len(self.comparisons)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.comparisons if c.parse_failed)

    @property
    def total_cells(self) -> int:
        return sum(c.total_cells for c in self.comparisons)

    @property
    def total_exact(self) -> int:
        return sum(c.exact_matches for c in self.comparisons)

    @property
    def macro_accuracy(self) -> float:
        """Mean per-image accuracy (equally weighted)."""
        parsed = [c for c in self.comparisons if not c.parse_failed]
        if not parsed:
            return 0.0
        return sum(c.accuracy for c in parsed) / len(parsed)

    @property
    def micro_accuracy(self) -> float:
        """Accuracy across all cells pooled together."""
        if self.total_cells == 0:
            return 0.0
        return self.total_exact / self.total_cells

    def summary(self) -> dict[str, object]:
        return {
            "n_images": self.n_images,
            "n_failed_parse": self.n_failed,
            "total_cells": self.total_cells,
            "macro_accuracy": round(self.macro_accuracy, 4),
            "micro_accuracy": round(self.micro_accuracy, 4),
            "total_exact_matches": self.total_exact,
            "total_false_positive": sum(c.false_positive for c in self.comparisons),
            "total_false_negative": sum(c.false_negative for c in self.comparisons),
            "total_value_wrong": sum(c.value_wrong for c in self.comparisons),
        }


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------


def compare_grids(
    stem: str,
    predicted: DailyRainfallGrid | None,
    ground_truth: DailyRainfallGrid,
    tolerance: float = DEFAULT_TOLERANCE,
) -> GridComparison:
    """Compare *predicted* against *ground_truth* cell by cell.

    If *predicted* is ``None`` (parse failed) the comparison is marked as
    failed and all counts remain zero.
    """
    result = GridComparison(stem=stem)

    if predicted is None:
        result.parse_failed = True
        return result

    all_keys = _DAY_KEYS + ["Totals"]

    for key in all_keys:
        if key == "Totals":
            truth_row = ground_truth.totals
            pred_row = predicted.totals
        else:
            truth_row = ground_truth.days.get(key, [None] * _MONTHS)
            pred_row = predicted.days.get(key, [None] * _MONTHS)

        for month_idx in range(_MONTHS):
            t = truth_row[month_idx] if month_idx < len(truth_row) else None
            p = pred_row[month_idx] if month_idx < len(pred_row) else None

            match = _cells_match(t, p, tolerance)
            result.cells.append(CellResult(key=key, month=month_idx + 1, truth=t, predicted=p, match=match))
            result.total_cells += 1
            if match:
                result.exact_matches += 1
            if t is None and p is None:
                result.null_correct += 1
            elif t is None and p is not None:
                result.false_positive += 1
            elif t is not None and p is None:
                result.false_negative += 1
            else:
                if match:
                    result.value_close += 1
                else:
                    result.value_wrong += 1

    return result


def _cells_match(
    truth: float | None,
    predicted: float | None,
    tolerance: float,
) -> bool:
    if truth is None and predicted is None:
        return True
    if truth is None or predicted is None:
        return False
    return abs(truth - predicted) <= tolerance


# ---------------------------------------------------------------------------
# Per-record and dataset evaluation
# ---------------------------------------------------------------------------


def score_record(
    record: DailyRainfallRecord,
    config,  # AppConfig — avoid circular import
    tolerance: float = DEFAULT_TOLERANCE,
) -> GridComparison:
    """Run inference on *record* and compare against its ground truth grid."""
    from weather_doc_extractor.inference import extract_grid

    predicted, _ = extract_grid(record.image_path, config.model)
    return compare_grids(record.stem, predicted, record.grid, tolerance)  # type: ignore[arg-type]


def evaluate_dataset(
    records: Sequence[DailyRainfallRecord],
    config,
    tolerance: float = DEFAULT_TOLERANCE,
    limit: int | None = None,
) -> EvaluationReport:
    """Evaluate the model over all paired *records*.

    Parameters
    ----------
    records:
        A sequence of :class:`DailyRainfallRecord` objects.  Unpaired records
        (``grid is None``) are silently skipped.
    config:
        The :class:`~weather_doc_extractor.config.AppConfig` instance.
    tolerance:
        Numeric tolerance in inches for a cell to count as correct.
    limit:
        If set, evaluate at most this many paired records.
    """
    paired = [r for r in records if r.grid is not None]
    if limit is not None:
        paired = paired[:limit]

    report = EvaluationReport()
    for i, record in enumerate(paired, 1):
        print(f"  [{i}/{len(paired)}] {record.stem} …", flush=True)
        comparison = score_record(record, config, tolerance)
        report.comparisons.append(comparison)
        acc = f"{comparison.accuracy:.1%}" if not comparison.parse_failed else "PARSE FAILED"
        print(f"    accuracy: {acc}")

    return report
