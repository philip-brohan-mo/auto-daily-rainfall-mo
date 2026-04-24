import unittest

from weather_doc_extractor.evaluate import (
    DEFAULT_TOLERANCE,
    GridComparison,
    compare_grids,
)
from weather_doc_extractor.schemas import DailyRainfallGrid

MONTHS = 12
_ALL_DAYS = [f"Day {i}" for i in range(1, 32)]


def _make_grid(value: float | None = 0.10, total: float | None = None) -> DailyRainfallGrid:
    if total is None:
        total = value  # keep everything uniform by default
    days = {day: [value] * MONTHS for day in _ALL_DAYS}
    totals = [total] * MONTHS
    return DailyRainfallGrid(days=days, totals=totals)


class CompareGridsTests(unittest.TestCase):
    def test_perfect_match(self) -> None:
        grid = _make_grid(0.10)
        result = compare_grids("test", grid, grid)
        self.assertFalse(result.parse_failed)
        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.false_positive, 0)
        self.assertEqual(result.false_negative, 0)
        self.assertEqual(result.value_wrong, 0)

    def test_total_cell_count(self) -> None:
        grid = _make_grid(0.10)
        result = compare_grids("test", grid, grid)
        # 31 days + 1 totals row × 12 months
        self.assertEqual(result.total_cells, 32 * MONTHS)

    def test_parse_failed_when_predicted_none(self) -> None:
        truth = _make_grid(0.10)
        result = compare_grids("test", None, truth)
        self.assertTrue(result.parse_failed)
        self.assertEqual(result.total_cells, 0)

    def test_value_within_tolerance(self) -> None:
        truth = _make_grid(0.10)
        pred = _make_grid(0.10 + DEFAULT_TOLERANCE * 0.9)  # safely within tolerance
        result = compare_grids("test", pred, truth)
        self.assertEqual(result.accuracy, 1.0)

    def test_value_outside_tolerance(self) -> None:
        truth = _make_grid(0.10)
        pred = _make_grid(0.10 + DEFAULT_TOLERANCE + 0.001)
        result = compare_grids("test", pred, truth)
        self.assertEqual(result.accuracy, 0.0)
        self.assertEqual(result.value_wrong, 32 * MONTHS)

    def test_false_positive(self) -> None:
        truth = _make_grid(None)   # all nulls
        pred = _make_grid(0.10)    # all values
        result = compare_grids("test", pred, truth)
        self.assertEqual(result.false_positive, 32 * MONTHS)
        self.assertEqual(result.null_correct, 0)

    def test_false_negative(self) -> None:
        truth = _make_grid(0.10)
        pred = _make_grid(None)
        result = compare_grids("test", pred, truth)
        self.assertEqual(result.false_negative, 32 * MONTHS)

    def test_mixed_nulls(self) -> None:
        truth_days = {day: ([0.10] * 6 + [None] * 6) for day in _ALL_DAYS}
        pred_days = {day: ([0.10] * 6 + [None] * 6) for day in _ALL_DAYS}
        truth = DailyRainfallGrid(days=truth_days, totals=[None] * MONTHS)
        pred = DailyRainfallGrid(days=pred_days, totals=[None] * MONTHS)
        result = compare_grids("test", pred, truth)
        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.null_correct, 31 * 6 + MONTHS)  # 6 null months per day + all totals
        self.assertEqual(result.value_close, 31 * 6)

    def test_stem_stored(self) -> None:
        grid = _make_grid()
        result = compare_grids("DRain_1871-1880_Cornwall-59", grid, grid)
        self.assertEqual(result.stem, "DRain_1871-1880_Cornwall-59")

    def test_accuracy_property(self) -> None:
        truth = _make_grid(0.10)
        pred_days = {day: [0.10] * MONTHS for day in _ALL_DAYS}
        pred_days["Day 1"] = [0.99] * MONTHS  # all wrong
        pred = DailyRainfallGrid(days=pred_days, totals=[0.10] * MONTHS)  # totals match truth
        result = compare_grids("test", pred, truth)
        total = 32 * MONTHS
        wrong = MONTHS  # only Day 1's 12 cells
        self.assertAlmostEqual(result.accuracy, (total - wrong) / total)

    def test_summary_keys(self) -> None:
        grid = _make_grid()
        result = compare_grids("test", grid, grid)
        s = result.summary()
        for key in ("stem", "parse_failed", "total_cells", "accuracy",
                    "exact_matches", "null_correct", "false_positive",
                    "false_negative", "value_close", "value_wrong"):
            self.assertIn(key, s)

    def test_zero_accuracy_property_when_no_cells(self) -> None:
        c = GridComparison(stem="empty", parse_failed=True)
        self.assertEqual(c.accuracy, 0.0)


if __name__ == "__main__":
    unittest.main()
