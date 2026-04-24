"""Tests for visualize.py — no model or display required."""

from __future__ import annotations

from pathlib import Path

import pytest

from weather_doc_extractor.schemas import DailyRainfallGrid
from weather_doc_extractor.visualize import (
    _cell_text,
    _diff_text_colours,
    _fmt,
    make_figure,
    save_figure,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid(base: float = 0.1) -> DailyRainfallGrid:
    days = {f"Day {i}": [round(base * i + 0.01 * j, 3) for j in range(12)]
            for i in range(1, 32)}
    totals = [round(base * j, 2) for j in range(12)]
    return DailyRainfallGrid(days=days, totals=totals)


def _make_image(tmp_path: Path) -> Path:
    from PIL import Image as PILImage
    img_path = tmp_path / "test.jpg"
    PILImage.new("RGB", (8, 8), color=(200, 200, 200)).save(str(img_path))
    return img_path


# ---------------------------------------------------------------------------
# _fmt
# ---------------------------------------------------------------------------


class TestFmt:
    def test_none_returns_dash(self):
        assert _fmt(None) == "—"

    def test_integer_value(self):
        assert _fmt(1.0) == "1"

    def test_decimal_value(self):
        assert _fmt(0.37) == "0.37"

    def test_zero(self):
        assert _fmt(0.0) == "0"


# ---------------------------------------------------------------------------
# _cell_text
# ---------------------------------------------------------------------------


class TestCellText:
    def test_has_32_rows(self):
        grid = _make_grid()
        rows = _cell_text(grid)
        assert len(rows) == 32  # 31 day rows + Totals

    def test_each_row_has_12_cols(self):
        grid = _make_grid()
        for row in _cell_text(grid):
            assert len(row) == 12

    def test_totals_row_last(self):
        grid = _make_grid()
        rows = _cell_text(grid)
        # Totals row values come from grid.totals
        expected = [_fmt(v) for v in grid.totals]
        assert rows[-1] == expected


# ---------------------------------------------------------------------------
# _diff_text_colours
# ---------------------------------------------------------------------------


class TestDiffTextColours:
    def test_returns_32_rows_of_12(self):
        g = _make_grid()
        colours = _diff_text_colours(g, g)
        assert len(colours) == 32
        assert all(len(r) == 12 for r in colours)

    def test_perfect_match_is_blue(self):
        g = _make_grid()
        colours = _diff_text_colours(g, g)
        for row in colours:
            for c in row:
                assert c == "blue"

    def test_mismatch_is_red(self):
        pred = _make_grid(base=0.1)
        gt = _make_grid(base=9.9)
        colours = _diff_text_colours(pred, gt)
        # Day 1, Jan: pred=0.1, gt=9.9 → clear mismatch
        assert colours[0][0] == "red"

    def test_none_pred_is_red(self):
        pred = DailyRainfallGrid(
            days={f"Day {i}": [None] * 12 for i in range(1, 32)},
            totals=[None] * 12,
        )
        gt = _make_grid()
        colours = _diff_text_colours(pred, gt)
        assert colours[0][0] == "red"

    def test_both_none_is_blue(self):
        pred = DailyRainfallGrid(
            days={f"Day {i}": [None] * 12 for i in range(1, 32)},
            totals=[None] * 12,
        )
        gt = DailyRainfallGrid(
            days={f"Day {i}": [None] * 12 for i in range(1, 32)},
            totals=[None] * 12,
        )
        colours = _diff_text_colours(pred, gt)
        assert colours[0][0] == "blue"


# ---------------------------------------------------------------------------
# make_figure
# ---------------------------------------------------------------------------


class TestMakeFigure:
    def test_returns_figure(self, tmp_path):
        import matplotlib.figure
        grid = _make_grid()
        fig = make_figure(_make_image(tmp_path), grid)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figure_with_ground_truth(self, tmp_path):
        import matplotlib.figure
        grid = _make_grid()
        fig = make_figure(_make_image(tmp_path), grid, ground_truth=_make_grid(base=0.2))
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_title(self, tmp_path):
        grid = _make_grid()
        fig = make_figure(_make_image(tmp_path), grid, title="My title")
        assert fig.texts[0].get_text() == "My title"


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------


class TestSaveFigure:
    def test_file_created(self, tmp_path):
        grid = _make_grid()
        out = tmp_path / "out.png"
        result = save_figure(_make_image(tmp_path), grid, out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path):
        grid = _make_grid()
        out = tmp_path / "deep" / "nested" / "fig.png"
        save_figure(_make_image(tmp_path), grid, out)
        assert out.exists()

    def test_with_ground_truth(self, tmp_path):
        grid = _make_grid()
        out = tmp_path / "diff.png"
        save_figure(_make_image(tmp_path), grid, out, ground_truth=_make_grid(base=0.5))
        assert out.exists()
