"""Page geometry and cell coordinate helpers.

All coordinates are normalised to [0, 1] with (0, 0) at the bottom-left
of the matplotlib figure, matching matplotlib's default axes convention.

Grid layout (portrait page):

  ┌─────────────────────────────────────────────┐  ← 1.0
  │              PAGE HEADER / TITLE            │
  ├────┬──────┬──────┬─────┬──────┬──────┬──────┤  ← GRID_TOP
  │    │ Jan  │ Feb  │ … │ Nov  │ Dec  │
  ├────┼──────┼──────┼─────┼──────┼──────┼──────┤
  │  1 │      │      │   │      │      │
  │  2 │ 0.12 │      │   │      │ 0.45 │
  │  … │      │      │   │      │      │
  │ 31 │      │      │   │      │      │
  ├────┼──────┼──────┼─────┼──────┼──────┼──────┤
  │Tot │ 3.21 │ 1.04 │   │ 5.61 │ 8.14 │
  └────┴──────┴──────┴─────┴──────┴──────┴──────┘  ← GRID_BOTTOM
  ↑                                             ↑
GRID_LEFT                                  GRID_RIGHT

Column 0   : day-label column
Columns 1–12 : Jan … Dec
Row 0      : month-name header
Rows 1–31  : Day 1 … Day 31
Row 32     : Totals
"""

from __future__ import annotations

# ── Page dimensions ──────────────────────────────────────────────────────────
PAGE_WIDTH_IN: float = 14.0
PAGE_HEIGHT_IN: float = 18.0
DPI: int = 100

# ── Grid bounding box (normalised page coordinates) ──────────────────────────
GRID_LEFT: float = 0.03
GRID_RIGHT: float = 0.99
GRID_TOP: float = 0.91
GRID_BOTTOM: float = 0.02

GRID_W: float = GRID_RIGHT - GRID_LEFT  # 0.96
GRID_H: float = GRID_TOP - GRID_BOTTOM  # 0.89

# ── Grid structure ────────────────────────────────────────────────────────────
N_LABEL_COLS: int = 1  # leftmost column holds day numbers / "Total"
N_DATA_COLS: int = 12  # one per month
N_COLS: int = N_LABEL_COLS + N_DATA_COLS  # 13

N_HEADER_ROWS: int = 1  # top row holds month names
N_DATA_ROWS: int = 31  # Day 1 … Day 31
N_TOTAL_ROWS: int = 1  # bottom row holds monthly totals
N_ROWS: int = N_HEADER_ROWS + N_DATA_ROWS + N_TOTAL_ROWS  # 33

CELL_W: float = GRID_W / N_COLS  # ≈ 0.0738
CELL_H: float = GRID_H / N_ROWS  # ≈ 0.0270


# ── Coordinate helpers ────────────────────────────────────────────────────────


def grid_line_x(col: int) -> float:
    """X position of the *left* edge of *col* (0 = leftmost grid line)."""
    return GRID_LEFT + col * CELL_W


def grid_line_y(row: int) -> float:
    """Y position of the *top* edge of *row* (0 = topmost grid line)."""
    return GRID_TOP - row * CELL_H


def cell_center(row: int, col: int) -> tuple[float, float]:
    """Centre (x, y) of the cell at (*row*, *col*).

    row 0          : month-name header
    rows 1 … 31    : Day 1 … Day 31
    row 32         : Totals
    col 0          : day-label column
    cols 1 … 12    : Jan … Dec
    """
    x = GRID_LEFT + (col + 0.5) * CELL_W
    y = GRID_TOP - (row + 0.5) * CELL_H
    return x, y


def header_center() -> tuple[float, float]:
    """Centre of the page-header area (above the grid)."""
    x = (GRID_LEFT + GRID_RIGHT) / 2
    y = (GRID_TOP + 1.0) / 2
    return x, y
