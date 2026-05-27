"""Render a fake daily-rainfall document as a matplotlib Figure.

The rendered image mimics a pre-printed form with typewritten rainfall
values: off-white paper, dark grid lines, month names across the top,
day numbers down the left, and individual cell values.
"""

from __future__ import annotations

import random

import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from . import geometry as geo
from . import jitter as jtr
from .constants import MONTHS, get_available_font_families

matplotlib.use("Agg")


# ── Internal helpers ──────────────────────────────────────────────────────────


def _ink(g: float) -> tuple[float, float, float]:
    """Return an RGB tuple for a gray level *g* (0 = black, 1 = white)."""
    return (g, g, g)


def _tx(x: float, scale_x: float, shift_x: float) -> float:
    """Apply horizontal table scaling and translation around grid centre."""
    cx = (geo.GRID_LEFT + geo.GRID_RIGHT) / 2
    return cx + (x - cx) * scale_x + shift_x


def _ty(y: float, scale_y: float, shift_y: float) -> float:
    """Apply vertical table scaling and translation around grid centre."""
    cy = (geo.GRID_TOP + geo.GRID_BOTTOM) / 2
    return cy + (y - cy) * scale_y + shift_y


def _txy(
    x: float,
    y: float,
    scale_x: float,
    scale_y: float,
    shift_x: float,
    shift_y: float,
) -> tuple[float, float]:
    """Apply 2D table transform (scale + translation)."""
    return _tx(x, scale_x, shift_x), _ty(y, scale_y, shift_y)


def _draw_background(ax: plt.Axes, rng: np.random.Generator) -> None:
    """Fill the page with a slightly jittered off-white background."""
    bg = jtr.gray(0.94, rng, sigma=0.015)
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, color=_ink(bg), zorder=0))


def _draw_grid_lines(
    ax: plt.Axes,
    rng: np.random.Generator,
    ink: float,
    jitter_pts: float,
    scale_x: float,
    scale_y: float,
    shift_x: float,
    shift_y: float,
) -> None:
    """Draw all vertical and horizontal grid lines with slight positional jitter."""
    border_lw = jtr.linewidth(1.6, rng, 0.15)
    inner_lw = jtr.linewidth(0.8, rng, 0.08)
    col = _ink(ink)

    # Vertical lines (one per column boundary, plus right edge)
    for ci in range(geo.N_COLS + 1):
        base_x = _tx(geo.grid_line_x(ci), scale_x, shift_x)
        base_y0 = _ty(geo.GRID_BOTTOM, scale_y, shift_y)
        base_y1 = _ty(geo.GRID_TOP, scale_y, shift_y)
        x = jtr.pos(base_x, rng, jitter_pts)
        y0 = jtr.pos(base_y0, rng, jitter_pts)
        y1 = jtr.pos(base_y1, rng, jitter_pts)
        lw = border_lw if ci in (0, geo.N_COLS) else inner_lw
        ax.add_line(mlines.Line2D([x, x], [y0, y1], color=col, linewidth=lw, zorder=2))

    # Horizontal lines (one per row boundary, plus bottom edge)
    for ri in range(geo.N_ROWS + 1):
        base_y = _ty(geo.grid_line_y(ri), scale_y, shift_y)
        base_x0 = _tx(geo.GRID_LEFT, scale_x, shift_x)
        base_x1 = _tx(geo.GRID_RIGHT, scale_x, shift_x)
        y = jtr.pos(base_y, rng, jitter_pts)
        x0 = jtr.pos(base_x0, rng, jitter_pts)
        x1 = jtr.pos(base_x1, rng, jitter_pts)
        # Draw the line separating header/data and data/totals slightly heavier
        lw = border_lw if ri in (0, 1, geo.N_ROWS - 1, geo.N_ROWS) else inner_lw
        ax.add_line(mlines.Line2D([x0, x1], [y, y], color=col, linewidth=lw, zorder=2))


def _draw_month_headers(
    ax: plt.Axes,
    rng: np.random.Generator,
    ink: float,
    font_family: str,
    font_size: float,
    jitter_pts: float,
    scale_x: float,
    scale_y: float,
    shift_x: float,
    shift_y: float,
) -> None:
    """Render month abbreviations in the header row (row 0, cols 1–12)."""
    col = _ink(jtr.gray(ink, rng, 0.03))
    for mi, month in enumerate(MONTHS):
        cx, cy = geo.cell_center(0, mi + 1)  # col 0 is label column
        cx, cy = _txy(cx, cy, scale_x, scale_y, shift_x, shift_y)
        ax.text(
            jtr.pos(cx, rng, jitter_pts),
            jtr.pos(cy, rng, jitter_pts),
            month,
            ha="center",
            va="center",
            fontfamily=font_family,
            fontsize=jtr.size(font_size, rng, 0.4),
            rotation=jtr.rotation(rng, 0.4),
            color=col,
            zorder=3,
        )


def _draw_day_labels(
    ax: plt.Axes,
    rng: np.random.Generator,
    ink: float,
    font_family: str,
    font_size: float,
    jitter_pts: float,
    scale_x: float,
    scale_y: float,
    shift_x: float,
    shift_y: float,
) -> None:
    """Render day numbers 1–31 and 'Total' in the label column (col 0)."""
    col = _ink(jtr.gray(ink, rng, 0.03))
    label_size = jtr.size(font_size * 0.9, rng, 0.3)

    # Day numbers: rows 1–31
    for day in range(1, 32):
        cx, cy = geo.cell_center(day, 0)
        cx, cy = _txy(cx, cy, scale_x, scale_y, shift_x, shift_y)
        ax.text(
            jtr.pos(cx, rng, jitter_pts),
            jtr.pos(cy, rng, jitter_pts),
            str(day),
            ha="center",
            va="center",
            fontfamily=font_family,
            fontsize=jtr.size(label_size, rng, 0.2),
            rotation=jtr.rotation(rng, 0.3),
            color=col,
            zorder=3,
        )

    # "Total" label in the last row
    cx, cy = geo.cell_center(geo.N_ROWS - 1, 0)
    cx, cy = _txy(cx, cy, scale_x, scale_y, shift_x, shift_y)
    ax.text(
        jtr.pos(cx, rng, jitter_pts),
        jtr.pos(cy, rng, jitter_pts),
        "Total",
        ha="center",
        va="center",
        fontfamily=font_family,
        fontsize=jtr.size(font_size * 0.8, rng, 0.3),
        rotation=jtr.rotation(rng, 0.3),
        color=col,
        zorder=3,
    )


def _fmt_value(val_str: str, rng: np.random.Generator) -> str:
    """Format a value string for rendering; occasionally drop leading zero."""
    if val_str == "null":
        return ""
    # ~20% of the time render ".37" instead of "0.37" (older style)
    if val_str.startswith("0.") and rng.random() < 0.2:
        return val_str[1:]  # drop the leading "0"
    return val_str


def _draw_data_values(
    ax: plt.Axes,
    data: dict[str, list[str]],
    rng: np.random.Generator,
    ink: float,
    font_family: str,
    font_size: float,
    jitter_pts: float,
    scale_x: float,
    scale_y: float,
    shift_x: float,
    shift_y: float,
) -> None:
    """Render daily rainfall values in their grid cells (rows 1–31, cols 1–12)."""
    for day in range(1, 32):
        row_values = data.get(f"Day {day}", ["null"] * 12)
        for mi, val_str in enumerate(row_values):
            text = _fmt_value(val_str, rng)
            if not text:
                continue
            cx, cy = geo.cell_center(day, mi + 1)
            cx, cy = _txy(cx, cy, scale_x, scale_y, shift_x, shift_y)
            col = _ink(jtr.gray(ink * 0.8, rng, 0.04))
            ax.text(
                jtr.pos(cx, rng, jitter_pts),
                jtr.pos(cy, rng, jitter_pts),
                text,
                ha="center",
                va="center",
                fontfamily=font_family,
                fontsize=jtr.size(font_size * 0.85, rng, 0.3),
                rotation=jtr.rotation(rng, 0.25),
                color=col,
                zorder=3,
            )


def _draw_totals(
    ax: plt.Axes,
    totals: list[str],
    rng: np.random.Generator,
    ink: float,
    font_family: str,
    font_size: float,
    jitter_pts: float,
    scale_x: float,
    scale_y: float,
    shift_x: float,
    shift_y: float,
) -> None:
    """Render monthly totals in the bottom row (row N_ROWS-1, cols 1–12)."""
    row = geo.N_ROWS - 1
    for mi, val_str in enumerate(totals):
        text = _fmt_value(val_str, rng)
        if not text:
            continue
        cx, cy = geo.cell_center(row, mi + 1)
        cx, cy = _txy(cx, cy, scale_x, scale_y, shift_x, shift_y)
        col = _ink(jtr.gray(ink * 0.8, rng, 0.04))
        ax.text(
            jtr.pos(cx, rng, jitter_pts),
            jtr.pos(cy, rng, jitter_pts),
            text,
            ha="center",
            va="center",
            fontfamily=font_family,
            fontsize=jtr.size(font_size * 0.85, rng, 0.3),
            rotation=jtr.rotation(rng, 0.25),
            color=col,
            zorder=3,
        )


def _draw_page_header(
    ax: plt.Axes,
    year: int,
    county: str,
    station_id: int,
    rng: np.random.Generator,
    ink: float,
    font_family: str,
    font_size: float,
    shift_x: float,
    shift_y: float,
) -> None:
    """Render the station title block above the grid."""
    hx, hy = geo.header_center()
    hx += shift_x
    hy += shift_y * 0.6
    col = _ink(jtr.gray(ink, rng, 0.03))

    # Main title line
    ax.text(
        hx,
        hy + 0.025,
        f"DAILY RAINFALL  {year}",
        ha="center",
        va="center",
        fontfamily=font_family,
        fontsize=jtr.size(font_size * 1.3, rng, 0.5),
        rotation=jtr.rotation(rng, 0.2),
        color=col,
        fontweight="bold",
        zorder=3,
    )
    # Sub-title line: county + station number
    ax.text(
        hx,
        hy - 0.020,
        f"{county}   No. {station_id}",
        ha="center",
        va="center",
        fontfamily=font_family,
        fontsize=jtr.size(font_size * 1.0, rng, 0.4),
        rotation=jtr.rotation(rng, 0.2),
        color=col,
        zorder=3,
    )


# ── Public API ────────────────────────────────────────────────────────────────


def draw_document(
    data: dict[str, list[str]],
    year: int,
    county: str,
    station_id: int,
    *,
    font_family: str | None = None,
    font_size: float = 20.0,
    jitter_grid_points: float = 0.0008,
    table_scale_x: float | None = None,
    table_scale_y: float | None = None,
    table_shift_x: float | None = None,
    table_shift_y: float | None = None,
    rng: np.random.Generator | None = None,
) -> plt.Figure:
    """Render *data* as a synthetic historical daily-rainfall document.

    Parameters
    ----------
    data:
        Dict produced by :func:`make_data.generate_data`.
    year:
        Calendar year shown in the page header.
    county:
        County name shown in the page header.
    station_id:
        Numeric station ID shown in the page header.
    font_family:
        Matplotlib font family string.  Chosen at random from
        :data:`constants.FONT_FAMILIES` when *None*.
    font_size:
        Base font size in points; individual elements are scaled from this.
    jitter_grid_points:
        Std-dev of position jitter applied to grid lines and text (normalised
        page coordinates).
    table_scale_x, table_scale_y:
        Optional table scale factors. If omitted, random values are drawn
        per image from [0.95, 1.0] (slight shrink only).
    table_shift_x, table_shift_y:
        Optional table offsets in normalised page coordinates. If omitted,
        random values are drawn per image from small ranges.
    rng:
        NumPy random generator.  A new default generator is used when *None*.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.  Call ``fig.savefig(path, ...)`` to persist it.
    """
    if rng is None:
        rng = np.random.default_rng()
    if font_family is None:
        font_family = random.choice(get_available_font_families())
    if table_scale_x is None:
        table_scale_x = float(rng.uniform(0.95, 1.0))
    if table_scale_y is None:
        table_scale_y = float(rng.uniform(0.95, 1.0))
    if table_shift_x is None:
        table_shift_x = float(rng.uniform(-0.015, 0.015))
    if table_shift_y is None:
        table_shift_y = float(rng.uniform(-0.02, 0.02))

    ink = jtr.gray(0.08, rng, sigma=0.04)  # near-black ink colour

    fig = plt.figure(
        figsize=(geo.PAGE_WIDTH_IN, geo.PAGE_HEIGHT_IN),
        dpi=geo.DPI,
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    _draw_background(ax, rng)
    _draw_grid_lines(
        ax,
        rng,
        ink,
        jitter_grid_points,
        table_scale_x,
        table_scale_y,
        table_shift_x,
        table_shift_y,
    )
    _draw_month_headers(
        ax,
        rng,
        ink,
        font_family,
        font_size,
        jitter_grid_points,
        table_scale_x,
        table_scale_y,
        table_shift_x,
        table_shift_y,
    )
    _draw_day_labels(
        ax,
        rng,
        ink,
        font_family,
        font_size,
        jitter_grid_points,
        table_scale_x,
        table_scale_y,
        table_shift_x,
        table_shift_y,
    )
    _draw_data_values(
        ax,
        data,
        rng,
        ink,
        font_family,
        font_size,
        jitter_grid_points,
        table_scale_x,
        table_scale_y,
        table_shift_x,
        table_shift_y,
    )
    _draw_totals(
        ax,
        data.get("Totals", ["null"] * 12),
        rng,
        ink,
        font_family,
        font_size,
        jitter_grid_points,
        table_scale_x,
        table_scale_y,
        table_shift_x,
        table_shift_y,
    )
    _draw_page_header(
        ax,
        year,
        county,
        station_id,
        rng,
        ink,
        font_family,
        font_size,
        table_shift_x,
        table_shift_y,
    )

    return fig
