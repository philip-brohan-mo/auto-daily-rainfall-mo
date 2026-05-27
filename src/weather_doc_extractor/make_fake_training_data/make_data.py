"""Generate random daily-rainfall data records.

The output format mirrors the real transcription JSONs: a dict whose keys are
``"Day 1"`` … ``"Day 31"`` plus ``"Totals"``, and whose values are 12-element
lists (Jan–Dec) of strings ``"0.37"`` or ``"null"``.
"""

from __future__ import annotations

import numpy as np

from .constants import DAYS_PER_MONTH


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _days_in_month(year: int) -> list[int]:
    days = list(DAYS_PER_MONTH)
    if _is_leap(year):
        days[1] = 29
    return days


def _random_value(rng: np.random.Generator, null_prob: float = 0.45) -> str:
    """Return a rainfall string ``"D.DD"`` or ``"null"``.

    Most UK daily values are below 0.5 inch; heavier falls are rare.
    We model them with an exponential distribution (mean ≈ 0.18 inch).
    """
    if rng.random() < null_prob:
        return "null"
    # Exponential with mean 0.18"; capped at 3.0"
    val = min(float(rng.exponential(0.18)), 3.0)
    # Round to 2 dp; ensure at least 0.01
    val = max(0.01, round(val, 2))
    return f"{val:.2f}"


def generate_data(
    year: int,
    rng: np.random.Generator | None = None,
) -> dict[str, list[str]]:
    """Return a fake daily-rainfall record for *year*.

    Parameters
    ----------
    year:
        Calendar year (e.g. 1875).  Used for leap-year detection only;
        the returned dict does not store the year itself.
    rng:
        NumPy random generator.  A new default generator is created if
        *None* is supplied.

    Returns
    -------
    dict
        Keys ``"Day 1"`` … ``"Day 31"`` and ``"Totals"``, each mapped to a
        12-element list of strings (``"0.37"`` or ``"null"``).
    """
    if rng is None:
        rng = np.random.default_rng()

    days_in_month = _days_in_month(year)
    monthly_totals = [0.0] * 12
    grid: dict[str, list[str]] = {}

    for day in range(1, 32):
        row: list[str] = []
        for month in range(12):
            if day > days_in_month[month]:
                # Day doesn't exist in this month (e.g. 30 Feb)
                row.append("null")
            else:
                val_str = _random_value(rng)
                row.append(val_str)
                if val_str != "null":
                    monthly_totals[month] += float(val_str)
        grid[f"Day {day}"] = row

    grid["Totals"] = [f"{t:.2f}" for t in monthly_totals]
    return grid


def decade_range(year: int) -> str:
    """Return the decade string for *year*, e.g. 1875 → ``"1871-1880"``."""
    start = ((year - 1) // 10) * 10 + 1
    return f"{start}-{start + 9}"


def make_stem(year: int, county: str, station_id: int) -> str:
    """Build a filename stem compatible with ``parse_stem()``.

    Example: ``"DRain_1871-1880_Cornwall-59"``
    """
    decade = decade_range(year)
    return f"DRain_{decade}_{county}-{station_id}"
