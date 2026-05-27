"""Static constants for synthetic daily-rainfall document generation."""

from __future__ import annotations

from functools import lru_cache

from matplotlib import font_manager

MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

# Days per month in a non-leap year
DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# County names used to generate fake station metadata
COUNTIES = [
    "Anglesey",
    "Bedfordshire",
    "Berkshire",
    "Buckinghamshire",
    "Cambridgeshire",
    "Cheshire",
    "Cornwall",
    "Cumberland",
    "Derbyshire",
    "Devon",
    "Dorset",
    "Durham",
    "Essex",
    "Flintshire",
    "Gloucestershire",
    "Hampshire",
    "Herefordshire",
    "Hertfordshire",
    "Huntingdonshire",
    "Kent",
    "Lancashire",
    "Leicestershire",
    "Lincolnshire",
    "Middlesex",
    "Norfolk",
    "Northamptonshire",
    "Northumberland",
    "Nottinghamshire",
    "Oxfordshire",
    "Rutland",
    "Shropshire",
    "Somerset",
    "Staffordshire",
    "Suffolk",
    "Surrey",
    "Sussex",
    "Warwickshire",
    "Westmorland",
    "Wiltshire",
    "Worcestershire",
    "Yorkshire",
]

# Font families available in matplotlib on most Linux systems.
# DejaVu variants are always present (bundled with matplotlib).
FONT_FAMILIES = [
    "DejaVu Sans",
    "DejaVu Serif",
    "DejaVu Sans Mono",
    "Liberation Sans",
    "Liberation Serif",
    "Liberation Mono",
    "FreeSans",
    "FreeSerif",
    "FreeMono",
    "Nimbus Roman",
    "Nimbus Sans",
    "URW Bookman",
    "Century Schoolbook L",
    "Bitstream Charter",
    "monospace",
    "serif",
    "sans-serif",
]


@lru_cache(maxsize=1)
def get_available_font_families() -> list[str]:
    """Return configured font families that are available on this machine."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    available = [f for f in FONT_FAMILIES if f in installed]

    # Generic family names are always valid in matplotlib.
    for generic in ["monospace", "serif", "sans-serif"]:
        if generic in FONT_FAMILIES and generic not in available:
            available.append(generic)

    # Defensive fallback: matplotlib ships with DejaVu Sans.
    if not available:
        return ["DejaVu Sans"]
    return available
