from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class WeatherObservation:
    station_id: str | None = None
    observation_date: str | None = None
    observation_time: str | None = None
    temperature_c: float | None = None
    pressure_hpa: float | None = None
    humidity_percent: float | None = None
    wind_speed_kph: float | None = None
    wind_direction: str | None = None
    precipitation_mm: float | None = None
    cloud_cover: str | None = None
    source_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DailyRainfallGrid:
    """31-day × 12-month grid of rainfall values (inches).

    ``days`` maps "Day 1" … "Day 31" to a 12-element list (Jan–Dec).
    ``totals`` holds the 12 monthly totals row.
    Missing values are represented as ``None``.
    """

    days: dict[str, list[float | None]]
    totals: list[float | None]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DailyRainfallRecord:
    """One paired document image and its rainfall transcription."""

    stem: str
    county: str
    station_id: str
    decade: str
    image_path: Path
    transcription_path: Path | None = None
    grid: DailyRainfallGrid | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
