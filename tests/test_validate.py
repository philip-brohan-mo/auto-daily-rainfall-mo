"""Unit tests for weather_doc_extractor.validate.

These tests exercise the local validation logic only — no model inference,
no GPU, and no Azure connectivity required.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from weather_doc_extractor.evaluate import EvaluationReport
from weather_doc_extractor.schemas import DailyRainfallGrid
from weather_doc_extractor.validate import (
    ValidationRecord,
    load_extraction_results,
    print_summary,
    run_validation,
    save_report,
)

MONTHS = 12
_ALL_DAYS = [f"Day {i}" for i in range(1, 32)]


def _make_grid(value: float | None = 0.10) -> DailyRainfallGrid:
    days = {day: [value] * MONTHS for day in _ALL_DAYS}
    return DailyRainfallGrid(days=days, totals=[value] * MONTHS)


def _write_extraction(
    tmp_path: Path, stem: str, grid: DailyRainfallGrid | None
) -> Path:
    """Write a mock extraction JSON file as batch-extract would produce."""
    out = tmp_path / f"{stem}.json"
    if grid is None:
        out.write_text(
            json.dumps({"stem": stem, "parse_failed": True, "raw_text": "oops"})
        )
    else:
        out.write_text(
            json.dumps({"stem": stem, "parse_failed": False, "grid": grid.to_dict()})
        )
    return out


def _write_transcription(tmp_path: Path, stem: str, grid: DailyRainfallGrid) -> Path:
    """Write a minimal transcription JSON as ingest.load_grid expects."""
    payload: dict[str, object] = {}
    for day_key, vals in grid.days.items():
        payload[day_key] = [str(v) if v is not None else "null" for v in vals]
    payload["Totals"] = [str(v) if v is not None else "null" for v in grid.totals]
    out = tmp_path / f"{stem}.json"
    out.write_text(json.dumps(payload))
    return out


def _write_dummy_image(tmp_path: Path, stem: str) -> Path:
    """Write a tiny 1×1 JPEG-like file so the path exists (not actually read in unit tests)."""
    img_path = tmp_path / f"{stem}.jpg"
    # Minimal JPEG magic bytes — enough for pathlib.exists(), but we won't actually open it.
    img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    return img_path


class LoadExtractionResultsTest(unittest.TestCase):
    def _setup_dirs(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        """Create standard directory layout under tmp_path and return paths."""
        ext_dir = tmp_path / "extractions"
        img_dir = tmp_path / "test_data" / "images"
        tr_dir = tmp_path / "test_data" / "transcriptions"
        ext_dir.mkdir(parents=True)
        img_dir.mkdir(parents=True)
        tr_dir.mkdir(parents=True)
        return ext_dir, img_dir, tr_dir

    def test_single_matched_pair(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ext_dir, img_dir, tr_dir = self._setup_dirs(tmp)
            stem = "DRain_1881-1890_Cornwall-1"
            grid = _make_grid(0.20)
            _write_extraction(ext_dir, stem, grid)
            _write_transcription(tr_dir, stem, grid)
            _write_dummy_image(img_dir, stem)

            records = load_extraction_results(ext_dir, tmp / "test_data")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].stem, stem)
        self.assertIsNotNone(records[0].predicted)
        self.assertIsNotNone(records[0].ground_truth)

    def test_parse_failed_gives_none_predicted(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ext_dir, img_dir, tr_dir = self._setup_dirs(tmp)
            stem = "DRain_1881-1890_Cornwall-2"
            grid = _make_grid(0.15)
            _write_extraction(ext_dir, stem, None)  # parse_failed = True
            _write_transcription(tr_dir, stem, grid)
            _write_dummy_image(img_dir, stem)

            records = load_extraction_results(ext_dir, tmp / "test_data")

        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0].predicted)

    def test_missing_extraction_skips_stem(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ext_dir, img_dir, tr_dir = self._setup_dirs(tmp)
            stem = "DRain_1881-1890_Cornwall-3"
            grid = _make_grid(0.10)
            # Intentionally do NOT write an extraction file.
            _write_transcription(tr_dir, stem, grid)
            _write_dummy_image(img_dir, stem)

            import io
            from contextlib import redirect_stderr

            buf = io.StringIO()
            with redirect_stderr(buf):
                records = load_extraction_results(ext_dir, tmp / "test_data")

        self.assertEqual(len(records), 0)
        self.assertIn("no extraction result", buf.getvalue())

    def test_missing_image_skips_stem(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ext_dir, img_dir, tr_dir = self._setup_dirs(tmp)
            stem = "DRain_1881-1890_Cornwall-4"
            grid = _make_grid(0.10)
            _write_extraction(ext_dir, stem, grid)
            _write_transcription(tr_dir, stem, grid)
            # Intentionally do NOT write an image.

            import io
            from contextlib import redirect_stderr

            buf = io.StringIO()
            with redirect_stderr(buf):
                records = load_extraction_results(ext_dir, tmp / "test_data")

        self.assertEqual(len(records), 0)
        self.assertIn("image not found", buf.getvalue())

    def test_multiple_stems_sorted(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ext_dir, img_dir, tr_dir = self._setup_dirs(tmp)
            stems = ["DRain_1881-1890_Cornwall-10", "DRain_1871-1880_Cornwall-9"]
            grid = _make_grid(0.10)
            for stem in stems:
                _write_extraction(ext_dir, stem, grid)
                _write_transcription(tr_dir, stem, grid)
                _write_dummy_image(img_dir, stem)

            records = load_extraction_results(ext_dir, tmp / "test_data")

        self.assertEqual(len(records), 2)
        # Records come back in sorted stem order.
        self.assertEqual(records[0].stem, "DRain_1871-1880_Cornwall-9")
        self.assertEqual(records[1].stem, "DRain_1881-1890_Cornwall-10")


class RunValidationTest(unittest.TestCase):
    def _make_record(
        self,
        stem: str,
        pred_value: float | None = 0.10,
        gt_value: float | None = 0.10,
        parse_failed: bool = False,
    ) -> ValidationRecord:
        gt = _make_grid(gt_value)
        pred = None if parse_failed else _make_grid(pred_value)
        return ValidationRecord(
            stem=stem,
            predicted=pred,
            ground_truth=gt,
            image_path=Path(f"/fake/{stem}.jpg"),
        )

    def test_perfect_match(self) -> None:
        records = [self._make_record("stem1", 0.10, 0.10)]
        report = run_validation(records)
        self.assertEqual(report.n_images, 1)
        self.assertAlmostEqual(report.macro_accuracy, 1.0)

    def test_all_wrong(self) -> None:
        records = [self._make_record("stem1", 0.10, 0.99)]
        report = run_validation(records)
        self.assertAlmostEqual(report.macro_accuracy, 0.0)

    def test_parse_failed_counted(self) -> None:
        records = [
            self._make_record("stem1", parse_failed=True),
            self._make_record("stem2", 0.10, 0.10),
        ]
        report = run_validation(records)
        self.assertEqual(report.n_failed, 1)
        self.assertEqual(report.n_images, 2)

    def test_returns_evaluation_report(self) -> None:
        records = [self._make_record("stem1")]
        report = run_validation(records)
        self.assertIsInstance(report, EvaluationReport)

    def test_empty_records(self) -> None:
        report = run_validation([])
        self.assertEqual(report.n_images, 0)


class PrintSummaryTest(unittest.TestCase):
    def test_no_crash_on_valid_report(self) -> None:
        """print_summary should not raise for a normal report."""
        import io
        from contextlib import redirect_stdout

        records = [
            ValidationRecord(
                stem="DRain_1881-1890_Cornwall-1",
                predicted=_make_grid(0.10),
                ground_truth=_make_grid(0.10),
                image_path=Path("/fake/DRain_1881-1890_Cornwall-1.jpg"),
            )
        ]
        report = run_validation(records)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_summary(report)
        output = buf.getvalue()
        self.assertIn("AGGREGATE", output)
        self.assertIn("100.0%", output)

    def test_no_crash_with_parse_failure(self) -> None:
        import io
        from contextlib import redirect_stdout

        records = [
            ValidationRecord(
                stem="DRain_1881-1890_Cornwall-2",
                predicted=None,
                ground_truth=_make_grid(0.10),
                image_path=Path("/fake/DRain_1881-1890_Cornwall-2.jpg"),
            )
        ]
        report = run_validation(records)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_summary(report)
        output = buf.getvalue()
        self.assertIn("FAILED", output)


class SaveReportTest(unittest.TestCase):
    def test_json_structure(self) -> None:
        import tempfile

        records = [
            ValidationRecord(
                stem="DRain_1881-1890_Cornwall-1",
                predicted=_make_grid(0.10),
                ground_truth=_make_grid(0.10),
                image_path=Path("/fake/img.jpg"),
            )
        ]
        report = run_validation(records)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "report.json"
            returned = save_report(report, path)
            data = json.loads(path.read_text())

        self.assertEqual(returned, path)
        self.assertIn("summary", data)
        self.assertIn("comparisons", data)
        self.assertEqual(len(data["comparisons"]), 1)
        self.assertEqual(data["comparisons"][0]["stem"], "DRain_1881-1890_Cornwall-1")


if __name__ == "__main__":
    unittest.main()
