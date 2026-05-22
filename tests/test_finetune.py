"""Unit tests for finetune.py — no model or GPU required."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from weather_doc_extractor.finetune import (
    _ground_truth_json,
    build_training_example,
    build_training_examples,
)
from weather_doc_extractor.schemas import DailyRainfallGrid, DailyRainfallRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid() -> DailyRainfallGrid:
    days = {
        f"Day {i}": [float(i) if j < 6 else None for j in range(12)]
        for i in range(1, 32)
    }
    totals = [1.0, 2.0, None, 0.5, 1.2, None, 0.3, None, 0.9, 1.1, None, 2.3]
    return DailyRainfallGrid(days=days, totals=totals)


def _make_record(tmp_path: Path, paired: bool = True) -> DailyRainfallRecord:
    img = tmp_path / "DRain_1950-1959_Kent-1.jpg"
    # Write a minimal valid JPEG-ish file so PIL can open it
    from PIL import Image as PILImage

    PILImage.new("RGB", (4, 4), color=(128, 128, 128)).save(str(img))
    return DailyRainfallRecord(
        stem="DRain_1950-1959_Kent-1",
        county="Kent",
        station_id="1",
        decade="1950-1959",
        image_path=img,
        transcription_path=tmp_path / "DRain_1950-1959_Kent-1.json" if paired else None,
        grid=_make_grid() if paired else None,
    )


# ---------------------------------------------------------------------------
# _ground_truth_json
# ---------------------------------------------------------------------------


class TestGroundTruthJson:
    def test_valid_json(self, tmp_path):
        record = _make_record(tmp_path)
        text = _ground_truth_json(record)
        parsed = json.loads(text)
        assert isinstance(parsed, dict)

    def test_has_all_day_keys(self, tmp_path):
        record = _make_record(tmp_path)
        parsed = json.loads(_ground_truth_json(record))
        for i in range(1, 32):
            assert f"Day {i}" in parsed

    def test_has_totals_key(self, tmp_path):
        record = _make_record(tmp_path)
        parsed = json.loads(_ground_truth_json(record))
        assert "Totals" in parsed

    def test_day_arrays_have_12_elements(self, tmp_path):
        record = _make_record(tmp_path)
        parsed = json.loads(_ground_truth_json(record))
        for i in range(1, 32):
            assert len(parsed[f"Day {i}"]) == 12

    def test_totals_has_12_elements(self, tmp_path):
        record = _make_record(tmp_path)
        parsed = json.loads(_ground_truth_json(record))
        assert len(parsed["Totals"]) == 12

    def test_raises_for_unpaired_record(self, tmp_path):
        record = _make_record(tmp_path, paired=False)
        with pytest.raises(AssertionError):
            _ground_truth_json(record)


# ---------------------------------------------------------------------------
# build_training_example — SmolVLM family
# ---------------------------------------------------------------------------


class TestBuildTrainingExampleSmolvlm:
    def test_returns_dict_with_image_and_messages(self, tmp_path):
        record = _make_record(tmp_path)
        ex = build_training_example(record, "smolvlm")
        assert "image" in ex
        assert "messages" in ex

    def test_image_is_pil(self, tmp_path):
        from PIL import Image as PILImage

        record = _make_record(tmp_path)
        ex = build_training_example(record, "smolvlm")
        assert isinstance(ex["image"], PILImage.Image)

    def test_messages_has_user_and_assistant(self, tmp_path):
        record = _make_record(tmp_path)
        ex = build_training_example(record, "smolvlm")
        roles = [m["role"] for m in ex["messages"]]
        assert roles == ["user", "assistant"]

    def test_user_message_has_image_placeholder(self, tmp_path):
        record = _make_record(tmp_path)
        ex = build_training_example(record, "smolvlm")
        user_content = ex["messages"][0]["content"]
        image_items = [c for c in user_content if c.get("type") == "image"]
        assert len(image_items) == 1
        # SmolVLM placeholder: no "image" or "url" key inside the item
        assert "image" not in image_items[0]
        assert "url" not in image_items[0]

    def test_assistant_message_contains_valid_json(self, tmp_path):
        record = _make_record(tmp_path)
        ex = build_training_example(record, "smolvlm")
        assistant_text = ex["messages"][1]["content"][0]["text"]
        parsed = json.loads(assistant_text)
        assert "Day 1" in parsed
        assert "Totals" in parsed

    def test_generic_family_same_as_smolvlm(self, tmp_path):
        record = _make_record(tmp_path)
        smol = build_training_example(record, "smolvlm")
        generic = build_training_example(record, "generic")
        smol_img_item = [
            c for c in smol["messages"][0]["content"] if c.get("type") == "image"
        ][0]
        gen_img_item = [
            c for c in generic["messages"][0]["content"] if c.get("type") == "image"
        ][0]
        assert smol_img_item == gen_img_item

    def test_smolvlm2_uses_placeholder_like_smolvlm(self, tmp_path):
        record = _make_record(tmp_path)
        ex = build_training_example(record, "smolvlm2")
        user_content = ex["messages"][0]["content"]
        image_items = [c for c in user_content if c.get("type") == "image"]
        assert len(image_items) == 1
        # SmolVLM2 should use placeholder (no url, no image field)
        assert "url" not in image_items[0]
        assert "image" not in image_items[0]


# ---------------------------------------------------------------------------
# build_training_example — Granite family
# ---------------------------------------------------------------------------


class TestBuildTrainingExampleGranite:
    def test_image_item_contains_pil(self, tmp_path):
        from PIL import Image as PILImage

        record = _make_record(tmp_path)
        ex = build_training_example(record, "granite")
        user_content = ex["messages"][0]["content"]
        image_items = [c for c in user_content if c.get("type") == "image"]
        assert len(image_items) == 1
        # Granite embeds the PIL image directly
        assert isinstance(image_items[0].get("image"), PILImage.Image)

    def test_no_url_key_in_image_item(self, tmp_path):
        """Granite training uses PIL images, not file URLs."""
        record = _make_record(tmp_path)
        ex = build_training_example(record, "granite")
        user_content = ex["messages"][0]["content"]
        image_item = next(c for c in user_content if c.get("type") == "image")
        assert "url" not in image_item

    def test_assistant_json_matches_ground_truth(self, tmp_path):
        record = _make_record(tmp_path)
        ex = build_training_example(record, "granite")
        assistant_text = ex["messages"][1]["content"][0]["text"]
        expected = json.loads(_ground_truth_json(record))
        assert json.loads(assistant_text) == expected

    def test_granite4_uses_same_embedded_image_pattern(self, tmp_path):
        from PIL import Image as PILImage

        record = _make_record(tmp_path)
        ex = build_training_example(record, "granite4")
        user_content = ex["messages"][0]["content"]
        image_item = next(c for c in user_content if c.get("type") == "image")
        assert isinstance(image_item.get("image"), PILImage.Image)
        assert "url" not in image_item


# ---------------------------------------------------------------------------
# build_training_examples
# ---------------------------------------------------------------------------


class TestBuildTrainingExamples:
    def test_filters_unpaired_records(self, tmp_path):
        paired = _make_record(tmp_path)
        unpaired = _make_record(tmp_path, paired=False)
        examples = build_training_examples([paired, unpaired], "smolvlm")
        assert len(examples) == 1

    def test_empty_list_when_no_paired(self, tmp_path):
        unpaired = _make_record(tmp_path, paired=False)
        examples = build_training_examples([unpaired], "smolvlm")
        assert examples == []

    def test_count_matches_paired_records(self, tmp_path):
        records = [_make_record(tmp_path) for _ in range(3)]
        records.append(_make_record(tmp_path, paired=False))
        examples = build_training_examples(records, "smolvlm")
        assert len(examples) == 3

    def test_granite_family_produces_embedded_pil(self, tmp_path):
        from PIL import Image as PILImage

        record = _make_record(tmp_path)
        examples = build_training_examples([record], "granite")
        assert len(examples) == 1
        user_content = examples[0]["messages"][0]["content"]
        img_item = next(c for c in user_content if c.get("type") == "image")
        assert isinstance(img_item.get("image"), PILImage.Image)

    def test_granite4_family_produces_embedded_pil(self, tmp_path):
        from PIL import Image as PILImage

        record = _make_record(tmp_path)
        examples = build_training_examples([record], "granite4")
        assert len(examples) == 1
        user_content = examples[0]["messages"][0]["content"]
        img_item = next(c for c in user_content if c.get("type") == "image")
        assert isinstance(img_item.get("image"), PILImage.Image)


# ---------------------------------------------------------------------------
# CLI integration — finetune command argument parsing
# ---------------------------------------------------------------------------


class TestFinetuneCliArgParsing:
    """Verify that the finetune command parses flags without loading a model."""

    def test_finetune_command_unknown_without_finetune_available(self):
        """Smoke test: CLI does not crash on import for finetune command."""
        from weather_doc_extractor.cli import run

        # We mock run_finetune so no model is loaded
        with patch("weather_doc_extractor.cli.run_finetune") as mock_ft:
            mock_ft.return_value = Path("/tmp/adapter")
            result = run(
                [
                    "finetune",
                    "--model",
                    "smolvlm",
                    "--epochs",
                    "1",
                    "--eval-split",
                    "0.2",
                ]
            )
        assert result == 0
        mock_ft.assert_called_once()

    def test_finetune_command_updates_config(self):
        """--epochs and --eval-split are applied to config before run_finetune is called."""
        from weather_doc_extractor.cli import run
        from weather_doc_extractor.config import AppConfig

        captured_configs: list[AppConfig] = []

        def _capture(config):
            captured_configs.append(config)
            return Path("/tmp/adapter")

        with patch("weather_doc_extractor.cli.run_finetune", side_effect=_capture):
            run(["finetune", "--epochs", "5", "--eval-split", "0.15"])

        cfg = captured_configs[0]
        assert cfg.training.epochs == 5
        assert abs(cfg.training.eval_split - 0.15) < 1e-9
