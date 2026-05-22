import io
import json
import unittest
from contextlib import redirect_stdout

from weather_doc_extractor.cli import run
from weather_doc_extractor.config import MODEL_PRESETS


class CliTests(unittest.TestCase):
    def test_info_command_prints_project_summary(self) -> None:
        stream = io.StringIO()

        with redirect_stdout(stream):
            exit_code = run(["info"])

        lines = stream.getvalue().strip().splitlines()
        self.assertEqual(exit_code, 0)
        self.assertEqual(lines[0], "Weather document extraction project")
        payload = json.loads("\n".join(lines[1:]))
        self.assertEqual(
            payload["model"]["model_name"], "HuggingFaceTB/SmolVLM-500M-Instruct"
        )

    def test_unknown_command_returns_error(self) -> None:
        stream = io.StringIO()

        with redirect_stdout(stream):
            exit_code = run(["bad-command"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Unknown command: bad-command", stream.getvalue())

    def test_model_presets_contains_smolvlm_and_granite(self) -> None:
        self.assertIn("smolvlm", MODEL_PRESETS)
        self.assertIn("smolvlm2", MODEL_PRESETS)
        self.assertIn("granite", MODEL_PRESETS)
        self.assertIn("granite4", MODEL_PRESETS)
        self.assertIn("SmolVLM", MODEL_PRESETS["smolvlm"])
        self.assertEqual(
            MODEL_PRESETS["smolvlm2"], "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        )
        self.assertEqual(MODEL_PRESETS["granite"], "ibm-granite/granite-vision-3.2-2b")
        self.assertEqual(MODEL_PRESETS["granite4"], "ibm-granite/granite-vision-4.1-4b")

    def test_model_presets_contains_gemma(self) -> None:
        self.assertIn("gemma3", MODEL_PRESETS)
        self.assertIn("gemma4", MODEL_PRESETS)
        self.assertIn("gemma-3", MODEL_PRESETS["gemma3"])
        self.assertIn("gemma-4", MODEL_PRESETS["gemma4"])

    def test_model_presets_contains_ministral(self) -> None:
        self.assertIn("ministral", MODEL_PRESETS)
        self.assertIn("Mistral-Small", MODEL_PRESETS["ministral"])

    def test_extract_missing_path_returns_error(self) -> None:
        import sys
        from io import StringIO
        from contextlib import redirect_stderr

        err = StringIO()
        with redirect_stderr(err):
            code = run(["extract"])
        self.assertEqual(code, 1)

    def test_extract_model_flag_nonexistent_image(self) -> None:
        """--model flag is parsed; nonexistent image still returns error code 1."""
        import sys
        from io import StringIO
        from contextlib import redirect_stderr

        err = StringIO()
        with redirect_stderr(err):
            code = run(["extract", "--model", "granite", "/nonexistent/image.jpg"])
        self.assertEqual(code, 1)
        self.assertIn("not found", err.getvalue())


if __name__ == "__main__":
    unittest.main()
