import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from martha_plus.cli import main

class CliTests(unittest.TestCase):
    def test_demo_command_outputs_json(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            status = main(["demo", "--bars", "100", "--symbol", "TEST"])
        self.assertEqual(status, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["symbol"], "TEST")
        self.assertIn("metrics", payload)

    def test_demo_rejects_too_few_bars(self):
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            status = main(["demo", "--bars", "30"])
        self.assertEqual(status, 2)

    def test_config_check_redacts_api_token(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"MARTHA_API_TOKEN": "very-secret-token-value-123456789"}):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                status = main(["config-check", "--json"])
        self.assertEqual(status, 0)
        payload = json.loads(buffer.getvalue())
        self.assertNotEqual(payload["api_token"], "very-secret-token-value-123456789")

    def test_backtest_writes_output_file(self):
        sample = Path(__file__).parent.parent / "examples" / "sample_bars.csv"
        # The sample is deliberately short; extend the fixture with reproducible bars.
        import csv
        from martha_plus.marketdata.synthetic import SyntheticMarketData
        data = SyntheticMarketData(seed=123).generate("TEST", 100)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "many.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "symbol", "open", "high", "low", "close", "volume"])
                for bar in data:
                    writer.writerow([bar.timestamp.isoformat(), bar.symbol, bar.open, bar.high, bar.low, bar.close, bar.volume])
            output = Path(temp) / "results" / "test.json"
            status = main(["backtest", "--csv", str(path), "--symbol", "TEST", "--output", str(output)])
            self.assertEqual(status, 0)
            self.assertTrue(output.is_file())
            self.assertIn("metrics", json.loads(output.read_text(encoding="utf-8")))

if __name__ == "__main__":
    unittest.main()
