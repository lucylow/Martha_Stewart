import json
import logging
import unittest
from martha_plus.observability import MetricsRegistry, configure_logging, redact_secrets
from martha_plus.observability.logging import JsonFormatter

class ObservabilityTests(unittest.TestCase):
    def test_redacts_api_token(self):
        value = redact_secrets("api_token=supersecret123 Authorization: Bearer abcdef123")
        self.assertNotIn("supersecret123", value)
        self.assertNotIn("abcdef123", value)
        self.assertIn("[REDACTED]", value)

    def test_json_formatter_has_timestamp_and_level(self):
        record = logging.LogRecord("test", logging.INFO, "a.py", 10, "hello token=abc", (), None)
        output = JsonFormatter().format(record)
        payload = json.loads(output)
        self.assertEqual(payload["level"], "INFO")
        self.assertIn("timestamp", payload)
        self.assertNotIn("abc", payload["message"])

    def test_metrics_registry_counters_gauges_observations(self):
        metrics = MetricsRegistry()
        metrics.inc("orders")
        metrics.inc("orders", 2)
        metrics.set_gauge("open_orders", 4)
        for value in (1, 2, 3, 4, 5):
            metrics.observe("latency", value)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["counters"]["orders"], 3)
        self.assertEqual(snapshot["gauges"]["open_orders"], 4)
        self.assertEqual(snapshot["observations"]["latency"]["count"], 5)

    def test_metrics_reject_negative_counter(self):
        with self.assertRaises(ValueError):
            MetricsRegistry().inc("count", -1)

    def test_configure_logging_is_idempotent(self):
        logger = configure_logging("DEBUG", "martha_plus_test_logger")
        count = len(logger.handlers)
        configure_logging("DEBUG", "martha_plus_test_logger")
        self.assertEqual(len(logger.handlers), count)

if __name__ == "__main__":
    unittest.main()
