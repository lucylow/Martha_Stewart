import unittest
from martha_plus.config import Settings
from martha_plus.errors import ConfigurationError

class SettingsTests(unittest.TestCase):
    def test_safe_defaults_are_non_live(self):
        config = Settings.from_env({})
        self.assertEqual(config.mode, "research")
        self.assertEqual(config.max_gross_exposure, 0.75)
        self.assertEqual(config.api_token, "")

    def test_env_parsing(self):
        config = Settings.from_env({"MARTHA_MODE": "paper", "MARTHA_ALLOW_SHORT": "yes",
                                    "MARTHA_STARTING_CASH": "25000", "MARTHA_MAX_OPEN_ORDERS": "7"})
        self.assertTrue(config.allow_short)
        self.assertEqual(config.starting_cash, 25000)
        self.assertEqual(config.max_open_orders, 7)

    def test_live_mode_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            Settings.from_env({"MARTHA_MODE": "live"})

    def test_invalid_exposure_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            Settings.from_mapping({"max_gross_exposure": 1.1})

    def test_bad_boolean_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            Settings.from_env({"MARTHA_ALLOW_SHORT": "sometimes"})

    def test_nonpositive_cash_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            Settings.from_mapping({"starting_cash": 0})

    def test_unknown_mapping_key_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            Settings.from_mapping({"unexpected": 2})

    def test_secrets_are_masked(self):
        config = Settings.from_mapping({"api_token": "secret-token-value-1234567890"})
        self.assertNotEqual(config.as_safe_dict()["api_token"], config.api_token)
        self.assertIn("configured", config.as_safe_dict()["api_token"])

if __name__ == "__main__":
    unittest.main()
