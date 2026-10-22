"""Regression checks for the legacy test package beside the research test suite."""
import importlib
import unittest


class LegacyTestImportTests(unittest.TestCase):
    def test_legacy_test_modules_import_from_their_own_package(self):
        example = importlib.import_module("testing.test_example")
        selenium = importlib.import_module("testing.test_selenium")
        base = importlib.import_module("testing.test_base")
        self.assertTrue(callable(example.test_pages))
        self.assertTrue(callable(selenium.test_selenium))
        self.assertIn("/home", base.urls)


if __name__ == "__main__":
    unittest.main()
