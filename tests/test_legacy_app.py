"""Optional smoke coverage for the legacy Flask application alongside martha_plus."""
import importlib.util
import unittest
from unittest.mock import patch

_REQUIRED = (
    "flask",
    "flask_login",
    "flask_migrate",
    "flask_wtf",
    "flask_sqlalchemy",
    "sqlalchemy",
)
_HAS_LEGACY_STACK = all(importlib.util.find_spec(name) for name in _REQUIRED)


@unittest.skipUnless(_HAS_LEGACY_STACK, "install the legacy Flask requirements to run app smoke tests")
class LegacyAppSmokeTests(unittest.TestCase):
    def test_factory_routes_and_one_time_database_initialization(self):
        from app import create_app, db

        class TestConfig:
            SECRET_KEY = "test-secret-not-for-production"
            SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
            SQLALCHEMY_TRACK_MODIFICATIONS = False
            DEFAULT_THEME = None
            TESTING = True
            WTF_CSRF_ENABLED = False

        app = create_app(TestConfig, selenium=True)
        with patch("app.db.create_all", wraps=db.create_all) as create_all:
            with app.test_client() as client:
                expected = {
                    "/login": 200,
                    "/create_user": 200,
                    "/index": 200,
                    "/error-403": 200,
                    "/error-404": 200,
                    "/error-500": 200,
                    "/this/path/does/not/exist": 404,
                }
                for path, status in expected.items():
                    with self.subTest(path=path):
                        self.assertEqual(client.get(path).status_code, status)
                created = client.post("/create_user", data={
                    "username": "smoke-user",
                    "email": "smoke@example.test",
                    "password": "test-password",
                    "register": "1",
                })
                self.assertEqual(created.status_code, 200)
                logged_in = client.post("/login", data={
                    "username": "smoke-user",
                    "password": "test-password",
                    "login": "1",
                }, follow_redirects=True)
                self.assertEqual(logged_in.status_code, 200)
            self.assertEqual(create_all.call_count, 1)
        self.assertTrue(app.extensions["martha_database_initialized"])


if __name__ == "__main__":
    unittest.main()
