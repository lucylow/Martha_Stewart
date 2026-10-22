import unittest
from datetime import datetime, timezone
from martha_plus.clock import FrozenClock
from martha_plus.marketdata.synthetic import SyntheticMarketData
from martha_plus.risk import KillSwitch
from martha_plus.services import TradingResearchEngine
from martha_plus.strategies import MomentumStrategy
from tests.helpers import settings

try:
    from flask import Flask
    from martha_plus.api import create_blueprint
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

@unittest.skipUnless(FLASK_AVAILABLE, "Flask is optional")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.source = SyntheticMarketData(seed=90)
        self.source.generate("TEST", 100)
        self.clock = FrozenClock(self.source.latest_quote("TEST").timestamp)
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.settings = settings(kill_switch_file=self.temp.name + "/HALT")
        self.engine = TradingResearchEngine(self.settings, self.source, MomentumStrategy(5, 20),
                                            clock=self.clock, kill_switch=KillSwitch(self.settings.kill_switch_file))
        self.app = Flask(__name__)
        self.app.register_blueprint(create_blueprint(self.engine, "a-very-long-test-token-1234567890"))
        self.client = self.app.test_client()
        self.auth = {"Authorization": "Bearer a-very-long-test-token-1234567890"}

    def tearDown(self):
        self.temp.cleanup()

    def test_health_is_public_and_minimal(self):
        response = self.client.get("/martha-plus/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["paper_only"])
        self.assertNotIn("positions", response.get_json())

    def test_status_requires_auth(self):
        self.assertEqual(self.client.get("/martha-plus/status").status_code, 401)
        response = self.client.get("/martha-plus/status", headers=self.auth)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["paper_only"])

    def test_kill_switch_api(self):
        response = self.client.post("/martha-plus/kill-switch", headers=self.auth,
                                    json={"reason": "operator request"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.engine.kill_switch.active)
        response = self.client.delete("/martha-plus/kill-switch", headers=self.auth)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(self.engine.kill_switch.active)

    def test_invalid_short_token_rejected(self):
        with self.assertRaises(ValueError):
            create_blueprint(self.engine, "short")

if __name__ == "__main__":
    unittest.main()
