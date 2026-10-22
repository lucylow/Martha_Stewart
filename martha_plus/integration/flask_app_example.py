"""Example showing how to mount the operations blueprint in an existing Flask app.

Copy the blueprint registration into the original app factory only after reviewing
its authentication, proxy, deployment, and observability conventions.
"""
from flask import Flask
from martha_plus.api import create_blueprint
from martha_plus.config import Settings
from martha_plus.marketdata import CSVBarProvider
from martha_plus.portfolio import PortfolioLedger
from martha_plus.services import TradingResearchEngine
from martha_plus.strategies import MomentumStrategy


def create_app() -> Flask:
    settings = Settings.from_env()
    if not settings.api_token:
        raise RuntimeError("Set MARTHA_API_TOKEN to a random secret of at least 24 characters")
    data = CSVBarProvider("examples/sample_bars.csv")
    strategy = MomentumStrategy()
    engine = TradingResearchEngine(settings, data, strategy,
                                   ledger=PortfolioLedger(settings.starting_cash))
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(engine, settings.api_token))
    return app
