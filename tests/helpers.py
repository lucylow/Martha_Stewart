"""Shared deterministic fixtures for the test suite."""
from datetime import datetime, timezone
from martha_plus.config import Settings
from martha_plus.marketdata.synthetic import SyntheticMarketData
from martha_plus.portfolio.ledger import PortfolioLedger
from martha_plus.types import Quote


def bars(symbol="TEST", count=180, seed=5):
    source = SyntheticMarketData(seed=seed)
    return list(source.generate(symbol, count=count, start_price=100, drift=0.0004, volatility=0.008))


def quote(symbol="TEST", price=100.0, stamp=None):
    return Quote(symbol, price - 0.01, price + 0.01,
                 stamp or datetime.now(timezone.utc), bid_size=100, ask_size=100)


def settings(**overrides):
    values = {"mode": "paper", "starting_cash": 100_000.0,
              "max_order_notional": 5_000.0, "max_position_notional": 15_000.0,
              "max_gross_exposure": 0.75, "max_daily_loss": 2_000.0,
              "max_drawdown_fraction": 0.12, "max_open_orders": 20,
              "max_quote_age_seconds": 5.0, "commission_bps": 1.0,
              "slippage_bps": 2.0, "paper_fill_probability": 1.0,
              "allow_short": False, "allow_fractional_shares": False,
              "kill_switch_file": "/tmp/martha-plus-test-KILL_SWITCH"}
    values.update(overrides)
    return Settings.from_mapping(values)


def ledger(cash=100_000.0):
    return PortfolioLedger(cash)
