"""Command-line entry points for local research and paper-only demonstrations."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .backtest.engine import BacktestEngine
from .config import Settings
from .marketdata import CSVBarProvider, SyntheticMarketData
from .observability.logging import configure_logging
from .strategies import MomentumStrategy, MeanReversionStrategy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="martha-plus", description="Research and paper trading tools")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run a synthetic backtest")
    demo.add_argument("--symbol", default="DEMO")
    demo.add_argument("--bars", type=int, default=500)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--strategy", choices=["momentum", "mean-reversion"], default="momentum")
    backtest = sub.add_parser("backtest", help="backtest one symbol from a CSV file")
    backtest.add_argument("--csv", required=True)
    backtest.add_argument("--symbol", required=True)
    backtest.add_argument("--strategy", choices=["momentum", "mean-reversion"], default="momentum")
    backtest.add_argument("--starting-cash", type=float, default=100_000.0)
    backtest.add_argument("--output", default="")
    check = sub.add_parser("config-check", help="validate MARTHA_* environment variables")
    check.add_argument("--json", action="store_true")
    return parser


def _strategy(name: str):
    return MomentumStrategy() if name == "momentum" else MeanReversionStrategy()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    if args.command == "config-check":
        settings = Settings.from_env()
        payload = settings.as_safe_dict()
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for key, value in sorted(payload.items()):
                print(f"{key}={value}")
        return 0
    if args.command == "demo":
        if args.bars < 80:
            print("--bars must be at least 80 for the default strategy", file=sys.stderr)
            return 2
        source = SyntheticMarketData(seed=args.seed)
        bars = source.generate(args.symbol, count=args.bars)
        settings = Settings(starting_cash=100_000, mode="research")
        result = BacktestEngine(settings).run(bars, _strategy(args.strategy), minimum_history=35)
    else:
        provider = CSVBarProvider(args.csv)
        symbol = args.symbol.strip().upper()
        bars = provider.bars(symbol)
        settings = Settings(starting_cash=args.starting_cash, mode="research")
        result = BacktestEngine(settings).run(bars, _strategy(args.strategy), minimum_history=35)
    output = json.dumps(result.to_dict(), indent=2, default=str)
    if getattr(args, "output", ""):
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
