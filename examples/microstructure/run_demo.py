#!/usr/bin/env python3
"""Run a seeded, broker-free market-microstructure replay demonstration."""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

# Allow execution directly from a source checkout without an editable install.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from martha_plus.microstructure.matching.engine import MatchingEngine
from martha_plus.microstructure.reporting import ReportExporter, RunSummaryBuilder
from martha_plus.microstructure.replay.runner import MarketReplay
from martha_plus.microstructure.simulation.synthetic import SyntheticFeedConfig, SyntheticMarketFeed
from martha_plus.microstructure.strategies.imbalance import ImbalanceConfig, ImbalanceTaker
from martha_plus.microstructure.types import SymbolSpec
from martha_plus.microstructure.data.archive import archive_events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=3000,
                        help="number of synthetic updates after the initial book (default: 3000)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--symbol", default="SIM")
    parser.add_argument("--archive", type=Path,
                        help="optional path for a checksummed JSONL market-event archive")
    parser.add_argument("--report", type=Path,
                        help="optional JSON summary output path")
    args = parser.parse_args()
    if args.events < 1:
        parser.error("--events must be positive")

    config = SyntheticFeedConfig(symbol=args.symbol, events=args.events)
    events = SyntheticMarketFeed(config, seed=args.seed).generate()
    if args.archive:
        stats = archive_events(args.archive, events, metadata={
            "symbol": args.symbol.upper(), "seed": str(args.seed),
            "feed": "synthetic", "mode": "paper-only",
        })
        print(f"Archived {stats.records} events to {stats.path} (sha256={stats.sha256})")

    engine = MatchingEngine(SymbolSpec(args.symbol))
    strategy = ImbalanceTaker(
        strategy_id="demo-imbalance",
        config=ImbalanceConfig(threshold=Decimal("0.55"),
                               quantity=1, cooldown_events=1),
    )
    result = MarketReplay(engine, strategy).run(events)
    summary = RunSummaryBuilder().from_run_result(result)
    payload = ReportExporter.to_json(summary)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote summary to {args.report}")
    print(payload)
    print("\nDetailed replay diagnostics:")
    print(json.dumps({
        "accepted_orders": result.accepted_orders,
        "rejected_orders": result.rejected_orders,
        "risk_rejections": len(result.risk_rejections),
        "executions": len(result.executions),
        "filled_quantity": result.filled_quantity,
        "anomalies": list(result.anomalies[:10]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
