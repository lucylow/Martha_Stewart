"""Deterministic run summaries and safe report exports."""
from __future__ import annotations

import csv
import html
import io
import json
from dataclasses import dataclass
from decimal import Decimal
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from ..types import Execution


@dataclass(frozen=True, slots=True)
class RunSummary:
    symbol: str
    duration_ns: int
    market_events: int
    scheduled_tasks: int
    decisions: int
    accepted_orders: int
    rejected_orders: int
    risk_rejections: int
    executions: int
    traded_quantity: int
    traded_notional_ticks: int
    maker_quantity: int
    taker_quantity: int
    maker_fraction: Decimal
    mean_trade_size: Decimal | None
    mean_inter_trade_ns: Decimal | None
    anomaly_count: int
    final_cash: Decimal | None
    final_position: int | None
    final_equity: Decimal | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "duration_ns": self.duration_ns,
            "market_events": self.market_events,
            "scheduled_tasks": self.scheduled_tasks,
            "decisions": self.decisions,
            "accepted_orders": self.accepted_orders,
            "rejected_orders": self.rejected_orders,
            "risk_rejections": self.risk_rejections,
            "executions": self.executions,
            "traded_quantity": self.traded_quantity,
            "traded_notional_ticks": self.traded_notional_ticks,
            "maker_quantity": self.maker_quantity,
            "taker_quantity": self.taker_quantity,
            "maker_fraction": str(self.maker_fraction),
            "mean_trade_size": str(self.mean_trade_size) if self.mean_trade_size is not None else None,
            "mean_inter_trade_ns": str(self.mean_inter_trade_ns) if self.mean_inter_trade_ns is not None else None,
            "anomaly_count": self.anomaly_count,
            "final_cash": str(self.final_cash) if self.final_cash is not None else None,
            "final_position": self.final_position,
            "final_equity": str(self.final_equity) if self.final_equity is not None else None,
        }


class RunSummaryBuilder:
    """Build summary metrics from a replay result without mutating it."""

    def from_run_result(self, result: Any) -> RunSummary:
        executions: Sequence[Execution] = tuple(result.executions)
        quantity = sum(x.quantity for x in executions)
        notional_ticks = sum(x.price_ticks * x.quantity for x in executions)
        strategy_order_ids = {ack.client_order_id for ack in result.venue_acks if ack.accepted}
        maker_quantity = sum(x.quantity for x in executions if x.maker_order_id in strategy_order_ids)
        taker_quantity = sum(x.quantity for x in executions if x.taker_order_id in strategy_order_ids)
        total_strategy_qty = maker_quantity + taker_quantity
        maker_fraction = Decimal(maker_quantity) / Decimal(total_strategy_qty) if total_strategy_qty else Decimal(0)
        trade_sizes = [x.quantity for x in executions]
        trade_times = sorted(x.ts_ns for x in executions)
        intervals = [b - a for a, b in zip(trade_times, trade_times[1:])]
        ledger_rows = tuple((result.ledger or {}).values())
        final_cash = (sum((Decimal(str(row["cash"])) for row in ledger_rows), Decimal(0))
                      if ledger_rows else None)
        final_position = (sum(int(row["quantity"]) for row in ledger_rows)
                          if ledger_rows else None)
        final_equity = (sum((Decimal(str(row["equity"])) for row in ledger_rows), Decimal(0))
                        if ledger_rows else None)
        return RunSummary(
            result.symbol, max(0, result.end_ns - result.start_ns),
            result.market_events, result.scheduled_tasks,
            len(result.strategy_decisions), result.accepted_orders,
            result.rejected_orders, len(result.risk_rejections), len(executions),
            quantity, notional_ticks, maker_quantity, taker_quantity,
            maker_fraction,
            Decimal(str(fmean(trade_sizes))) if trade_sizes else None,
            Decimal(str(fmean(intervals))) if intervals else None,
            len(result.anomalies),
            final_cash, final_position, final_equity,
        )


class ReportExporter:
    """Export authored metrics as JSON, CSV, or escaped standalone HTML."""

    @staticmethod
    def to_json(summary: RunSummary, *, indent: int = 2) -> str:
        if indent < 0 or indent > 8:
            raise ValueError("indent must be between zero and eight")
        return json.dumps(summary.to_dict(), sort_keys=True, indent=indent, allow_nan=False)

    @staticmethod
    def to_csv(summary: RunSummary) -> str:
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(("metric", "value"))
        for key, value in summary.to_dict().items():
            writer.writerow((key, "" if value is None else value))
        return buffer.getvalue()

    @staticmethod
    def to_html(summary: RunSummary, *, title: str = "Martha Plus Simulation Report") -> str:
        title_safe = html.escape(title)
        rows = []
        for key, value in summary.to_dict().items():
            value_text = "—" if value is None else str(value)
            rows.append(f"<tr><th scope=\"row\">{html.escape(key)}</th><td>{html.escape(value_text)}</td></tr>")
        return (
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{title_safe}</title><style>body{{font:16px system-ui;max-width:960px;margin:2rem auto;padding:0 1rem}}"
            "table{border-collapse:collapse;width:100%}th,td{padding:.55rem;border-bottom:1px solid #ddd;text-align:left}"
            "th{width:48%}</style></head><body>"
            f"<h1>{title_safe}</h1><p>Offline simulation report. Not a live trading record.</p>"
            f"<table><tbody>{''.join(rows)}</tbody></table></body></html>"
        )


class ExecutionTableExporter:
    """Export raw executions to CSV without losing integer tick units."""

    FIELDS = ("execution_id", "ts_ns", "symbol", "price_ticks", "quantity",
              "maker_order_id", "taker_order_id", "maker_owner", "taker_owner",
              "maker_side", "taker_side", "maker_fee", "taker_fee")

    def to_csv(self, executions: Iterable[Execution]) -> str:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=self.FIELDS)
        writer.writeheader()
        for row in executions:
            writer.writerow({
                "execution_id": row.execution_id, "ts_ns": row.ts_ns,
                "symbol": row.symbol, "price_ticks": row.price_ticks,
                "quantity": row.quantity, "maker_order_id": row.maker_order_id,
                "taker_order_id": row.taker_order_id, "maker_owner": row.maker_owner,
                "taker_owner": row.taker_owner, "maker_side": row.maker_side.value,
                "taker_side": row.taker_side.value, "maker_fee": str(row.maker_fee),
                "taker_fee": str(row.taker_fee),
            })
        return buffer.getvalue()
