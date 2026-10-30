"""Multi-currency paper portfolio ledger with FIFO lots and corporate actions.

This module is an analytical book of record for simulations. It is not a tax
accounting package, broker statement reconciler, custody ledger or production
settlement system. Amounts are floating-point research values; use Decimal and
venue-specific settlement calendars at external accounting boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Mapping, Sequence
import math

from .core import FinanceInputError, Number, finite, positive


@dataclass
class Lot:
    symbol: str
    quantity: float
    unit_cost: float
    currency: str
    acquired: date


@dataclass(frozen=True)
class JournalEntry:
    timestamp: datetime
    event_type: str
    symbol: str | None
    quantity: float
    amount: float
    currency: str
    memo: str = ""


@dataclass(frozen=True)
class RealizedTrade:
    symbol: str
    quantity: float
    proceeds: float
    cost_basis: float
    realized_pnl: float
    currency: str


class PortfolioLedger:
    """Small long-only ledger with cash balances and FIFO cost basis.

    The ledger records trade cash movements in settlement currency. FX marks
    translate every currency cash balance and security market value to one
    reporting currency when ``net_asset_value`` is called.
    """
    def __init__(self, base_currency: str = "USD", initial_cash: Number = 0.0):
        if not base_currency or len(base_currency) != 3:
            raise FinanceInputError("base_currency must be a three-letter code")
        self.base_currency = base_currency.upper()
        self.cash: dict[str, float] = {self.base_currency: finite(initial_cash)}
        self.lots: dict[str, list[Lot]] = {}
        self.prices: dict[str, float] = {}
        self.security_currency: dict[str, str] = {}
        self.fx_to_base: dict[str, float] = {self.base_currency: 1.0}
        self.journal: list[JournalEntry] = []
        self.realized: list[RealizedTrade] = []
        self.external_flows: list[tuple[datetime, float]] = []

    def set_fx(self, currency: str, base_per_unit: Number) -> None:
        """Set reporting-currency value of one unit of foreign currency."""
        code = _currency(currency)
        rate = positive(base_per_unit, "base_per_unit")
        if code == self.base_currency and abs(rate-1.0) > 1e-10:
            raise FinanceInputError("base currency FX rate must equal one")
        self.fx_to_base[code] = rate

    def deposit(self, amount: Number, currency: str | None = None,
                *, timestamp: datetime | None = None, external: bool = True) -> None:
        """Deposit cash; external flows are tracked separately from investment P&L."""
        code = _currency(currency or self.base_currency)
        value = positive(amount, "amount", allow_zero=True)
        self.cash[code] = self.cash.get(code, 0.0)+value
        when = timestamp or datetime.now(timezone.utc)
        self.journal.append(JournalEntry(when, "deposit", None, 0.0, value, code))
        if external:
            self.external_flows.append((when, value*self._fx(code)))

    def withdraw(self, amount: Number, currency: str | None = None,
                 *, timestamp: datetime | None = None, external: bool = True) -> None:
        """Withdraw cash, rejecting overdrafts."""
        code = _currency(currency or self.base_currency)
        value = positive(amount, "amount", allow_zero=True)
        if self.cash.get(code, 0.0) < value-1e-10:
            raise FinanceInputError("insufficient cash")
        self.cash[code] = self.cash.get(code, 0.0)-value
        when = timestamp or datetime.now(timezone.utc)
        self.journal.append(JournalEntry(when, "withdrawal", None, 0.0, -value, code))
        if external:
            self.external_flows.append((when, -value*self._fx(code)))

    def trade(self, symbol: str, side: str, quantity: Number, price: Number,
              currency: str = "USD", fee: Number = 0.0,
              *, timestamp: datetime | None = None) -> RealizedTrade | None:
        """Execute a simulated spot equity trade and update FIFO cost lots."""
        ticker, code = symbol.strip().upper(), _currency(currency)
        if not ticker:
            raise FinanceInputError("symbol cannot be blank")
        qty, px, commission = positive(quantity), positive(price), positive(fee, allow_zero=True)
        direction = side.lower()
        if direction not in {"buy", "sell"}:
            raise FinanceInputError("side must be buy or sell")
        when = timestamp or datetime.now(timezone.utc)
        lots = self.lots.setdefault(ticker, [])
        prior_currency = self.security_currency.get(ticker)
        if prior_currency and prior_currency != code:
            raise FinanceInputError("one symbol cannot change settlement currency within this ledger")
        self.security_currency[ticker] = code
        notional = qty*px
        if direction == "buy":
            total = notional+commission
            if self.cash.get(code, 0.0) < total-1e-10:
                raise FinanceInputError("insufficient settled cash for simulated buy")
            self.cash[code] = self.cash.get(code, 0.0)-total
            lots.append(Lot(ticker, qty, (notional+commission)/qty, code, when.date()))
            realized = None
        else:
            held = sum(lot.quantity for lot in lots)
            if qty > held+1e-10:
                raise FinanceInputError("sale exceeds long inventory; short accounting is not enabled")
            remaining, cost_basis = qty, 0.0
            while remaining > 1e-10:
                lot = lots[0]
                matched = min(remaining, lot.quantity)
                cost_basis += matched*lot.unit_cost
                lot.quantity -= matched
                remaining -= matched
                if lot.quantity <= 1e-10:
                    lots.pop(0)
            net_proceeds = notional-commission
            self.cash[code] = self.cash.get(code, 0.0)+net_proceeds
            realized_pnl = net_proceeds-cost_basis
            realized = RealizedTrade(ticker, qty, net_proceeds, cost_basis,
                                     realized_pnl, code)
            self.realized.append(realized)
        self.prices[ticker] = px
        self.journal.append(JournalEntry(when, f"trade_{direction}", ticker,
                                         qty if direction == "buy" else -qty,
                                         (-notional-commission) if direction == "buy" else (notional-commission),
                                         code, f"price={px:.10g}; fee={commission:.10g}"))
        return realized

    def apply_split(self, symbol: str, split_ratio: Number,
                    *, timestamp: datetime | None = None) -> None:
        """Apply a split ratio expressed as new shares per old share."""
        ticker = symbol.strip().upper()
        ratio = positive(split_ratio, "split_ratio")
        if ticker not in self.lots:
            raise FinanceInputError("unknown symbol")
        for lot in self.lots[ticker]:
            lot.quantity *= ratio
            lot.unit_cost /= ratio
        if ticker in self.prices:
            self.prices[ticker] /= ratio
        self.journal.append(JournalEntry(timestamp or datetime.now(timezone.utc), "stock_split",
                                         ticker, ratio, 0.0,
                                         self.security_currency[ticker], f"ratio={ratio}"))

    def dividend(self, symbol: str, amount_per_share: Number,
                 *, withholding_rate: Number = 0.0,
                 timestamp: datetime | None = None) -> float:
        """Credit net cash dividend based on currently held settled inventory."""
        ticker = symbol.strip().upper()
        if ticker not in self.security_currency:
            raise FinanceInputError("unknown symbol")
        rate, withholding = finite(amount_per_share), finite(withholding_rate)
        if rate < 0 or not 0 <= withholding <= 1:
            raise FinanceInputError("dividend must be non-negative and withholding in [0,1]")
        quantity = self.position(ticker)
        gross = quantity*rate
        net = gross*(1-withholding)
        currency = self.security_currency[ticker]
        self.cash[currency] = self.cash.get(currency, 0.0)+net
        self.journal.append(JournalEntry(timestamp or datetime.now(timezone.utc), "dividend",
                                         ticker, quantity, net, currency,
                                         f"gross={gross}; withholding={withholding}"))
        return net

    def accrue_cash_interest(self, currency: str, annual_rate: Number,
                             year_fraction: Number,
                             *, timestamp: datetime | None = None) -> float:
        """Accrue simple interest on positive or negative cash balance."""
        code, rate, tau = _currency(currency), finite(annual_rate), positive(year_fraction, allow_zero=True)
        amount = self.cash.get(code, 0.0)*rate*tau
        self.cash[code] = self.cash.get(code, 0.0)+amount
        self.journal.append(JournalEntry(timestamp or datetime.now(timezone.utc), "cash_interest",
                                         None, 0.0, amount, code))
        return amount

    def mark(self, prices: Mapping[str, Number],
             fx_to_base: Mapping[str, Number] | None = None) -> None:
        """Update security prices and optionally currency conversion marks."""
        for symbol, price in prices.items():
            ticker = symbol.strip().upper()
            if ticker not in self.security_currency:
                raise FinanceInputError(f"cannot mark unknown symbol {ticker}")
            self.prices[ticker] = positive(price, f"prices[{ticker}]")
        for currency, fx in (fx_to_base or {}).items():
            self.set_fx(currency, fx)

    def position(self, symbol: str) -> float:
        """Current long inventory quantity for one symbol."""
        ticker = symbol.strip().upper()
        return sum(lot.quantity for lot in self.lots.get(ticker, []))

    def average_cost(self, symbol: str) -> float:
        """Weighted-average remaining unit cost for reporting purposes."""
        ticker = symbol.strip().upper()
        lots = self.lots.get(ticker, [])
        quantity = sum(lot.quantity for lot in lots)
        if quantity <= 0:
            return 0.0
        return sum(lot.quantity*lot.unit_cost for lot in lots)/quantity

    def market_value(self, symbol: str) -> float:
        """Market value in security settlement currency."""
        ticker = symbol.strip().upper()
        if ticker not in self.prices:
            raise FinanceInputError(f"missing mark for {ticker}")
        return self.position(ticker)*self.prices[ticker]

    def net_asset_value(self) -> float:
        """Translate cash and marked positions into reporting currency."""
        cash_value = sum(amount*self._fx(currency) for currency, amount in self.cash.items())
        position_value = 0.0
        for symbol in self.lots:
            if self.position(symbol) > 0:
                position_value += self.market_value(symbol)*self._fx(self.security_currency[symbol])
        return cash_value+position_value

    def unrealized_pnl(self) -> float:
        """Unrealized P&L in base currency from FIFO cost basis and current marks."""
        result = 0.0
        for symbol, lots in self.lots.items():
            currency = self.security_currency[symbol]
            mark = self.prices.get(symbol)
            if mark is None and lots:
                raise FinanceInputError(f"missing mark for {symbol}")
            result += sum(lot.quantity*(mark-lot.unit_cost) for lot in lots)*self._fx(currency)
        return result

    def realized_pnl(self) -> float:
        """Cumulative realized trade P&L translated at current FX marks."""
        return sum(item.realized_pnl*self._fx(item.currency) for item in self.realized)

    def cash_balances_base(self) -> dict[str, float]:
        """Per-currency cash values converted to reporting currency."""
        return {currency: amount*self._fx(currency) for currency, amount in self.cash.items()}

    def positions(self) -> dict[str, dict[str, float | str]]:
        """Snapshot of quantity, average cost, mark, market value and currency."""
        output = {}
        for symbol in self.lots:
            quantity = self.position(symbol)
            if quantity <= 0:
                continue
            mark = self.prices.get(symbol)
            if mark is None:
                raise FinanceInputError(f"missing mark for {symbol}")
            output[symbol] = {"quantity": quantity, "average_cost": self.average_cost(symbol),
                              "mark": mark, "market_value": quantity*mark,
                              "currency": self.security_currency[symbol]}
        return output

    def journal_report(self) -> dict[str, float]:
        """Basic activity counts and cash movement totals by transaction type."""
        totals: dict[str, float] = {}
        for entry in self.journal:
            key = entry.event_type
            totals[key] = totals.get(key, 0.0)+entry.amount
        return totals

    def _fx(self, currency: str) -> float:
        code = _currency(currency)
        if code not in self.fx_to_base:
            raise FinanceInputError(f"missing FX mark for currency {code}")
        return self.fx_to_base[code]


def _currency(value: str) -> str:
    if not isinstance(value, str) or len(value.strip()) != 3 or not value.strip().isalpha():
        raise FinanceInputError("currency must be a three-letter alphabetic code")
    return value.strip().upper()


def time_weighted_return(nav_values: Sequence[Number],
                         external_flows: Sequence[Number]) -> float:
    """Geometrically link subperiod returns after adjusting end-of-period flows.

    ``external_flows[i]`` is the cash flow occurring immediately after NAV
    observation i; the last NAV point is assumed to be after its flow.
    """
    nav = tuple(map(finite, nav_values))
    flows = tuple(map(finite, external_flows))
    if len(nav) < 2 or len(flows) != len(nav)-1:
        raise FinanceInputError("need n NAV observations and n-1 intervening external flows")
    factors = []
    for i, flow in enumerate(flows):
        if nav[i] <= 0 or nav[i+1] <= 0 or nav[i]+flow <= 0:
            raise FinanceInputError("NAV and post-flow capital must remain positive")
        factors.append(nav[i+1]/(nav[i]+flow))
    wealth = math.prod(factors)
    return wealth-1


def money_weighted_return(cashflows: Sequence[tuple[float, float]],
                          *, lower: Number = -0.99, upper: Number = 10.0) -> float:
    """Solve annualized IRR for dated-in-years investor cash flows."""
    from .core import bisect
    flows = [(positive(t, "time", allow_zero=True), finite(v)) for t, v in cashflows]
    if len(flows) < 2 or not any(v < 0 for _,v in flows) or not any(v > 0 for _,v in flows):
        raise FinanceInputError("IRR requires at least one negative and one positive cash flow")
    def npv(rate: float) -> float:
        if rate <= -1:
            return math.inf
        return sum(value/(1+rate)**time for time, value in flows)
    return bisect(npv, 0.0, finite(lower), finite(upper))
