"""Portfolio concentration, turnover, and drift analytics."""
from __future__ import annotations
from math import isfinite, sqrt
from typing import Mapping, Sequence


def concentration_report(weights: Mapping[str, float]) -> dict[str, object]:
    normalized = {str(s).strip().upper(): float(w) for s, w in weights.items()}
    if any(not isfinite(w) for w in normalized.values()):
        raise ValueError("weights must be finite")
    gross = sum(abs(w) for w in normalized.values())
    net = sum(normalized.values())
    absolute = {s: abs(w) / gross for s, w in normalized.items()} if gross else {s: 0.0 for s in normalized}
    hhi = sum(w * w for w in absolute.values())
    return {"positions": len([w for w in normalized.values() if abs(w) > 1e-12]),
            "gross_exposure": gross, "net_exposure": net, "hhi": hhi,
            "effective_number_of_positions": 1 / hhi if hhi > 0 else 0.0,
            "largest_weight": max(absolute.values(), default=0.0),
            "top_five_gross_fraction": sum(sorted(absolute.values(), reverse=True)[:5]),
            "weights": absolute}


def turnover(previous_weights: Mapping[str, float], current_weights: Mapping[str, float], *,
             normalize_one_way: bool = False) -> dict[str, float]:
    symbols = set(previous_weights) | set(current_weights)
    changes = {s: float(current_weights.get(s, 0.0)) - float(previous_weights.get(s, 0.0)) for s in symbols}
    if any(not isfinite(v) for v in changes.values()):
        raise ValueError("weights must be finite")
    gross = sum(abs(v) for v in changes.values())
    return {"gross_turnover": gross / 2 if normalize_one_way else gross,
            "buy_turnover": sum(max(0.0, d) for d in changes.values()),
            "sell_turnover": sum(max(0.0, -d) for d in changes.values()),
            "changed_positions": sum(abs(v) > 1e-10 for v in changes.values()),
            "net_weight_change": sum(changes.values())}


def weight_drift(target_weights: Mapping[str, float], current_weights: Mapping[str, float]) -> dict[str, object]:
    symbols = sorted(set(target_weights) | set(current_weights))
    target = {s: float(target_weights.get(s, 0.0)) for s in symbols}
    current = {s: float(current_weights.get(s, 0.0)) for s in symbols}
    if any(not isfinite(x) for x in list(target.values()) + list(current.values())):
        raise ValueError("weights must be finite")
    delta = {s: current[s] - target[s] for s in symbols}
    gross = sum(abs(v) for v in delta.values())
    return {"l1_drift": gross, "l2_drift": sqrt(sum(v * v for v in delta.values())),
            "max_abs_drift": max((abs(v) for v in delta.values()), default=0.0),
            "by_symbol": {s: {"target": target[s], "current": current[s], "delta": delta[s]} for s in symbols}}


def sector_exposure(weights: Mapping[str, float], sectors: Mapping[str, str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for symbol, weight in weights.items():
        w = float(weight)
        if not isfinite(w):
            raise ValueError("weights must be finite")
        sector = sectors.get(symbol, "unknown")
        result[sector] = result.get(sector, 0.0) + abs(w)
    return dict(sorted(result.items()))


def rolling_turnover(weight_history: Sequence[Mapping[str, float]]) -> list[float | None]:
    result: list[float | None] = [None]
    for previous, current in zip(weight_history, weight_history[1:]):
        result.append(sum(abs(float(current.get(s, 0.0)) - float(previous.get(s, 0.0)))
                          for s in set(previous) | set(current)))
    return result


def allocation_summary(weights: Mapping[str, float], sectors: Mapping[str, str] | None = None) -> dict[str, object]:
    concentration = concentration_report(weights)
    return {"concentration": concentration,
            "sector_gross_exposure": sector_exposure(weights, sectors or {}),
            "short_positions": sorted(s for s, w in weights.items() if float(w) < 0),
            "unallocated_weight": 1.0 - sum(float(w) for w in weights.values())}
