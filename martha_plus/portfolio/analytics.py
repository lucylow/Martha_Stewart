"""Portfolio time-series analytics and drawdown episode summaries."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence
from ..math_utils import max_drawdown, sharpe_ratio, annualized_volatility
from ..types import PortfolioSnapshot

@dataclass(frozen=True, slots=True)
class DrawdownEpisode:
    peak_time: datetime
    trough_time: datetime
    recovery_time: datetime | None
    drawdown_fraction: float
    duration_periods: int


def snapshot_returns(snapshots: Sequence[PortfolioSnapshot]) -> list[float]:
    ordered = sorted(snapshots, key=lambda s: s.timestamp)
    returns_: list[float] = []
    for prior, current in zip(ordered, ordered[1:]):
        if prior.equity <= 0:
            raise ValueError("prior equity must be positive")
        returns_.append(current.equity / prior.equity - 1.0)
    return returns_


def performance_summary(snapshots: Sequence[PortfolioSnapshot], periods_per_year: float = 252.0) -> dict:
    ordered = sorted(snapshots, key=lambda s: s.timestamp)
    if not ordered:
        return {"observations": 0, "total_return": 0.0, "sharpe": 0.0,
                "annualized_volatility": 0.0, "max_drawdown": 0.0}
    returns_ = snapshot_returns(ordered)
    dd, peak, trough = max_drawdown([s.equity for s in ordered])
    return {"observations": len(ordered),
            "total_return": ordered[-1].equity / ordered[0].equity - 1.0,
            "sharpe": sharpe_ratio(returns_, periods_per_year=periods_per_year),
            "annualized_volatility": annualized_volatility(returns_, periods_per_year),
            "max_drawdown": dd, "drawdown_peak_index": peak, "drawdown_trough_index": trough,
            "average_gross_exposure": sum(s.gross_exposure for s in ordered) / len(ordered),
            "average_net_exposure": sum(s.net_exposure for s in ordered) / len(ordered)}


def drawdown_episodes(snapshots: Sequence[PortfolioSnapshot]) -> tuple[DrawdownEpisode, ...]:
    ordered = sorted(snapshots, key=lambda s: s.timestamp)
    if not ordered:
        return ()
    high = ordered[0].equity
    peak_index = 0
    open_episode: tuple[int, int, float] | None = None
    episodes: list[DrawdownEpisode] = []
    for i, snapshot in enumerate(ordered):
        if snapshot.equity >= high:
            if open_episode is not None:
                ep_peak, ep_trough, ep_depth = open_episode
                episodes.append(DrawdownEpisode(ordered[ep_peak].timestamp,
                                                ordered[ep_trough].timestamp,
                                                snapshot.timestamp, ep_depth, i - ep_peak))
                open_episode = None
            high = snapshot.equity
            peak_index = i
        else:
            depth = 1 - snapshot.equity / high if high > 0 else 0.0
            if open_episode is None:
                open_episode = (peak_index, i, depth)
            elif depth > open_episode[2]:
                open_episode = (open_episode[0], i, depth)
    if open_episode is not None:
        ep_peak, ep_trough, ep_depth = open_episode
        episodes.append(DrawdownEpisode(ordered[ep_peak].timestamp,
                                        ordered[ep_trough].timestamp, None, ep_depth,
                                        len(ordered) - 1 - ep_peak))
    return tuple(episodes)
