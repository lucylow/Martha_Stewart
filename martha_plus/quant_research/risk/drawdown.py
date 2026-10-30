"""Drawdown-tail measures including conditional drawdown-at-risk."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series,quantile
from ..performance.metrics import drawdown_series,max_drawdown


def conditional_drawdown_at_risk(returns: Sequence[float], confidence: float = 0.95) -> dict:
    """Mean of the worst drawdown observations (positive loss convention)."""
    if not 0.5<confidence<1: raise ValueError("confidence must be in (0.5,1)")
    dd=[-v for v in drawdown_series(returns)]
    threshold=quantile(dd,confidence)
    tail=[v for v in dd if v>=threshold]
    return {"drawdown_at_risk":threshold,"conditional_drawdown_at_risk":sum(tail)/len(tail),
            "confidence":confidence,"observations":len(dd)}


def average_drawdown(returns: Sequence[float]) -> float:
    dd=[-v for v in drawdown_series(returns)]
    return sum(dd)/len(dd)


def drawdown_duration(returns: Sequence[float]) -> dict:
    dd=drawdown_series(returns); durations=[]; current=0
    for value in dd:
        if value<0: current+=1
        elif current:
            durations.append(current); current=0
    if current: durations.append(current)
    return {"completed_drawdowns":len(durations),"max_duration":max(durations,default=0),
            "average_duration":sum(durations)/len(durations) if durations else 0.0,
            "currently_in_drawdown":current>0,"current_duration":current}


def drawdown_summary(returns: Sequence[float], confidence: float = 0.95) -> dict:
    return {"max_drawdown":max_drawdown(returns),"cdar":conditional_drawdown_at_risk(returns,confidence),
            "average_drawdown":average_drawdown(returns),"duration":drawdown_duration(returns)}
