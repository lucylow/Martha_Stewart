"""Financial-ML event sampling, triple-barrier labels and uniqueness weights.

All event end indices are inclusive. When both horizontal barriers are crossed
on one OHLC bar, the configurable tie rule defaults to the conservative stop.
These routines create research labels only and do not imply executable fills.
"""
from __future__ import annotations
from math import log
from typing import Sequence
from ..core import finite_series


def fixed_horizon_labels(prices: Sequence[float], horizon: int = 1,
                         threshold: float = 0.0) -> dict:
    p=finite_series(prices,name="prices",min_length=3)
    if any(v<=0 for v in p) or horizon<1 or horizon>=len(p) or threshold<0: raise ValueError("invalid prices, horizon or threshold")
    returns=[log(p[i+horizon]/p[i]) for i in range(len(p)-horizon)]
    labels=[1 if r>threshold else -1 if r< -threshold else 0 for r in returns]
    starts=tuple(range(len(labels))); ends=tuple(i+horizon for i in starts)
    return {"labels":tuple(labels),"forward_returns":tuple(returns),"event_start":starts,"event_end":ends,
            "horizon":horizon,"threshold":threshold}


def triple_barrier_labels(close: Sequence[float], volatility: Sequence[float],
                          *, profit_take: float = 2.0, stop_loss: float = 2.0,
                          max_holding: int = 20, side: Sequence[int] | None = None,
                          high: Sequence[float] | None = None,
                          low: Sequence[float] | None = None,
                          tie_rule: str = "stop") -> dict:
    """Generate triple-barrier labels using close prices or optional OHLC paths.

    `volatility[i]` is the event-time return scale (for example daily sigma).
    A positive side means the primary signal is long; negative means short.
    """
    c=finite_series(close,name="close",min_length=3); v=finite_series(volatility,name="volatility")
    n=len(c)
    if len(v)!=n or any(x<=0 for x in c) or any(x<0 for x in v): raise ValueError("close/volatility lengths or values invalid")
    if profit_take<=0 or stop_loss<=0 or max_holding<1 or tie_rule not in {"stop","profit"}: raise ValueError("invalid barrier settings")
    h=c if high is None else finite_series(high,name="high")
    l=c if low is None else finite_series(low,name="low")
    if len(h)!=n or len(l)!=n or any(h[i]<l[i] for i in range(n)): raise ValueError("OHLC arrays invalid")
    sides=[1]*n if side is None else list(map(int,side))
    if len(sides)!=n or any(s not in {-1,0,1} for s in sides): raise ValueError("side must contain -1,0,1 and match close")
    labels=[]; ends=[]; returns=[]; exits=[]; reasons=[]; upper=[]; lower=[]
    for start in range(n-1):
        direction=sides[start]
        if direction==0 or v[start]<=0:
            labels.append(0); ends.append(start); returns.append(0.0); exits.append(c[start]); reasons.append("no_side_or_volatility"); upper.append(None); lower.append(None); continue
        up=c[start]*(1+profit_take*v[start]); dn=c[start]*(1-stop_loss*v[start])
        # For short positions, profit and stop barriers reverse in price space.
        profit_level,stop_level=(up,dn) if direction>0 else (dn,up)
        end=min(n-1,start+max_holding); reason="vertical"; exit_price=c[end]
        for t in range(start+1,end+1):
            profit_hit=(h[t]>=profit_level) if direction>0 else (l[t]<=profit_level)
            stop_hit=(l[t]<=stop_level) if direction>0 else (h[t]>=stop_level)
            if profit_hit and stop_hit:
                reason=tie_rule; exit_price=stop_level if tie_rule=="stop" else profit_level; end=t; break
            if profit_hit:
                reason="profit"; exit_price=profit_level; end=t; break
            if stop_hit:
                reason="stop"; exit_price=stop_level; end=t; break
        signed_return=direction*(exit_price/c[start]-1.0)
        label=1 if signed_return>0 else -1 if signed_return<0 else 0
        labels.append(label); ends.append(end); returns.append(signed_return); exits.append(exit_price)
        reasons.append(reason); upper.append(profit_level); lower.append(stop_level)
    return {"labels":tuple(labels),"signed_returns":tuple(returns),"event_start":tuple(range(n-1)),
            "event_end":tuple(ends),"exit_prices":tuple(exits),"exit_reasons":tuple(reasons),
            "profit_barriers":tuple(upper),"stop_barriers":tuple(lower),
            "tie_rule":tie_rule,"max_holding":max_holding}


def cusum_event_filter(prices: Sequence[float], threshold: float,
                       volatility: Sequence[float] | None = None) -> tuple[int,...]:
    """Symmetric CUSUM filter over log returns; returns sampled event indices."""
    p=finite_series(prices,name="prices",min_length=3)
    if any(v<=0 for v in p) or threshold<=0: raise ValueError("positive prices and threshold required")
    vol=None if volatility is None else finite_series(volatility,name="volatility")
    if vol is not None and len(vol)!=len(p): raise ValueError("volatility length mismatch")
    s_pos=s_neg=0.0; events=[]
    for i in range(1,len(p)):
        ret=log(p[i]/p[i-1]); scale=1.0 if vol is None else max(vol[i],1e-12)
        s_pos=max(0.0,s_pos+ret/scale); s_neg=min(0.0,s_neg+ret/scale)
        if s_pos>=threshold:
            events.append(i); s_pos=0.0; s_neg=0.0
        elif s_neg<=-threshold:
            events.append(i); s_pos=0.0; s_neg=0.0
    return tuple(events)


def meta_labels(primary_side: Sequence[int], realized_returns: Sequence[float]) -> tuple[int,...]:
    """Meta-label whether the primary side had positive signed return."""
    side=list(map(int,primary_side)); ret=finite_series(realized_returns)
    if len(side)!=len(ret) or any(v not in {-1,1} for v in side): raise ValueError("side must be -1/+1 and match returns")
    return tuple(1 if s*r>0 else 0 for s,r in zip(side,ret))


def average_uniqueness(event_start: Sequence[int], event_end: Sequence[int],
                       n_observations: int | None = None) -> tuple[float,...]:
    """Average inverse concurrency over each event's inclusive label interval."""
    starts=list(map(int,event_start)); ends=list(map(int,event_end))
    if not starts or len(starts)!=len(ends) or any(a<0 or b<a for a,b in zip(starts,ends)): raise ValueError("invalid event intervals")
    n=max(ends)+1 if n_observations is None else n_observations
    if n<=max(ends): raise ValueError("n_observations must exceed all event end indices")
    concurrency=[0]*n
    for a,b in zip(starts,ends):
        for t in range(a,b+1): concurrency[t]+=1
    result=[]
    for a,b in zip(starts,ends): result.append(sum(1/concurrency[t] for t in range(a,b+1))/(b-a+1))
    return tuple(result)


def time_decay_weights(event_end: Sequence[int], *, oldest_weight: float = 0.5) -> tuple[float,...]:
    """Linear age decay where latest event receives weight one."""
    e=list(map(int,event_end))
    if not e or oldest_weight<0 or oldest_weight>1: raise ValueError("invalid event ends or oldest_weight")
    oldest=min(e); newest=max(e)
    if newest==oldest: return tuple(1.0 for _ in e)
    return tuple(oldest_weight+(1-oldest_weight)*(v-oldest)/(newest-oldest) for v in e)


def sample_weights(event_start: Sequence[int], event_end: Sequence[int],
                   returns: Sequence[float] | None = None,
                   *, time_decay: float | None = None) -> tuple[float,...]:
    """Combine label uniqueness with optional absolute-return and age weighting."""
    unique=average_uniqueness(event_start,event_end)
    weights=list(unique)
    if returns is not None:
        r=finite_series(returns,name="event_returns")
        if len(r)!=len(weights): raise ValueError("returns must match event count")
        weights=[weights[i]*abs(r[i]) for i in range(len(weights))]
    if time_decay is not None:
        decay=time_decay_weights(event_end,oldest_weight=time_decay)
        weights=[a*b for a,b in zip(weights,decay)]
    total=sum(weights)
    return tuple(v*len(weights)/total for v in weights) if total>0 else tuple(1.0 for _ in weights)
