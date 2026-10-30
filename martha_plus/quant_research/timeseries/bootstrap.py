"""Dependence-preserving bootstrap methods for time-series inference."""
from __future__ import annotations
import random
from math import sqrt
from typing import Callable,Sequence
from ..core import finite_series,mean,quantile


def circular_block_bootstrap(values: Sequence[float], samples: int = 1000,
                             block_size: int = 5, seed: int = 7) -> list[tuple[float,...]]:
    x=finite_series(values,name="values")
    if samples<1 or block_size<1 or block_size>len(x): raise ValueError("invalid bootstrap controls")
    rng=random.Random(seed); n=len(x); blocks=(n+block_size-1)//block_size; result=[]
    for _ in range(samples):
        out=[]
        for _b in range(blocks):
            start=rng.randrange(n); out.extend(x[(start+j)%n] for j in range(block_size))
        result.append(tuple(out[:n]))
    return result


def stationary_bootstrap(values: Sequence[float], samples: int = 1000,
                         expected_block: float = 5.0, seed: int = 7) -> list[tuple[float,...]]:
    """Politis-Romano stationary bootstrap with geometrically distributed blocks."""
    x=finite_series(values,name="values")
    if samples<1 or expected_block<1: raise ValueError("invalid stationary bootstrap controls")
    rng=random.Random(seed); n=len(x); restart=1.0/expected_block; result=[]
    for _ in range(samples):
        idx=rng.randrange(n); out=[]
        for t in range(n):
            if t==0 or rng.random()<restart: idx=rng.randrange(n)
            else: idx=(idx+1)%n
            out.append(x[idx])
        result.append(tuple(out))
    return result


def bootstrap_statistic(values: Sequence[float], statistic: Callable[[Sequence[float]],float],
                        samples: int = 1000, block_size: int = 5, confidence: float = 0.95,
                        seed: int = 7) -> dict:
    x=finite_series(values,name="values")
    if not 0<confidence<1: raise ValueError("confidence must be in (0,1)")
    draws=circular_block_bootstrap(x,samples,block_size,seed)
    estimates=[float(statistic(draw)) for draw in draws]
    tail=(1-confidence)/2
    center=statistics_median(estimates)
    return {"estimate":float(statistic(x)),"bootstrap_mean":mean(estimates),"bootstrap_median":center,
            "standard_error":sqrt(sum((v-mean(estimates))**2 for v in estimates)/max(1,len(estimates)-1)),
            "lower":quantile(estimates,tail),"upper":quantile(estimates,1-tail),
            "confidence":confidence,"samples":samples,"block_size":block_size}


def statistics_median(values: Sequence[float]) -> float:
    x=sorted(values); n=len(x)
    if n==0: raise ValueError("cannot take median of empty data")
    return x[n//2] if n%2 else (x[n//2-1]+x[n//2])/2


def permutation_test_mean_difference(sample_a: Sequence[float], sample_b: Sequence[float],
                                      permutations: int = 2000, seed: int = 7) -> dict:
    """Two-sided randomization test for mean difference under exchangeability."""
    a=finite_series(sample_a); b=finite_series(sample_b)
    if permutations<1: raise ValueError("permutations must be positive")
    observed=mean(a)-mean(b); pooled=a+b; n=len(a); rng=random.Random(seed); exceed=0
    for _ in range(permutations):
        order=list(range(len(pooled))); rng.shuffle(order)
        left=[pooled[i] for i in order[:n]]; right=[pooled[i] for i in order[n:]]
        exceed+=abs(mean(left)-mean(right))>=abs(observed)
    return {"observed_difference":observed,"p_value":(exceed+1)/(permutations+1),
            "permutations":permutations,"assumption":"exchangeability under null"}


def bootstrap_correlation(x: Sequence[float], y: Sequence[float], samples: int = 1000,
                          block_size: int = 5, confidence: float = 0.95, seed: int = 7) -> dict:
    from ..core import correlation
    a=finite_series(x); b=finite_series(y)
    if len(a)!=len(b): raise ValueError("series lengths must match")
    draws_a=circular_block_bootstrap(a,samples,block_size,seed)
    draws_b=circular_block_bootstrap(b,samples,block_size,seed+1)
    estimates=[correlation(u,v) for u,v in zip(draws_a,draws_b)]
    tail=(1-confidence)/2
    return {"correlation":correlation(a,b),"lower":quantile(estimates,tail),"upper":quantile(estimates,1-tail),
            "bootstrap_std":sqrt(sum((v-mean(estimates))**2 for v in estimates)/max(1,samples-1)),"samples":samples}
