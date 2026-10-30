"""Multiple-testing and backtest-overfitting diagnostics."""
from __future__ import annotations
from math import erf, exp, log, sqrt
from itertools import combinations
from typing import Sequence
from ..core import finite_series, mean, variance, normal_cdf, normal_ppf, quantile


def _skew_kurtosis(x: Sequence[float]) -> tuple[float,float]:
    m=mean(x); s2=sum((v-m)**2 for v in x)/len(x)
    if s2==0: return 0.0,3.0
    s=sqrt(s2); skew=sum(((v-m)/s)**3 for v in x)/len(x)
    kurt=sum(((v-m)/s)**4 for v in x)/len(x)
    return skew,kurt


def probabilistic_sharpe(returns: Sequence[float], benchmark: float = 0.0,
                         periods_per_year: float = 252.0) -> float:
    """Probabilistic Sharpe that annualizes the mean and volatility consistently."""
    x=finite_series(returns,name="returns",min_length=3)
    if periods_per_year<=0: raise ValueError("periods_per_year must be positive")
    s=sqrt(variance(x)); sr=(mean(x)-benchmark)/(s+1e-30)*sqrt(periods_per_year)
    skew,kurt=_skew_kurtosis(x); n=len(x)
    # Standard error of annualized Sharpe under non-normal returns.
    se=sqrt(max(1e-15,(1.0- skew*sr/sqrt(periods_per_year) + (kurt-1.0)*sr*sr/(4.0*periods_per_year))/(n-1))) * sqrt(periods_per_year)
    return normal_cdf(sr/max(se,1e-15))


def deflated_sharpe(returns: Sequence[float], trials: int, *, periods_per_year: float = 252.0,
                    expected_max_sharpe: float | None = None) -> dict:
    """Deflated Sharpe approximation adjusting for the number of trials.

    ``expected_max_sharpe`` can be supplied from a more appropriate null model;
    otherwise an independent-Gaussian-trials approximation is used.
    """
    x=finite_series(returns,min_length=3)
    if trials<1 or periods_per_year<=0: raise ValueError("invalid trials or annualization")
    skew,kurt=_skew_kurtosis(x); n=len(x)
    sr=mean(x)/(sqrt(variance(x))+1e-30)*sqrt(periods_per_year)
    if expected_max_sharpe is None:
        euler=0.5772156649015329
        z=normal_ppf(1.0-1.0/trials) if trials>1 else 0.0
        z2=normal_ppf(1.0-1.0/(trials*2.718281828)) if trials>1 else 0.0
        expected_max_sharpe=sqrt(max(0.0,(1.0- euler)*z*z+euler*z2*z2))*sqrt(periods_per_year/max(n,1))
    se=sqrt(max(1e-15,(1-skew*sr/sqrt(periods_per_year)+(kurt-1)*sr*sr/(4*periods_per_year))/(n-1)))*sqrt(periods_per_year)
    probability=normal_cdf((sr-expected_max_sharpe)/max(se,1e-15))
    return {"sharpe":sr,"expected_max_sharpe":expected_max_sharpe,"deflated_sharpe_probability":probability,
            "trials":trials,"observations":n,"skewness":skew,"kurtosis":kurt}


def probabilistic_performance_ratio(returns: Sequence[float], benchmark: float = 0.0,
                                    periods_per_year: float = 252.0) -> float:
    """Alias-style API emphasizing that benchmark may be a target return."""
    return probabilistic_sharpe(returns,benchmark,periods_per_year)


def reality_check(benchmark_returns: Sequence[float], strategy_returns: Sequence[Sequence[float]],
                  *, bootstrap_samples: int = 500, block_size: int = 5, seed: int = 7) -> dict:
    """White's Reality Check using circular moving-block bootstrap.

    Returns a conservative p-value for the best mean excess return. The finite
    bootstrap estimate is a research diagnostic, not an exact hypothesis test.
    """
    base=finite_series(benchmark_returns,name="benchmark",min_length=10)
    strategies=[finite_series(s,name="strategy",min_length=10) for s in strategy_returns]
    if not strategies or any(len(s)!=len(base) for s in strategies): raise ValueError("strategies must match benchmark length")
    if bootstrap_samples<1 or block_size<1 or block_size>len(base): raise ValueError("invalid bootstrap controls")
    excess=[[s[i]-base[i] for i in range(len(base))] for s in strategies]
    observed=max(mean(s) for s in excess)
    centered=[[v-mean(s) for v in s] for s in excess]
    import random
    rng=random.Random(seed); exceed=0; n=len(base); blocks=(n+block_size-1)//block_size
    for _ in range(bootstrap_samples):
        indices=[]
        for _b in range(blocks):
            start=rng.randrange(n)
            indices.extend((start+j)%n for j in range(block_size))
        indices=indices[:n]
        stat=max(sum(s[i] for i in indices)/n for s in centered)
        exceed += stat>=observed
    return {"observed_best_mean_excess":observed,"p_value":(exceed+1)/(bootstrap_samples+1),
            "bootstrap_samples":bootstrap_samples,"block_size":block_size,"trials":len(strategies)}


def probability_of_backtest_overfitting(performance_matrix: Sequence[Sequence[float]]) -> dict:
    """CSCV PBO estimate from strategy x contiguous time-block performance.

    Each split chooses half of the blocks in-sample and ranks the in-sample
    winner out-of-sample. Inputs must contain already-computed block scores.
    """
    rows=[finite_series(row,name="strategy block scores",min_length=4) for row in performance_matrix]
    if not rows or any(len(r)!=len(rows[0]) for r in rows): raise ValueError("score matrix must be rectangular")
    b=len(rows[0])
    if b%2: raise ValueError("number of time blocks must be even")
    half=b//2; logits=[]
    for chosen in combinations(range(b),half):
        test=[i for i in range(b) if i not in chosen]
        in_scores=[sum(row[i] for i in chosen)/half for row in rows]
        winner=max(range(len(rows)),key=in_scores.__getitem__)
        out_scores=sorted(sum(row[i] for i in test)/half for row in rows)
        rank=sum(v<=sum(rows[winner][i] for i in test)/half for v in out_scores)
        percentile=(rank-0.5)/len(rows)
        clipped=min(1-1e-6,max(1e-6,percentile))
        logits.append(log(clipped/(1-clipped)))
    return {"pbo":sum(v<0 for v in logits)/len(logits),"logits":tuple(logits),
            "splits":len(logits),"interpretation":"fraction of CSCV splits where the in-sample winner ranks below the OOS median"}


def benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> dict:
    p=finite_series(p_values,name="p_values")
    if any(v<0 or v>1 for v in p) or not 0<q<1: raise ValueError("p-values must be in [0,1] and q in (0,1)")
    order=sorted(range(len(p)),key=p.__getitem__); cutoff=-1
    for rank,idx in enumerate(order,1):
        if p[idx]<=q*rank/len(p): cutoff=rank
    rejected=[False]*len(p)
    for idx in order[:cutoff]: rejected[idx]=True
    adjusted=[1.0]*len(p); running=1.0
    for rank in range(len(p),0,-1):
        idx=order[rank-1]; running=min(running,p[idx]*len(p)/rank); adjusted[idx]=min(1.0,running)
    return {"rejected":tuple(rejected),"adjusted_p_values":tuple(adjusted),"cutoff_rank":cutoff,"q":q}
