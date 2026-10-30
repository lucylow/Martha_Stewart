"""Distribution functions and statistical diagnostics used across the package."""
from __future__ import annotations
from math import exp, lgamma, log, sqrt
from typing import Sequence
from .core import finite_series,mean,variance,normal_cdf


def regularized_gamma_q(a: float, x: float, *, tolerance: float = 1e-12, max_iter: int = 10000) -> float:
    """Regularized upper incomplete gamma Q(a,x), using series/continued fraction."""
    if a<=0 or x<0: raise ValueError("require a > 0 and x >= 0")
    if x==0: return 1.0
    if x<a+1:
        ap=a; term=1.0/a; total=term
        for _ in range(max_iter):
            ap+=1; term*=x/ap; total+=term
            if abs(term)<abs(total)*tolerance: break
        p=total*exp(-x+a*log(x)-lgamma(a))
        return min(1.0,max(0.0,1.0-p))
    b=x+1-a; c=1e300; d=1.0/max(b,1e-300); h=d
    for i in range(1,max_iter+1):
        an=-i*(i-a); b+=2.0; d=an*d+b
        if abs(d)<1e-300: d=1e-300
        c=b+an/c
        if abs(c)<1e-300: c=1e-300
        d=1.0/d; delta=d*c; h*=delta
        if abs(delta-1.0)<tolerance: break
    return min(1.0,max(0.0,exp(-x+a*log(x)-lgamma(a))*h))


def chi_square_survival(statistic: float, degrees_freedom: int) -> float:
    if statistic<0 or degrees_freedom<1: raise ValueError("invalid chi-square statistic or df")
    return regularized_gamma_q(degrees_freedom/2.0,statistic/2.0)


def jarque_bera(values: Sequence[float]) -> dict:
    x=finite_series(values,name="values",min_length=8); n=len(x); m=mean(x)
    s2=sum((v-m)**2 for v in x)/n
    if s2==0: return {"statistic":0.0,"p_value_approx":1.0,"skewness":0.0,"excess_kurtosis":-3.0}
    skew=sum((v-m)**3 for v in x)/n/(s2**1.5)
    kurt=sum((v-m)**4 for v in x)/n/(s2*s2)
    jb=n/6.0*(skew*skew+(kurt-3)**2/4.0)
    return {"statistic":jb,"p_value_approx":chi_square_survival(jb,2),"skewness":skew,"excess_kurtosis":kurt-3}


def mann_whitney_u(sample_a: Sequence[float], sample_b: Sequence[float]) -> dict:
    """Two-sided large-sample Mann-Whitney U with tie-corrected normal variance."""
    a=finite_series(sample_a); b=finite_series(sample_b); n1,n2=len(a),len(b)
    pooled=a+b; ordered=sorted(pooled); ranks=[0.0]*len(pooled); i=0; tie_sum=0.0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j]==ordered[i]: j+=1
        avg=(i+1+j)/2
        for k in range(i,j): ranks[k]=avg
        tie_sum+=(j-i)**3-(j-i); i=j
    # Recover original ranks by stable value-index pairs.
    indexed=sorted(enumerate(pooled),key=lambda z:z[1]); r=[0.0]*len(pooled)
    for rank_idx,(orig,_) in enumerate(indexed): r[orig]=ranks[rank_idx]
    u1=sum(r[:n1])-n1*(n1+1)/2; u2=n1*n2-u1; u=min(u1,u2)
    N=n1+n2; mu=n1*n2/2
    tie_factor=1-tie_sum/(N**3-N) if N>1 else 1.0
    sd=sqrt(max(1e-30,n1*n2*(N+1)*tie_factor/12))
    z=(u-mu+0.5)/sd
    p=2*(1-normal_cdf(abs(z)))
    return {"u":u,"u_a":u1,"u_b":u2,"z":z,"p_value_approx":p}


def bootstrap_mean(values: Sequence[float], samples: int = 1000, confidence: float = 0.95,
                   seed: int = 7) -> dict:
    import random
    x=finite_series(values,name="values")
    if samples<1 or not 0<confidence<1: raise ValueError("invalid bootstrap configuration")
    rng=random.Random(seed); estimates=[]
    for _ in range(samples): estimates.append(mean([x[rng.randrange(len(x))] for _ in x]))
    tail=(1-confidence)/2
    from .core import quantile
    return {"mean":mean(x),"lower":quantile(estimates,tail),"upper":quantile(estimates,1-tail),
            "confidence":confidence,"samples":samples,"method":"iid percentile bootstrap"}
