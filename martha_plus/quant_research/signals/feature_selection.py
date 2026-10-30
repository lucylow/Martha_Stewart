"""Feature stability, ensemble weighting and cross-sectional selection."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series,mean,rank,correlation


def univariate_ic_table(features: Sequence[Sequence[float]], forward_returns: Sequence[float],
                        names: Sequence[str] | None = None, *, rank_based: bool = True) -> list[dict]:
    y=finite_series(forward_returns,name="forward_returns")
    labels=list(names) if names is not None else [f"feature_{i}" for i in range(len(features))]
    if len(labels)!=len(features): raise ValueError("feature names must match feature count")
    results=[]
    for name,feature in zip(labels,features):
        x=finite_series(feature,name=name)
        if len(x)!=len(y): raise ValueError("feature length mismatch")
        ic=correlation(rank(x),rank(y)) if rank_based else correlation(x,y)
        results.append({"name":name,"ic":ic,"abs_ic":abs(ic),"sign":1 if ic>=0 else -1})
    return sorted(results,key=lambda r:r["abs_ic"],reverse=True)


def stability_selection(ic_by_fold: Sequence[Sequence[float]], names: Sequence[str],
                        *, min_frequency: float = 0.6, min_abs_mean_ic: float = 0.01) -> dict:
    """Select features whose IC sign and magnitude are stable across folds."""
    if not ic_by_fold or not names or any(len(row)!=len(names) for row in ic_by_fold):
        raise ValueError("fold x feature IC matrix must be rectangular")
    if not 0<min_frequency<=1 or min_abs_mean_ic<0: raise ValueError("invalid thresholds")
    results=[]
    for j,name in enumerate(names):
        values=[float(row[j]) for row in ic_by_fold]
        signs=[1 if v>=0 else -1 for v in values]
        positive=max(signs.count(1),signs.count(-1))/len(signs)
        m=mean(values)
        results.append({"name":name,"mean_ic":m,"ic_std":sqrt(sum((v-m)**2 for v in values)/max(1,len(values)-1)),
                        "sign_stability":positive,"selected":positive>=min_frequency and abs(m)>=min_abs_mean_ic})
    return {"selected":tuple(r["name"] for r in results if r["selected"]),"features":tuple(results),
            "folds":len(ic_by_fold)}


def ic_weighted_ensemble(features: Sequence[Sequence[float]], ic_estimates: Sequence[float],
                         *, cap: float = 0.5, normalize: bool = True) -> dict:
    """Combine z-scored feature values using signed historical IC weights."""
    from .diagnostics import cross_sectional_zscore
    if len(features)!=len(ic_estimates) or not features: raise ValueError("features and ICs must align")
    if not 0<cap<=1: raise ValueError("cap must be in (0,1]")
    n=len(features[0]); cols=[finite_series(c) for c in features]
    if any(len(c)!=n for c in cols): raise ValueError("features must have equal lengths")
    raw=[float(v) for v in ic_estimates]
    gross=sum(abs(v) for v in raw)
    weights=[0.0]*len(raw) if gross==0 else [v/gross for v in raw]
    weights=[max(-cap,min(cap,w)) for w in weights]
    if normalize and sum(abs(v) for v in weights)>0:
        gross=sum(abs(v) for v in weights); weights=[v/gross for v in weights]
    normalized=[cross_sectional_zscore(c) for c in cols]
    score=[sum(weights[j]*normalized[j][i] for j in range(len(cols))) for i in range(n)]
    return {"scores":tuple(score),"weights":tuple(weights),"features":len(cols)}


def rank_features_by_mutual_information(features: Sequence[float], target: Sequence[float],
                                        bins: int = 10) -> float:
    """Discrete plug-in mutual information in nats with equal-frequency bins."""
    from ..core import quantile
    x=finite_series(features); y=finite_series(target)
    if len(x)!=len(y) or bins<2 or bins>len(x): raise ValueError("invalid data or bins")
    cuts_x=[quantile(x,i/bins) for i in range(1,bins)]
    cuts_y=[quantile(y,i/bins) for i in range(1,bins)]
    def bucket(v,cuts):
        for i,c in enumerate(cuts):
            if v<=c: return i
        return len(cuts)
    joint={}; px={}; py={}; n=len(x)
    for a,b in zip(x,y):
        i,j=bucket(a,cuts_x),bucket(b,cuts_y)
        joint[i,j]=joint.get((i,j),0)+1; px[i]=px.get(i,0)+1; py[j]=py.get(j,0)+1
    import math
    return sum((count/n)*math.log((count*n)/(px[i]*py[j])) for (i,j),count in joint.items())


def pairwise_feature_correlation(features: Sequence[Sequence[float]]) -> list[list[float]]:
    from ..core import correlation
    cols=[finite_series(c) for c in features]
    if not cols or any(len(c)!=len(cols[0]) for c in cols): raise ValueError("features must be rectangular")
    return [[correlation(a,b) for b in cols] for a in cols]


def decorrelate_features(features: Sequence[Sequence[float]], ridge: float = 1e-6) -> dict:
    """Whiten standardized feature columns through covariance eigendecomposition."""
    from ..risk.covariance import sample_covariance
    from ..linalg import eigen_symmetric
    from ..core import stddev,mean
    cols=[finite_series(c) for c in features]
    if not cols or any(len(c)!=len(cols[0]) for c in cols): raise ValueError("features must have equal lengths")
    z=[[0.0 if stddev(c)==0 else (v-mean(c))/stddev(c) for v in c] for c in cols]
    cov=sample_covariance(list(zip(*z)))
    vals,vecs=eigen_symmetric(cov)
    invroot=[[sum(vecs[i][k]*vecs[j][k]/sqrt(max(vals[k],ridge)) for k in range(len(vals))) for j in range(len(vals))] for i in range(len(vals))]
    whitened=[tuple(sum(invroot[j][k]*z[k][t] for k in range(len(z))) for t in range(len(z[0]))) for j in range(len(z))]
    return {"whitened_features":tuple(whitened),"eigenvalues":tuple(vals),"ridge":ridge}
