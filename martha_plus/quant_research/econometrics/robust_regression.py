"""Robust and weighted regression estimators for noisy financial panels."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series,mean,quantile
from ..linalg import transpose,matmul,matvec,solve,inverse,dot
from ..regression import design_matrix


def weighted_least_squares(features: Sequence[Sequence[float]], target: Sequence[float],
                           weights: Sequence[float], *, add_intercept: bool = True,
                           ridge: float = 0.0) -> dict:
    """Weighted OLS using square-root row scaling."""
    y=finite_series(target,name="target"); w=finite_series(weights,name="weights")
    x=design_matrix(features,add_intercept=add_intercept)
    if len(x)!=len(y) or len(w)!=len(y) or any(v<0 for v in w) or sum(w)<=0: raise ValueError("invalid weighted regression inputs")
    weighted_x=[[row[j]*sqrt(wi) for j in range(len(row))] for row,wi in zip(x,w)]
    weighted_y=[yi*sqrt(wi) for yi,wi in zip(y,w)]
    xt=transpose(weighted_x); gram=matmul(xt,weighted_x)
    if ridge<0: raise ValueError("ridge must be nonnegative")
    for j in range(len(gram)): gram[j][j]+=ridge
    beta=solve(gram,matvec(xt,weighted_y))
    fitted=matvec(x,beta); residual=[a-b for a,b in zip(y,fitted)]
    weighted_mean=sum(a*b for a,b in zip(y,w))/sum(w)
    sse=sum(w[i]*residual[i]**2 for i in range(len(y)))
    tss=sum(w[i]*(y[i]-weighted_mean)**2 for i in range(len(y)))
    return {"coefficients":tuple(beta),"fitted":tuple(fitted),"residuals":tuple(residual),
            "weighted_r_squared":1-sse/tss if tss>0 else 0.0,"weighted_sse":sse}


def huber_regression(features: Sequence[Sequence[float]], target: Sequence[float],
                     *, tuning: float = 1.345, add_intercept: bool = True,
                     max_iter: int = 100, tolerance: float = 1e-8,
                     ridge: float = 1e-8) -> dict:
    """Huber M-regression via iteratively reweighted least squares."""
    from ..regression import ols
    from ..core import stddev
    y=finite_series(target,name="target",min_length=5)
    x=design_matrix(features,add_intercept=add_intercept)
    if len(x)!=len(y) or tuning<=0 or max_iter<1 or tolerance<=0: raise ValueError("invalid Huber regression controls")
    initial=ols(features,y,add_intercept=add_intercept,ridge=ridge)
    beta=list(initial.coefficients); converged=False; scale_value=max(stddev(initial.residuals),1e-12)
    for iteration in range(1,max_iter+1):
        residual=[y[i]-dot(x[i],beta) for i in range(len(y))]
        center=quantile(residual,0.5)
        scale_value=max(1.4826*quantile([abs(v-center) for v in residual],0.5),1e-12)
        weights=[]
        for value in residual:
            u=abs((value-center)/scale_value)
            weights.append(1.0 if u<=tuning else tuning/u)
        fit=weighted_least_squares(features,y,weights,add_intercept=add_intercept,ridge=ridge)
        candidate=list(fit["coefficients"])
        change=max(abs(a-b) for a,b in zip(candidate,beta))
        beta=candidate
        if change<tolerance: converged=True; break
    fitted=[dot(row,beta) for row in x]; residual=[a-b for a,b in zip(y,fitted)]
    return {"coefficients":tuple(beta),"fitted":tuple(fitted),"residuals":tuple(residual),
            "robust_scale":scale_value,"iterations":iteration,"converged":converged,"tuning":tuning,
            "estimator":"Huber IRLS; scale estimated by residual MAD"}


def quantile_regression(features: Sequence[Sequence[float]], target: Sequence[float],
                        quantile_level: float = 0.5, *, add_intercept: bool = True,
                        max_iter: int = 300, tolerance: float = 1e-7,
                        ridge: float = 1e-6) -> dict:
    """Smoothed check-loss quantile regression via iteratively reweighted LS.

    This is an approximate solver. The small smoothing floor prevents singular
    weights when residuals approach zero; use a specialized LP/convex solver
    for exact quantile-regression inference.
    """
    from ..regression import ols
    y=finite_series(target,name="target",min_length=5)
    x=design_matrix(features,add_intercept=add_intercept)
    if len(x)!=len(y) or not 0<quantile_level<1 or max_iter<1 or tolerance<=0: raise ValueError("invalid quantile regression controls")
    beta=list(ols(features,y,add_intercept=add_intercept,ridge=ridge).coefficients)
    converged=False; epsilon=1e-4
    for iteration in range(1,max_iter+1):
        residual=[y[i]-dot(x[i],beta) for i in range(len(y))]
        weights=[]
        for value in residual:
            u=value
            # IRLS majorization of asymmetric absolute/check loss.
            weights.append((quantile_level if u>=0 else 1-quantile_level)/max(abs(u),epsilon))
        fit=weighted_least_squares(features,y,weights,add_intercept=add_intercept,ridge=ridge)
        candidate=list(fit["coefficients"])
        change=max(abs(a-b) for a,b in zip(candidate,beta)); beta=candidate
        if change<tolerance: converged=True; break
    fitted=[dot(row,beta) for row in x]; residual=[a-b for a,b in zip(y,fitted)]
    check_loss=sum(quantile_level*e if e>=0 else (quantile_level-1)*e for e in residual)
    return {"coefficients":tuple(beta),"fitted":tuple(fitted),"residuals":tuple(residual),
            "quantile":quantile_level,"check_loss":check_loss,"iterations":iteration,"converged":converged,
            "estimator":"smoothed IRLS approximation"}


def rolling_coefficient_stability(features: Sequence[Sequence[float]], target: Sequence[float],
                                  window: int, *, add_intercept: bool = True) -> dict:
    from ..regression import rolling_ols
    path=rolling_ols(features,target,window,add_intercept=add_intercept)
    rows=[list(f.coefficients) for f in path if f is not None]
    if not rows: raise ValueError("no rolling fits available")
    columns=transpose(rows); summary=[]
    for i,col in enumerate(columns):
        m=mean(col); sd=sqrt(sum((v-m)**2 for v in col)/max(1,len(col)-1))
        summary.append({"coefficient_index":i,"mean":m,"std":sd,"min":min(col),"max":max(col)})
    return {"coefficient_summary":tuple(summary),"coefficient_path":tuple(tuple(r) for r in rows),"windows":len(rows)}
