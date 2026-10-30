"""Conjugate Bayesian estimators for returns and linear factor models."""
from __future__ import annotations
from math import lgamma, log, sqrt
from typing import Sequence
from ..core import finite_series, mean, variance
from ..linalg import transpose,matmul,matvec,solve,inverse,dot


def normal_normal_posterior(observations: Sequence[float], prior_mean: float,
                            prior_variance: float, observation_variance: float) -> dict:
    """Posterior for unknown mean with known observation variance."""
    x=finite_series(observations,name="observations")
    if prior_variance<=0 or observation_variance<=0: raise ValueError("variances must be positive")
    precision=1/prior_variance+len(x)/observation_variance
    post_var=1/precision
    post_mean=post_var*(prior_mean/prior_variance+sum(x)/observation_variance)
    return {"mean":post_mean,"variance":post_var,"standard_deviation":sqrt(post_var),
            "prior_mean":prior_mean,"observations":len(x)}


def normal_inverse_gamma_posterior(observations: Sequence[float], prior_mean: float = 0.0,
                                   prior_kappa: float = 0.01, prior_alpha: float = 2.0,
                                   prior_beta: float = 1e-4) -> dict:
    """Normal-inverse-gamma posterior for unknown Gaussian mean and variance."""
    x=finite_series(observations,name="observations")
    if prior_kappa<=0 or prior_alpha<=0 or prior_beta<=0: raise ValueError("prior parameters must be positive")
    n=len(x); sample_mean=mean(x); ss=sum((v-sample_mean)**2 for v in x)
    kappa=prior_kappa+n; alpha=prior_alpha+n/2
    beta=prior_beta+0.5*ss+(prior_kappa*n*(sample_mean-prior_mean)**2)/(2*kappa)
    mu=(prior_kappa*prior_mean+n*sample_mean)/kappa
    return {"mean_location":mu,"mean_precision":kappa,"variance_shape":alpha,"variance_scale":beta,
            "posterior_mean_variance":beta/(alpha-1) if alpha>1 else float("inf"),
            "observations":n}


def bayesian_linear_regression(features: Sequence[Sequence[float]], target: Sequence[float],
                               prior_mean: Sequence[float] | None = None,
                               prior_covariance: Sequence[Sequence[float]] | None = None,
                               noise_variance: float = 1.0, *, add_intercept: bool = True) -> dict:
    """Gaussian prior / Gaussian likelihood regression with known noise variance."""
    from ..regression import design_matrix
    y=finite_series(target,name="target",min_length=2); x=design_matrix(features,add_intercept=add_intercept)
    if len(x)!=len(y) or noise_variance<=0: raise ValueError("data dimensions or noise variance invalid")
    p=len(x[0]); m=[0.0]*p if prior_mean is None else list(map(float,prior_mean))
    if len(m)!=p: raise ValueError("prior mean dimension mismatch")
    cov=[[100.0 if i==j else 0.0 for j in range(p)] for i in range(p)] if prior_covariance is None else [list(map(float,r)) for r in prior_covariance]
    if len(cov)!=p or any(len(r)!=p for r in cov): raise ValueError("prior covariance dimension mismatch")
    precision=inverse(cov,ridge=1e-12)
    xt=transpose(x); likelihood_precision=matmul(xt,x)
    posterior_precision=[[precision[i][j]+likelihood_precision[i][j]/noise_variance for j in range(p)] for i in range(p)]
    rhs0=matvec(precision,m); rhs1=matvec(xt,y)
    rhs=[rhs0[i]+rhs1[i]/noise_variance for i in range(p)]
    post_cov=inverse(posterior_precision,ridge=1e-12); post_mean=matvec(post_cov,rhs)
    fitted=matvec(x,post_mean); residual=[a-b for a,b in zip(y,fitted)]
    return {"posterior_mean":tuple(post_mean),"posterior_covariance":tuple(tuple(r) for r in post_cov),
            "fitted":tuple(fitted),"residuals":tuple(residual),"noise_variance":noise_variance}


def bayesian_shrinkage_mean(returns: Sequence[float], prior_mean: float = 0.0,
                            prior_strength: float = 20.0) -> float:
    """Simple normal-prior shrinkage of a sample mean toward a target."""
    x=finite_series(returns)
    if prior_strength<0: raise ValueError("prior_strength must be non-negative")
    return (sum(x)+prior_strength*prior_mean)/(len(x)+prior_strength)
