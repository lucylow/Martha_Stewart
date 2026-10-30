"""Black-Litterman posterior expected returns and equilibrium priors."""
from __future__ import annotations
from typing import Sequence
from ..linalg import transpose,matmul,matvec,add,inverse,scale


def equilibrium_returns(covariance: Sequence[Sequence[float]], market_weights: Sequence[float],
                         risk_aversion: float) -> tuple[float,...]:
    w=list(map(float,market_weights))
    if risk_aversion<=0 or len(covariance)!=len(w): raise ValueError("invalid risk aversion or dimensions")
    return tuple(risk_aversion*v for v in matvec(covariance,w))


def posterior(covariance: Sequence[Sequence[float]], prior_returns: Sequence[float],
              view_matrix: Sequence[Sequence[float]], view_returns: Sequence[float],
              *, tau: float = 0.05, omega: Sequence[Sequence[float]] | None = None,
              omega_scale: float = 1.0) -> dict:
    """Compute BL posterior mean/covariance using linear-Gaussian updating.

    View rows encode P and Q. If Omega is omitted, use diagonal uncertainty
    proportional to the variance implied by each view, controlled by omega_scale.
    """
    sigma=[list(map(float,r)) for r in covariance]; pi=list(map(float,prior_returns))
    p=[list(map(float,r)) for r in view_matrix]; q=list(map(float,view_returns))
    n=len(pi); k=len(q)
    if len(sigma)!=n or any(len(r)!=n for r in sigma): raise ValueError("covariance dimension mismatch")
    if len(p)!=k or any(len(r)!=n for r in p): raise ValueError("view matrix dimension mismatch")
    if tau<=0 or omega_scale<=0: raise ValueError("tau and omega_scale must be positive")
    if omega is None:
        implied=matmul(matmul(p,scale(sigma,tau)),transpose(p))
        om=[[omega_scale*implied[i][i] if i==j else 0.0 for j in range(k)] for i in range(k)]
        for i in range(k): om[i][i]=max(om[i][i],1e-12)
    else:
        om=[list(map(float,r)) for r in omega]
        if len(om)!=k or any(len(r)!=k for r in om): raise ValueError("Omega dimension mismatch")
    tau_sigma=scale(sigma,tau)
    middle=add(matmul(matmul(p,tau_sigma),transpose(p)),om)
    middle_inv=inverse(middle,ridge=1e-12)
    innovation=[a-b for a,b in zip(q,matvec(p,pi))]
    gain=matmul(matmul(tau_sigma,transpose(p)),middle_inv)
    mu=[a+b for a,b in zip(pi,matvec(gain,innovation))]
    post_cov=add(sigma,scale(matmul(matmul(tau_sigma,transpose(p)),matmul(middle_inv,matmul(p,tau_sigma))),-1.0))
    return {"posterior_returns":tuple(mu),"posterior_covariance":tuple(tuple(r) for r in post_cov),
            "prior_returns":tuple(pi),"views":tuple(q),"tau":tau,"omega":tuple(tuple(r) for r in om)}


def black_litterman_weights(posterior_returns: Sequence[float], covariance: Sequence[Sequence[float]],
                            risk_aversion: float, *, normalize: bool = True) -> tuple[float,...]:
    if risk_aversion<=0: raise ValueError("risk_aversion must be positive")
    w=[v/risk_aversion for v in matvec(inverse(covariance,ridge=1e-10),posterior_returns)]
    if normalize:
        total=sum(w)
        if abs(total)<1e-14: raise ValueError("weights sum to zero")
        w=[v/total for v in w]
    return tuple(w)
