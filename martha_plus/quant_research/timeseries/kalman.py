"""Linear Kalman filtering and adaptive hedge-ratio estimation."""
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from typing import Sequence
from ..linalg import eye, matmul, matvec, transpose, add, subtract, inverse, symmetrize, zeros
from ..core import finite_series

@dataclass(frozen=True)
class FilterStep:
    state: tuple[float,...]
    covariance: tuple[tuple[float,...],...]
    innovation: tuple[float,...]
    innovation_covariance: tuple[tuple[float,...],...]
    log_likelihood: float

@dataclass(frozen=True)
class KalmanResult:
    steps: tuple[FilterStep,...]
    log_likelihood: float


class KalmanFilter:
    """Discrete linear Gaussian state-space filter.

    Model: x[t]=F x[t-1]+w, y[t]=H x[t]+v, with independent Gaussian noise.
    Missing observations are represented by None and cause prediction only.
    """
    def __init__(self, transition, observation, process_cov, observation_cov,
                 initial_state, initial_cov):
        self.f=[list(map(float,r)) for r in transition]
        self.h=[list(map(float,r)) for r in observation]
        self.q=[list(map(float,r)) for r in process_cov]
        self.r=[list(map(float,r)) for r in observation_cov]
        self.x=list(map(float,initial_state))
        self.p=[list(map(float,r)) for r in initial_cov]
        n=len(self.x); m=len(self.h)
        if len(self.f)!=n or any(len(r)!=n for r in self.f): raise ValueError("transition matrix shape mismatch")
        if len(self.h)!=m or any(len(r)!=n for r in self.h): raise ValueError("observation matrix shape mismatch")
        if len(self.q)!=n or any(len(r)!=n for r in self.q): raise ValueError("process covariance shape mismatch")
        if len(self.p)!=n or any(len(r)!=n for r in self.p): raise ValueError("initial covariance shape mismatch")
        if len(self.r)!=m or any(len(r)!=m for r in self.r): raise ValueError("observation covariance shape mismatch")

    def step(self, observation: Sequence[float] | None) -> FilterStep:
        xp=matvec(self.f,self.x)
        pp=add(matmul(matmul(self.f,self.p),transpose(self.f)),self.q)
        if observation is None:
            self.x,self.p=xp, symmetrize(pp)
            return FilterStep(tuple(self.x),tuple(map(tuple,self.p)),(),(),0.0)
        y=list(map(float,observation))
        if len(y)!=len(self.h): raise ValueError("observation vector shape mismatch")
        predicted=matvec(self.h,xp)
        innov=[a-b for a,b in zip(y,predicted)]
        s=add(matmul(matmul(self.h,pp),transpose(self.h)),self.r)
        sinv=inverse(s,ridge=1e-14)
        k=matmul(matmul(pp,transpose(self.h)),sinv)
        adjustment=matvec(k,innov)
        self.x=[a+b for a,b in zip(xp,adjustment)]
        # Joseph form is more numerically stable than (I-KH)P.
        ident=eye(len(self.x)); kh=matmul(k,self.h)
        ikh=subtract(ident,kh)
        self.p=symmetrize(add(matmul(matmul(ikh,pp),transpose(ikh)),matmul(matmul(k,self.r),transpose(k))))
        det=1.0
        try:
            # Cholesky determinant avoids a general eigenvalue routine.
            from ..linalg import cholesky
            c=cholesky(s,jitter=1e-14)
            det=1.0
            for i in range(len(c)): det*=c[i][i]**2
        except ValueError:
            det=max(1e-30,abs(s[0][0]))
        quad=sum(a*b for a,b in zip(innov,matvec(sinv,innov)))
        ll=-0.5*(len(y)*1.8378770664093453+__import__('math').log(max(det,1e-30))+quad)
        return FilterStep(tuple(self.x),tuple(map(tuple,self.p)),tuple(innov),tuple(map(tuple,s)),ll)

    def filter(self, observations: Sequence[Sequence[float] | None]) -> KalmanResult:
        steps=tuple(self.step(y) for y in observations)
        return KalmanResult(steps,sum(s.log_likelihood for s in steps))


def dynamic_regression(y: Sequence[float], x: Sequence[float], *,
                       observation_variance: float = 1e-4,
                       coefficient_process_variance: float = 1e-6,
                       initial_beta: float = 0.0,
                       initial_intercept: float = 0.0) -> dict:
    """Estimate time-varying intercept and slope in y[t]=a[t]+b[t]x[t]+noise."""
    a,b=finite_series(y),finite_series(x)
    if len(a)!=len(b): raise ValueError("series lengths must match")
    if observation_variance<=0 or coefficient_process_variance<0: raise ValueError("invalid variances")
    n=len(a); f=eye(2); q=[[coefficient_process_variance,0.0],[0.0,coefficient_process_variance]]
    h=[[1.0,v] for v in b]
    # Each observation row varies with t, so filter directly with a fresh H.
    filt=KalmanFilter(f,[[1.0,0.0]],q,[[observation_variance]],
                      [initial_intercept,initial_beta],[[1e3,0.0],[0.0,1e3]])
    steps=[]
    for yi,hi in zip(a,h):
        filt.h=[hi]
        steps.append(filt.step([yi]))
    return {"intercept":tuple(s.state[0] for s in steps),
            "beta":tuple(s.state[1] for s in steps),
            "state_covariance":tuple(s.covariance for s in steps),
            "log_likelihood":sum(s.log_likelihood for s in steps)}
