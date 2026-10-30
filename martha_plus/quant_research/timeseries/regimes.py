"""Univariate Gaussian hidden-Markov regime model estimated by EM."""
from __future__ import annotations
from dataclasses import dataclass
from math import exp, log, pi, sqrt
from typing import Sequence
from ..core import finite_series, mean, variance

@dataclass(frozen=True)
class HMMResult:
    means: tuple[float,...]
    variances: tuple[float,...]
    transition: tuple[tuple[float,...],...]
    initial: tuple[float,...]
    filtered_probabilities: tuple[tuple[float,...],...]
    smoothed_probabilities: tuple[tuple[float,...],...]
    states: tuple[int,...]
    log_likelihood: float
    iterations: int
    converged: bool


def _gaussian_pdf(x: float, mu: float, var: float) -> float:
    var=max(var,1e-10)
    return exp(-0.5*(x-mu)**2/var)/sqrt(2*pi*var)


def _normalize(row: Sequence[float]) -> list[float]:
    s=sum(row)
    return [v/s for v in row] if s>0 else [1.0/len(row)]*len(row)


def _forward(values, means, variances, transition, initial):
    k=len(means); alpha=[]; scales=[]
    first=[initial[j]*_gaussian_pdf(values[0],means[j],variances[j]) for j in range(k)]
    c=max(sum(first),1e-300); alpha.append([v/c for v in first]); scales.append(c)
    for t in range(1,len(values)):
        row=[_gaussian_pdf(values[t],means[j],variances[j])*sum(alpha[-1][i]*transition[i][j] for i in range(k)) for j in range(k)]
        c=max(sum(row),1e-300); alpha.append([v/c for v in row]); scales.append(c)
    return alpha,scales,sum(log(c) for c in scales)


def fit_gaussian_hmm(values: Sequence[float], states: int = 2, *, max_iter: int = 200,
                     tolerance: float = 1e-6, variance_floor: float = 1e-8,
                     transition_prior: float = 1.1) -> HMMResult:
    """Fit a small univariate Gaussian HMM using scaled forward-backward EM.

    This is a local optimum; use multiple initializations for serious model
    selection.  State labels are sorted by mean after fitting for stable output.
    """
    x=finite_series(values,name="values",min_length=20)
    if states<2 or states>5: raise ValueError("states must be between 2 and 5")
    if max_iter<1 or tolerance<=0 or variance_floor<=0 or transition_prior<1:
        raise ValueError("invalid EM controls")
    ordered=sorted(x); means=[ordered[min(len(x)-1,int((j+0.5)*len(x)/states))] for j in range(states)]
    global_var=max(variance(x),variance_floor)
    variances=[global_var for _ in range(states)]
    transition=[[0.85 if i==j else 0.15/(states-1) for j in range(states)] for i in range(states)]
    initial=[1.0/states]*states; previous=float("-inf"); converged=False
    alpha=[]; gamma=[]; ll=previous
    for iteration in range(1,max_iter+1):
        alpha,scales,ll=_forward(x,means,variances,transition,initial)
        # Scaled backward recursion.
        beta=[[1.0]*states for _ in x]
        for t in range(len(x)-2,-1,-1):
            for i in range(states):
                beta[t][i]=sum(transition[i][j]*_gaussian_pdf(x[t+1],means[j],variances[j])*beta[t+1][j] for j in range(states))/scales[t+1]
        gamma=[_normalize([alpha[t][j]*beta[t][j] for j in range(states)]) for t in range(len(x))]
        weights=[sum(gamma[t][j] for t in range(len(gamma))) for j in range(states)]
        new_means=[sum(gamma[t][j]*x[t] for t in range(len(x)))/max(weights[j],1e-15) for j in range(states)]
        new_vars=[max(variance_floor,sum(gamma[t][j]*(x[t]-new_means[j])**2 for t in range(len(x)))/max(weights[j],1e-15)) for j in range(states)]
        counts=[[transition_prior-1.0 for _ in range(states)] for _ in range(states)]
        for t in range(len(x)-1):
            denom=sum(alpha[t][i]*transition[i][j]*_gaussian_pdf(x[t+1],means[j],variances[j])*beta[t+1][j]
                      for i in range(states) for j in range(states))
            for i in range(states):
                for j in range(states):
                    numer=alpha[t][i]*transition[i][j]*_gaussian_pdf(x[t+1],means[j],variances[j])*beta[t+1][j]
                    counts[i][j]+=numer/max(denom,1e-300)
        new_transition=[_normalize(row) for row in counts]
        initial=_normalize(gamma[0])
        means,variances,transition=new_means,new_vars,new_transition
        if previous != float("-inf") and abs(ll-previous)<=tolerance*(1.0+abs(previous)):
            converged=True; break
        previous=ll
    alpha,scales,ll=_forward(x,means,variances,transition,initial)
    beta=[[1.0]*states for _ in x]
    for t in range(len(x)-2,-1,-1):
        for i in range(states):
            beta[t][i]=sum(transition[i][j]*_gaussian_pdf(x[t+1],means[j],variances[j])*beta[t+1][j] for j in range(states))/scales[t+1]
    gamma=[_normalize([alpha[t][j]*beta[t][j] for j in range(states)]) for t in range(len(x))]
    order=sorted(range(states),key=lambda i:means[i])
    means=[means[i] for i in order]; variances=[variances[i] for i in order]
    transition=[[transition[i][j] for j in order] for i in order]
    initial=[initial[i] for i in order]
    alpha=[[row[i] for i in order] for row in alpha]
    gamma=[[row[i] for i in order] for row in gamma]
    return HMMResult(tuple(means),tuple(variances),tuple(tuple(r) for r in transition),tuple(initial),
                     tuple(tuple(r) for r in alpha),tuple(tuple(r) for r in gamma),
                     tuple(max(range(states),key=row.__getitem__) for row in gamma),ll,iteration,converged)


def transition_half_life(probability: float) -> float:
    """Expected geometric persistence half-life for a self-transition probability."""
    if not 0 < probability < 1: raise ValueError("probability must lie in (0,1)")
    return log(0.5)/log(probability)
