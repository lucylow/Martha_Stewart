"""Bayesian model averaging and dynamic signal model weights."""
from __future__ import annotations
from math import exp,log
from typing import Sequence


def _softmax(log_values: Sequence[float]) -> list[float]:
    peak=max(log_values); values=[exp(v-peak) for v in log_values]; total=sum(values)
    return [v/total for v in values]


def bayesian_model_average(predictions: Sequence[Sequence[float]],
                           observation: Sequence[float],
                           predictive_variances: Sequence[Sequence[float]],
                           prior_probabilities: Sequence[float] | None = None) -> dict:
    """Posterior model probabilities for independent Gaussian predictive errors."""
    y=list(map(float,observation)); models=[list(map(float,p)) for p in predictions]
    variances=[list(map(float,v)) for v in predictive_variances]
    m=len(models)
    if m<1 or any(len(p)!=len(y) for p in models) or len(variances)!=m or any(len(v)!=len(y) for v in variances):
        raise ValueError("prediction matrix dimensions mismatch")
    priors=[1/m]*m if prior_probabilities is None else list(map(float,prior_probabilities))
    if len(priors)!=m or any(v<=0 for v in priors): raise ValueError("priors must be positive and match models")
    total=sum(priors); priors=[v/total for v in priors]
    log_post=[]; log_likelihoods=[]
    for model,var in zip(models,variances):
        if any(v<=0 for v in var): raise ValueError("predictive variances must be positive")
        ll=-0.5*sum(log(2*__import__('math').pi*v)+(obs-p)**2/v for obs,p,v in zip(y,model,var))
        log_likelihoods.append(ll); log_post.append(log(priors[len(log_post)])+ll)
    weights=_softmax(log_post)
    combined=[sum(weights[m]*models[m][t] for m in range(len(models))) for t in range(len(y))]
    return {"posterior_model_probabilities":tuple(weights),"log_likelihoods":tuple(log_likelihoods),
            "combined_prediction":tuple(combined),"prior_probabilities":tuple(priors)}


def log_score_weights(log_score_paths: Sequence[Sequence[float]], prior: Sequence[float] | None = None,
                      temperature: float = 1.0) -> dict:
    """Sequentially update model weights from per-period log predictive scores."""
    scores=[list(map(float,row)) for row in log_score_paths]
    if not scores or any(len(r)!=len(scores[0]) for r in scores) or temperature<=0: raise ValueError("invalid score panel")
    m=len(scores[0]); weights=[1/m]*m if prior is None else list(map(float,prior))
    if len(weights)!=m or any(v<=0 for v in weights): raise ValueError("invalid prior weights")
    total=sum(weights); weights=[v/total for v in weights]
    path=[]
    for row in scores:
        logw=[log(weights[j])+row[j]/temperature for j in range(m)]
        peak=max(logw); raw=[exp(v-peak) for v in logw]; total=sum(raw)
        weights=[v/total for v in raw]; path.append(tuple(weights))
    return {"final_weights":tuple(weights),"weight_path":tuple(path),"temperature":temperature}


def posterior_predictive_average(predictions: Sequence[Sequence[float]], weights: Sequence[float],
                                 variances: Sequence[Sequence[float]]) -> dict:
    p=[list(map(float,row)) for row in predictions]; w=list(map(float,weights)); v=[list(map(float,row)) for row in variances]
    if not p or len(p)!=len(w) or len(v)!=len(p) or any(len(r)!=len(p[0]) for r in p+v): raise ValueError("matrix dimensions mismatch")
    if any(x<0 for x in w) or sum(w)<=0: raise ValueError("weights must be nonnegative with positive sum")
    w=[x/sum(w) for x in w]; n=len(p[0]); mean_pred=[]; total_var=[]
    for t in range(n):
        mu=sum(w[j]*p[j][t] for j in range(len(p)))
        var=sum(w[j]*(v[j][t]+(p[j][t]-mu)**2) for j in range(len(p)))
        mean_pred.append(mu); total_var.append(var)
    return {"mean":tuple(mean_pred),"variance":tuple(total_var),"weights":tuple(w)}
