"""Permutation and ablation importance for time-series prediction models."""
from __future__ import annotations
import random
from typing import Callable,Sequence
from ..core import finite_series

ScoreFunction=Callable[[Sequence[Sequence[float]],Sequence[float]],float]


def permutation_importance(features: Sequence[Sequence[float]], target: Sequence[float],
                           scorer: ScoreFunction, *, repeats: int = 20,
                           block_size: int = 1, seed: int = 7) -> dict:
    """Measure score loss after permuting each feature column.

    Block permutation preserves local serial order within shuffled chunks; it
    does not preserve cross-feature dependence and is not causal attribution.
    ``scorer`` should return a higher-is-better score and must be deterministic.
    """
    cols=[finite_series(c,name="feature") for c in features]; y=finite_series(target)
    if not cols or any(len(c)!=len(y) for c in cols) or repeats<1 or block_size<1 or block_size>len(y):
        raise ValueError("invalid feature panel or permutation controls")
    rng=random.Random(seed); baseline=float(scorer(cols,y)); results=[]; n=len(y)
    for j,col in enumerate(cols):
        scores=[]
        for _ in range(repeats):
            starts=list(range(0,n,block_size)); rng.shuffle(starts)
            perm=[]
            for start in starts: perm.extend(col[start:min(n,start+block_size)])
            perm=perm[:n]; modified=[list(c) for c in cols]; modified[j]=perm
            scores.append(float(scorer(modified,y)))
        drops=[baseline-v for v in scores]
        results.append({"feature_index":j,"importance_mean":sum(drops)/len(drops),
                        "importance_std":(sum((v-sum(drops)/len(drops))**2 for v in drops)/max(1,len(drops)-1))**0.5,
                        "permutation_scores":tuple(scores)})
    return {"baseline_score":baseline,"features":tuple(sorted(results,key=lambda r:r["importance_mean"],reverse=True)),
            "repeats":repeats,"block_size":block_size,"warning":"correlated features can share or mask permutation importance"}


def ablation_importance(features: Sequence[Sequence[float]], target: Sequence[float],
                        scorer: ScoreFunction) -> dict:
    """Score each model after removing one complete feature column."""
    cols=[finite_series(c) for c in features]; y=finite_series(target)
    if len(cols)<2 or any(len(c)!=len(y) for c in cols): raise ValueError("need >=2 aligned features")
    baseline=float(scorer(cols,y)); rows=[]
    for j in range(len(cols)):
        reduced=cols[:j]+cols[j+1:]
        score=float(scorer(reduced,y))
        rows.append({"feature_index":j,"score_without_feature":score,"ablation_drop":baseline-score})
    return {"baseline_score":baseline,"features":tuple(rows)}


def time_series_permutation(values: Sequence[float], block_size: int = 5, seed: int = 7) -> tuple[float,...]:
    """Randomly reorder contiguous blocks, retaining within-block order."""
    x=finite_series(values)
    if block_size<1 or block_size>len(x): raise ValueError("invalid block size")
    rng=random.Random(seed); blocks=[x[i:i+block_size] for i in range(0,len(x),block_size)]; rng.shuffle(blocks)
    return tuple(v for block in blocks for v in block)
