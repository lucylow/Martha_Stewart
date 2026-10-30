"""Leakage-aware cross-validation splitters for financial labels."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Iterator, Sequence

@dataclass(frozen=True)
class Fold:
    train: tuple[int,...]
    test: tuple[int,...]
    purged: tuple[int,...]
    embargoed: tuple[int,...]


def _intervals(n: int, label_end: Sequence[int] | None) -> list[tuple[int,int]]:
    if label_end is None: return [(i,i) for i in range(n)]
    if len(label_end)!=n: raise ValueError("label_end length must equal number of observations")
    result=[]
    for i,end in enumerate(label_end):
        e=int(end)
        if e<i or e>=n: raise ValueError(f"label_end[{i}] must be in [{i},{n-1}]")
        result.append((i,e))
    return result


def purged_kfold(n_samples: int, n_splits: int = 5, *,
                 label_end: Sequence[int] | None = None, embargo: int = 0) -> list[Fold]:
    """K-fold CV that removes training labels overlapping each test interval.

    ``label_end[i]`` is the final observation used to form sample ``i``'s label.
    Embargo removes observations immediately following each test block.
    """
    if n_samples<3 or n_splits<2 or n_splits>n_samples or embargo<0:
        raise ValueError("invalid sample count, split count, or embargo")
    intervals=_intervals(n_samples,label_end)
    blocks=[list(range(i*n_samples//n_splits,(i+1)*n_samples//n_splits)) for i in range(n_splits)]
    folds=[]
    for test in blocks:
        test_lo=min(test); test_hi=max(intervals[i][1] for i in test)
        embargoed=set(range(test[-1]+1,min(n_samples,test[-1]+1+embargo)))
        purged=set(); train=[]
        for i,(start,end) in enumerate(intervals):
            if i in test or i in embargoed: continue
            if any(not (end < intervals[j][0] or start > intervals[j][1]) for j in test):
                purged.add(i)
            else:
                train.append(i)
        folds.append(Fold(tuple(train),tuple(test),tuple(sorted(purged)),tuple(sorted(embargoed))))
    return folds


def walk_forward(n_samples: int, *, initial_train: int, test_size: int = 1,
                 step: int | None = None, expanding: bool = True,
                 label_end: Sequence[int] | None = None, embargo: int = 0) -> list[Fold]:
    """Chronological train/test splits with purge and post-test embargo."""
    if initial_train<2 or test_size<1 or initial_train+test_size>n_samples or embargo<0:
        raise ValueError("invalid walk-forward controls")
    stride=test_size if step is None else step
    if stride<1: raise ValueError("step must be positive")
    intervals=_intervals(n_samples,label_end); result=[]; start=initial_train
    while start+test_size<=n_samples:
        test=list(range(start,start+test_size)); train_candidates=list(range(0,start)) if expanding else list(range(max(0,start-initial_train),start))
        purged=[]; train=[]
        for i in train_candidates:
            if any(not (intervals[i][1]<intervals[j][0] or intervals[i][0]>intervals[j][1]) for j in test): purged.append(i)
            else: train.append(i)
        emb=list(range(start+test_size,min(n_samples,start+test_size+embargo)))
        result.append(Fold(tuple(train),tuple(test),tuple(purged),tuple(emb)))
        start+=stride
    return result


def combinatorial_purged_cv(n_samples: int, n_groups: int = 6, n_test_groups: int = 2,
                            *, label_end: Sequence[int] | None = None,
                            embargo: int = 0, max_paths: int = 1000) -> list[Fold]:
    """Combinatorial purged CV folds; test blocks are combinations of groups."""
    if n_groups<2 or n_test_groups<1 or n_test_groups>=n_groups or n_groups>n_samples:
        raise ValueError("invalid group configuration")
    blocks=[list(range(i*n_samples//n_groups,(i+1)*n_samples//n_groups)) for i in range(n_groups)]
    combos=list(combinations(range(n_groups),n_test_groups))
    if len(combos)>max_paths: raise ValueError("too many combinations; increase max_paths deliberately")
    intervals=_intervals(n_samples,label_end); folds=[]
    for combo in combos:
        test=sorted(i for g in combo for i in blocks[g]); tset=set(test)
        t_intervals=[intervals[i] for i in test]
        purged=set(); train=[]; embargoed=set()
        for i in test:
            embargoed.update(range(i+1,min(n_samples,i+1+embargo)))
        for i,(s,e) in enumerate(intervals):
            if i in tset or i in embargoed: continue
            if any(not (e<a or s>b) for a,b in t_intervals): purged.add(i)
            else: train.append(i)
        folds.append(Fold(tuple(train),tuple(test),tuple(sorted(purged)),tuple(sorted(embargoed))))
    return folds


def assert_no_overlap(fold: Fold, label_end: Sequence[int] | None = None) -> None:
    """Raise when any training label interval intersects any test label interval."""
    n=max(fold.train+fold.test+fold.purged+fold.embargoed,default=-1)+1
    intervals=_intervals(n,label_end)
    for i in fold.train:
        for j in fold.test:
            a,b=intervals[i]; c,d=intervals[j]
            if not (b<c or a>d): raise AssertionError(f"training sample {i} overlaps test sample {j}")
