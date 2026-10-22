"""Walk-forward splitting primitives; caller supplies an explicit scoring function."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Generic, Sequence, TypeVar
T = TypeVar("T")
P = TypeVar("P")

@dataclass(frozen=True, slots=True)
class WalkForwardFold(Generic[T]):
    fold: int
    train: tuple[T, ...]
    test: tuple[T, ...]

@dataclass(frozen=True, slots=True)
class FoldScore(Generic[P]):
    fold: int
    parameters: P
    train_score: float
    test_score: float
    train_count: int
    test_count: int


def walk_forward_folds(items: Sequence[T], train_size: int, test_size: int,
                       step_size: int | None = None, expanding: bool = True) -> tuple[WalkForwardFold[T], ...]:
    if train_size < 2 or test_size < 1:
        raise ValueError("train_size >= 2 and test_size >= 1 are required")
    step = step_size or test_size
    if step < 1:
        raise ValueError("step_size must be positive")
    folds: list[WalkForwardFold[T]] = []
    test_start = train_size
    fold_number = 1
    while test_start < len(items):
        test_end = min(len(items), test_start + test_size)
        train_start = 0 if expanding else max(0, test_start - train_size)
        train = tuple(items[train_start:test_start])
        test = tuple(items[test_start:test_end])
        if len(train) >= 2 and test:
            folds.append(WalkForwardFold(fold_number, train, test))
            fold_number += 1
        test_start += step
    return tuple(folds)


def evaluate_walk_forward(items: Sequence[T], parameter_grid: Sequence[P],
                          score: Callable[[Sequence[T], P], float], train_size: int,
                          test_size: int, step_size: int | None = None,
                          expanding: bool = True) -> tuple[FoldScore[P], ...]:
    """Select parameters on each training fold and score once on its later test fold.

    The function cannot prevent leakage inside a user-supplied score callback. Keep
    transformations, scalers, and model fitting confined to the train subset.
    """
    if not parameter_grid:
        raise ValueError("parameter_grid cannot be empty")
    folds = walk_forward_folds(items, train_size, test_size, step_size, expanding)
    result: list[FoldScore[P]] = []
    for fold in folds:
        train_scores = [(score(fold.train, params), params) for params in parameter_grid]
        # Stable deterministic tie-break: earliest parameter in the supplied grid wins.
        best_train_score, best_params = max(train_scores, key=lambda item: item[0])
        test_score = score(fold.test, best_params)
        result.append(FoldScore(fold.fold, best_params, float(best_train_score),
                                float(test_score), len(fold.train), len(fold.test)))
    return tuple(result)
