"""Leakage-aware chronological folds for strategy and feature research."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TimeFold:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    embargo_end: int

    @property
    def train_indices(self) -> range:
        return range(self.train_start, self.train_end)

    @property
    def test_indices(self) -> range:
        return range(self.test_start, self.test_end)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.train_start < 0 or self.test_start < 0:
            issues.append("indices must be non-negative")
        if self.train_end <= self.train_start or self.test_end <= self.test_start:
            issues.append("train and test slices must be non-empty")
        if self.train_end > self.test_start:
            issues.append("training data overlaps test data")
        if self.embargo_end < self.test_end:
            issues.append("embargo_end must be >= test_end")
        return tuple(issues)


def expanding_window_folds(n_samples: int, *, min_train_size: int, test_size: int,
                           step_size: int | None = None, gap: int = 0,
                           embargo: int = 0, max_train_size: int | None = None) -> list[TimeFold]:
    """Build expanding-window folds. Test intervals never overlap when step >= test_size.

    `gap` excludes observations immediately before the test range from training.
    `embargo` describes post-test observations excluded from the next training set
    when a caller opts into fixed or custom fold assembly; it is included in the
    fold metadata and should also be enforced by the consuming evaluator.
    """
    step = test_size if step_size is None else step_size
    if n_samples < 0 or min_train_size < 1 or test_size < 1 or step < 1 or gap < 0 or embargo < 0:
        raise ValueError("invalid sample counts or fold dimensions")
    if max_train_size is not None and max_train_size < min_train_size:
        raise ValueError("max_train_size must be >= min_train_size")
    folds: list[TimeFold] = []
    first_test = min_train_size + gap
    fold_index = 0
    start_test = first_test
    while start_test + test_size <= n_samples:
        train_end = start_test - gap
        train_start = 0 if max_train_size is None else max(0, train_end - max_train_size)
        if train_end - train_start < min_train_size:
            start_test += step
            continue
        end_test = start_test + test_size
        folds.append(TimeFold(fold_index, train_start, train_end, start_test, end_test,
                              min(n_samples, end_test + embargo)))
        fold_index += 1
        start_test += step
    return folds


def purged_kfold_indices(n_samples: int, n_splits: int, *, purge: int = 0,
                         embargo: int = 0) -> list[tuple[list[int], list[int]]]:
    """Generate non-shuffled K-fold partitions with purging around each test block.

    Unlike expanding-window CV, this includes both past and future training data
    outside the purged interval. For strict forecasting, prefer expanding_window_folds.
    """
    if n_samples < n_splits or n_splits < 2 or purge < 0 or embargo < 0:
        raise ValueError("invalid purged K-fold parameters")
    base, remainder = divmod(n_samples, n_splits)
    folds: list[tuple[list[int], list[int]]] = []
    start = 0
    all_indices = set(range(n_samples))
    for i in range(n_splits):
        size = base + (1 if i < remainder else 0)
        stop = start + size
        test = list(range(start, stop))
        left = max(0, start - purge)
        right = min(n_samples, stop + embargo)
        train = sorted(all_indices - set(range(left, right)))
        if not train:
            raise ValueError("purge/embargo removed all training observations")
        folds.append((train, test))
        start = stop
    return folds


def fold_rows(rows: Sequence[T], fold: TimeFold) -> tuple[list[T], list[T]]:
    issues = fold.validate()
    if issues:
        raise ValueError("invalid fold: " + "; ".join(issues))
    if fold.train_end > len(rows) or fold.test_end > len(rows):
        raise ValueError("fold indices exceed data length")
    return list(rows[fold.train_start:fold.train_end]), list(rows[fold.test_start:fold.test_end])


def purged_train_test_split(n_samples: int, test_start: int, test_end: int, *,
                            purge: int = 0, embargo: int = 0) -> tuple[list[int], list[int]]:
    if n_samples < 1 or not 0 <= test_start < test_end <= n_samples:
        raise ValueError("invalid test interval")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be non-negative")
    excluded_start = max(0, test_start - purge)
    excluded_end = min(n_samples, test_end + embargo)
    test = list(range(test_start, test_end))
    train = list(range(0, excluded_start)) + list(range(excluded_end, n_samples))
    return train, test


def label_overlap_purge(label_end_indices: Sequence[int], test_start: int,
                        test_end: int) -> list[int]:
    """Return valid train indices whose labels do not extend into the test interval."""
    if not 0 <= test_start < test_end:
        raise ValueError("invalid test interval")
    valid: list[int] = []
    for i, end in enumerate(label_end_indices):
        if end < i:
            raise ValueError("label end index may not precede its sample index")
        if i < test_start and end < test_start:
            valid.append(i)
        elif i >= test_end and end >= test_end:
            valid.append(i)
    return valid


def validate_fold_separation(folds: Iterable[TimeFold]) -> dict[str, object]:
    items = list(folds)
    issues: list[str] = []
    previous_test_start = -1
    previous_test_end = -1
    for i, fold in enumerate(items):
        issues.extend(f"fold {i}: {problem}" for problem in fold.validate())
        if fold.test_start < previous_test_start:
            issues.append(f"fold {i}: test windows are not chronological")
        if fold.test_start < previous_test_end:
            issues.append(f"fold {i}: test window overlaps previous fold")
        previous_test_start, previous_test_end = fold.test_start, fold.test_end
    return {"valid": not issues, "fold_count": len(items), "issues": issues}
