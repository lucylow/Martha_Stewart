"""Signal calibration metrics, threshold search, and probability reliability bins."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, sqrt
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    observations: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    brier_score: float
    log_loss: float
    confusion: dict[str, int]


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_probability: float
    observed_rate: float


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    objective: float


def _validate(y_true: Sequence[int | bool], probabilities: Sequence[float]) -> None:
    if len(y_true) != len(probabilities) or not y_true:
        raise ValueError("labels and probabilities must have equal nonzero lengths")
    if any(label not in (0, 1, False, True) for label in y_true):
        raise ValueError("labels must be binary")
    if any(not isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("probabilities must be finite and within [0, 1]")


def classification_metrics(y_true: Sequence[int | bool], probabilities: Sequence[float], *,
                           threshold: float = 0.5, epsilon: float = 1e-12) -> ClassificationMetrics:
    _validate(y_true, probabilities)
    if not 0 <= threshold <= 1 or not 0 < epsilon < 0.5:
        raise ValueError("invalid threshold or epsilon")
    actual = [int(v) for v in y_true]
    predicted = [int(p >= threshold) for p in probabilities]
    tp = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    tn = sum(a == 0 and p == 0 for a, p in zip(actual, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    brier = sum((int(y) - p) ** 2 for y, p in zip(actual, probabilities)) / len(actual)
    logloss = -sum(int(y) * log(max(epsilon, p)) + (1 - int(y)) * log(max(epsilon, 1 - p))
                   for y, p in zip(actual, probabilities)) / len(actual)
    return ClassificationMetrics(len(actual), sum(a == b for a, b in zip(actual, predicted)) / len(actual),
        precision, recall, specificity, f1, brier, logloss,
        {"tp": tp, "fp": fp, "tn": tn, "fn": fn})


def reliability_bins(y_true: Sequence[int | bool], probabilities: Sequence[float], *,
                     bins: int = 10) -> tuple[ReliabilityBin, ...]:
    _validate(y_true, probabilities)
    if bins < 1 or bins > 1000:
        raise ValueError("bins must be between 1 and 1000")
    rows = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [(int(y), p) for y, p in zip(y_true, probabilities)
                    if lower <= p < upper or index == bins - 1 and p == 1.0]
        rows.append(ReliabilityBin(lower, upper, len(selected),
            sum(p for _, p in selected) / len(selected) if selected else 0.0,
            sum(y for y, _ in selected) / len(selected) if selected else 0.0))
    return tuple(rows)


def expected_calibration_error(y_true: Sequence[int | bool], probabilities: Sequence[float], *,
                               bins: int = 10) -> float:
    rows = reliability_bins(y_true, probabilities, bins=bins)
    total = sum(row.count for row in rows)
    return sum(row.count / total * abs(row.mean_probability - row.observed_rate)
               for row in rows if row.count) if total else 0.0


def search_threshold(y_true: Sequence[int | bool], probabilities: Sequence[float], *,
                     thresholds: Sequence[float] | None = None,
                     objective: str = "f1") -> ThresholdResult:
    _validate(y_true, probabilities)
    candidates = thresholds or tuple(i / 100 for i in range(1, 100))
    if any(not 0 <= threshold <= 1 for threshold in candidates):
        raise ValueError("all thresholds must be within [0, 1]")
    if objective not in {"f1", "balanced_accuracy", "precision", "recall"}:
        raise ValueError("unsupported objective")
    best = None
    for threshold in candidates:
        metrics = classification_metrics(y_true, probabilities, threshold=threshold)
        confusion = metrics.confusion
        if objective == "f1":
            score = metrics.f1
        elif objective == "balanced_accuracy":
            score = (metrics.recall + metrics.specificity) / 2
        elif objective == "precision":
            score = metrics.precision
        else:
            score = metrics.recall
        result = ThresholdResult(threshold, confusion["tp"], confusion["fp"], confusion["tn"],
            confusion["fn"], metrics.precision, metrics.recall, metrics.f1, score)
        if best is None or result.objective > best.objective or (
                result.objective == best.objective and result.threshold > best.threshold):
            best = result
    assert best is not None
    return best


def probability_rank_correlation(y_true: Sequence[int | bool], probabilities: Sequence[float]) -> float:
    """Simple rank correlation between probability and realized binary outcome."""
    _validate(y_true, probabilities)
    def ranks(values):
        ordered = sorted(enumerate(values), key=lambda pair: pair[1])
        result = [0.0] * len(values)
        i = 0
        while i < len(ordered):
            j = i + 1
            while j < len(ordered) and ordered[j][1] == ordered[i][1]:
                j += 1
            rank = (i + j - 1) / 2 + 1
            for k in range(i, j):
                result[ordered[k][0]] = rank
            i = j
        return result
    x, y = ranks([float(p) for p in probabilities]), ranks([float(v) for v in y_true])
    mx, my = sum(x) / len(x), sum(y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = sqrt(sum((a - mx) ** 2 for a in x))
    sy = sqrt(sum((b - my) ** 2 for b in y))
    return cov / (sx * sy) if sx and sy else 0.0
