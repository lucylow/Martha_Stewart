"""Parameter-grid diagnostics, Pareto filtering, and boundary robustness checks."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import Any, Callable, Iterable, Mapping, Sequence
from .experiments import fingerprint


@dataclass(frozen=True, slots=True)
class ParameterTrial:
    parameters: Mapping[str, Any]
    parameter_hash: str
    metrics: Mapping[str, float]
    valid: bool = True
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SensitivityReport:
    trial_count: int
    valid_count: int
    best_metric: float | None
    metric_spread: float | None
    adjacent_sign_change_fraction: float | None
    boundary_optimum: bool
    warnings: tuple[str, ...]


def expand_parameter_grid(grid: Mapping[str, Sequence[Any]], *, max_trials: int = 100_000) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    names = sorted(grid)
    values = [list(grid[name]) for name in names]
    if any(not values_for_name for values_for_name in values):
        raise ValueError("each parameter must have at least one value")
    count = 1
    for values_for_name in values:
        count *= len(values_for_name)
        if count > max_trials:
            raise ValueError(f"grid expands to more than max_trials={max_trials}")
    return [dict(zip(names, combo)) for combo in product(*values)]


def run_parameter_trials(grid: Mapping[str, Sequence[Any]], evaluator: Callable[[Mapping[str, Any]], Mapping[str, float]],
                         *, max_trials: int = 100_000) -> list[ParameterTrial]:
    trials: list[ParameterTrial] = []
    for parameters in expand_parameter_grid(grid, max_trials=max_trials):
        try:
            metrics_raw = evaluator(parameters)
            metrics = {str(k): float(v) for k, v in metrics_raw.items()}
            if any(not isfinite(v) for v in metrics.values()):
                raise ValueError("evaluator returned non-finite metric")
            trials.append(ParameterTrial(parameters, fingerprint(parameters), metrics))
        except Exception as exc:
            trials.append(ParameterTrial(parameters, fingerprint(parameters), {}, False,
                                         f"{type(exc).__name__}: {exc}"))
    return trials


def pareto_frontier(rows: Sequence[Mapping[str, float]], *, maximize: Sequence[str],
                    minimize: Sequence[str] = ()) -> list[int]:
    """Return row indices not dominated on all configured objectives."""
    if not maximize and not minimize:
        raise ValueError("at least one objective is required")
    objectives = list(maximize) + list(minimize)
    for row in rows:
        for key in objectives:
            if key not in row or not isfinite(float(row[key])):
                raise ValueError(f"missing or invalid objective {key}")
    def dominates(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
        at_least_as_good = all(float(a[k]) >= float(b[k]) for k in maximize) and all(float(a[k]) <= float(b[k]) for k in minimize)
        strictly_better = any(float(a[k]) > float(b[k]) for k in maximize) or any(float(a[k]) < float(b[k]) for k in minimize)
        return at_least_as_good and strictly_better
    return [i for i, row in enumerate(rows) if not any(j != i and dominates(other, row) for j, other in enumerate(rows))]


def summarize_sensitivity(trials: Sequence[ParameterTrial], metric: str,
                          *, parameter_name: str | None = None) -> SensitivityReport:
    valid = [trial for trial in trials if trial.valid and metric in trial.metrics]
    warnings: list[str] = []
    if not valid:
        return SensitivityReport(len(trials), 0, None, None, None, False,
                                 ("no valid trials contain the requested metric",))
    values = [float(trial.metrics[metric]) for trial in valid]
    if any(not isfinite(v) for v in values):
        raise ValueError("metric values must be finite")
    spread = max(values) - min(values)
    best_idx = max(range(len(valid)), key=lambda i: values[i])
    boundary = False
    sign_change: float | None = None
    if parameter_name:
        points = []
        for trial in valid:
            if parameter_name not in trial.parameters:
                raise ValueError(f"parameter {parameter_name} missing from a trial")
            points.append((float(trial.parameters[parameter_name]), float(trial.metrics[metric])))
        points.sort()
        if len(points) > 1:
            best_param = points[max(range(len(points)), key=lambda i: points[i][1])][0]
            boundary = best_param in {points[0][0], points[-1][0]}
            slopes = [points[i + 1][1] - points[i][1] for i in range(len(points) - 1)]
            sign_changes = sum((a < 0 < b) or (b < 0 < a) for a, b in zip(slopes, slopes[1:]))
            sign_change = sign_changes / max(1, len(slopes) - 1)
            if boundary:
                warnings.append("best observed parameter is on a grid boundary; expand the search region")
            if spread > 0 and max(values) - sorted(values)[-2] > spread * 0.5 if len(values) > 1 else False:
                warnings.append("best metric is a sharp outlier relative to other trials")
    return SensitivityReport(len(trials), len(valid), max(values), spread, sign_change, boundary, tuple(warnings))


def compare_walkforward_windows(scores: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Compare strategy scores across windows and report simple consistency metrics."""
    if not scores:
        return {"strategies": {}, "winner": None}
    result: dict[str, dict[str, float | int]] = {}
    for name, raw in scores.items():
        values = [float(v) for v in raw]
        if any(not isfinite(v) for v in values):
            raise ValueError("scores must be finite")
        avg = sum(values) / len(values) if values else 0.0
        variance = sum((v - avg) ** 2 for v in values) / max(1, len(values) - 1)
        result[name] = {"windows": len(values), "mean_score": avg,
                        "score_std": variance ** 0.5, "positive_window_fraction": sum(v > 0 for v in values) / len(values) if values else 0.0}
    winner = max(result, key=lambda name: result[name]["mean_score"])
    return {"strategies": result, "winner": winner,
            "warning": "Window scores are correlated and are not independent evidence of future profitability."}
