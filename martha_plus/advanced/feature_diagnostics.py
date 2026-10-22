"""Feature missingness, drift, correlation, and labeling diagnostics."""
from __future__ import annotations
from math import isfinite, sqrt
from statistics import mean, median
from typing import Any, Mapping, Sequence


def _numeric(values: Sequence[Any]) -> list[float]:
    result = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if any(not isfinite(v) for v in result):
        raise ValueError("numeric feature values must be finite")
    return result


def feature_profile(rows: Sequence[Mapping[str, Any]], *, exclude: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    """Summarize feature completeness and simple distribution statistics."""
    names = sorted({str(key) for row in rows for key in row} - set(exclude))
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        values = [row.get(name) for row in rows]
        observed = _numeric(values)
        missing = len(values) - len(observed)
        result[name] = {"rows": len(values), "observed": len(observed), "missing": missing,
                        "missing_fraction": missing / len(values) if values else 0.0,
                        "unique_numeric": len(set(observed)),
                        "minimum": min(observed) if observed else None,
                        "maximum": max(observed) if observed else None,
                        "mean": mean(observed) if observed else None,
                        "median": median(observed) if observed else None,
                        "std": sqrt(sum((v - mean(observed)) ** 2 for v in observed) / max(1, len(observed) - 1)) if observed else None}
    return result


def population_stability_index(reference: Sequence[float], current: Sequence[float], *,
                               buckets: int = 10, epsilon: float = 1e-6) -> dict[str, Any]:
    """Estimate PSI using reference-distribution quantile bins and smoothed frequencies."""
    ref, cur = _numeric(reference), _numeric(current)
    if len(ref) < 2 or not cur:
        raise ValueError("reference requires >= 2 observations and current requires >= 1")
    if buckets < 2 or epsilon <= 0:
        raise ValueError("buckets must be >= 2 and epsilon positive")
    ref_sorted = sorted(ref)
    cuts = []
    for i in range(1, buckets):
        idx = min(len(ref_sorted) - 1, max(0, int(i * len(ref_sorted) / buckets)))
        cuts.append(ref_sorted[idx])
    # Equal values may create duplicate cutpoints; retain them for the requested bins
    # and use <= so ties do not change between runs.
    def counts(values: Sequence[float]) -> list[int]:
        counts_ = [0] * buckets
        for value in values:
            index = 0
            while index < len(cuts) and value > cuts[index]:
                index += 1
            counts_[index] += 1
        return counts_
    rc, cc = counts(ref), counts(cur)
    rtotal, ctotal = len(ref), len(cur)
    psi = 0.0
    details = []
    for i, (r_count, c_count) in enumerate(zip(rc, cc)):
        rp = max(epsilon, r_count / rtotal)
        cp = max(epsilon, c_count / ctotal)
        contribution = (cp - rp) * __import__("math").log(cp / rp)
        psi += contribution
        details.append({"bucket": i, "reference_count": r_count, "current_count": c_count,
                        "reference_fraction": r_count / rtotal, "current_fraction": c_count / ctotal,
                        "psi_contribution": contribution})
    interpretation = "low" if psi < 0.1 else ("moderate" if psi < 0.25 else "high")
    return {"psi": psi, "interpretation": interpretation, "reference_count": rtotal,
            "current_count": ctotal, "cutpoints": cuts, "buckets": details,
            "note": "PSI thresholds are heuristics; inspect feature semantics and sampling changes."}


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = _numeric(left), _numeric(right)
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("correlation inputs must have equal lengths >= 2")
    ma, mb = mean(a), mean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / sqrt(va * vb)


def correlation_matrix(columns: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float]]:
    names = sorted(columns)
    lengths = {len(columns[name]) for name in names}
    if len(lengths) > 1:
        raise ValueError("correlation series must have aligned lengths")
    return {a: {b: pearson_correlation(columns[a], columns[b]) if a != b else 1.0 for b in names}
            for a in names}


def missingness_report(rows: Sequence[Mapping[str, Any]], *, threshold: float = 0.2) -> dict[str, Any]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    names = sorted({key for row in rows for key in row})
    entries = {}
    flagged = []
    for name in names:
        absent = sum(1 for row in rows if name not in row or row.get(name) is None)
        fraction = absent / len(rows) if rows else 0.0
        entries[name] = {"missing": absent, "total": len(rows), "fraction": fraction}
        if fraction >= threshold:
            flagged.append(name)
    return {"rows": len(rows), "features": entries, "flagged": flagged, "threshold": threshold}


def alignment_diagnostics(feature_timestamps: Sequence[Any], label_timestamps: Sequence[Any],
                          *, feature_horizon: int = 0) -> dict[str, Any]:
    """Inspect time alignment without claiming to prove absence of look-ahead bias.

    A feature timestamp later than its label timestamp is a clear temporal leakage
    risk. The expected horizon is explicit and merely informational; the caller must
    define whether labels represent the next bar, an event interval, or an endpoint.
    """
    if len(feature_timestamps) != len(label_timestamps):
        raise ValueError("feature and label timestamps must align one-to-one")
    if feature_horizon < 0:
        raise ValueError("feature_horizon must be non-negative")
    violations = []
    for i, (feature_time, label_time) in enumerate(zip(feature_timestamps, label_timestamps)):
        if feature_time is None or label_time is None:
            continue
        if feature_time > label_time:
            violations.append({"index": i, "feature_timestamp": str(feature_time),
                               "label_timestamp": str(label_time)})
    return {"aligned_count": len(feature_timestamps), "feature_horizon": feature_horizon,
            "temporal_violations": violations, "potential_leakage": bool(violations),
            "warning": "Timestamp ordering is necessary but not sufficient; audit every feature's information availability."}


def detect_constant_features(columns: Mapping[str, Sequence[float]]) -> tuple[str, ...]:
    constant = []
    for name, values in columns.items():
        observed = _numeric(values)
        if observed and max(observed) == min(observed):
            constant.append(name)
        elif not observed:
            constant.append(name)
    return tuple(sorted(constant))


def compare_profiles(reference: Mapping[str, Mapping[str, Any]],
                     current: Mapping[str, Mapping[str, Any]],
                     missing_fraction_delta: float = 0.1) -> dict[str, Any]:
    if not 0 <= missing_fraction_delta <= 1:
        raise ValueError("missing fraction delta must be in [0, 1]")
    changes = []
    for name in sorted(set(reference) | set(current)):
        if name not in reference:
            changes.append({"feature": name, "change": "added"})
        elif name not in current:
            changes.append({"feature": name, "change": "removed"})
        else:
            before = float(reference[name].get("missing_fraction", 0.0))
            after = float(current[name].get("missing_fraction", 0.0))
            delta = after - before
            if abs(delta) >= missing_fraction_delta:
                changes.append({"feature": name, "change": "missingness_shift", "delta": delta,
                                "before": before, "after": after})
    return {"matches": not changes, "changes": changes, "change_count": len(changes)}
