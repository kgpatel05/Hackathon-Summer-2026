"""Cohort-level decision rules for Iteration 9.

The classifiers in earlier iterations make one independent argmax decision per cell.
That is suboptimal when two facts hold simultaneously:

1. train and test are IID halves of the same sections and animals; and
2. several rare classes are well ranked (high one-vs-rest AUC) but almost never win an
   argmax after probability shrinkage.

This module uses only labelled-training counts and unlabelled evaluation probabilities.
It never reads evaluation labels.  The decoder solves a linear assignment problem inside
each deterministic metadata stratum.  A tunable ``strength`` interpolates between the
model's unconstrained predicted counts (0.0, an exact no-op) and counts expected from the
labelled cells (1.0).  Intermediate values are a robust, soft cohort prior: they retain
the model's evidence about sampling fluctuation while preventing a few common classes
from consuming every ambiguous cell.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


MASK_COLS = ("Region", "Excitatory_vs_Inhibitory", "Segment")


def compatibility_mask(
    meta_train: pd.DataFrame,
    y_train: np.ndarray,
    meta_eval: pd.DataFrame,
    classes: Sequence[str],
    columns: Sequence[str] = MASK_COLS,
) -> np.ndarray:
    """Return whether each evaluation cell/class pair is training-compatible.

    A value that was never observed in the labelled data is left unconstrained.  This
    makes the routine safe for future data while reproducing the adopted Iteration 8
    hard metadata mask on the present challenge.
    """
    labels = np.asarray(y_train, dtype=str)
    allow = np.ones((len(meta_eval), len(classes)), dtype=bool)
    for column in columns:
        train_values = meta_train[column].astype(str).to_numpy()
        known = set(train_values)
        seen = [set(train_values[labels == label]) for label in classes]
        for row, value in enumerate(meta_eval[column].astype(str).to_numpy()):
            if value in known:
                allow[row] &= np.fromiter(
                    (value in class_values for class_values in seen),
                    dtype=bool,
                    count=len(classes),
                )
    allow[~allow.any(axis=1)] = True
    return allow


def metadata_strata(
    meta: pd.DataFrame,
    columns: Sequence[str] = MASK_COLS,
) -> np.ndarray:
    """Stable string key for the deterministic label-compatible metadata tuple."""
    values = meta[list(columns)].astype(str).to_numpy()
    return np.asarray(["\x1f".join(row) for row in values], dtype=object)


def largest_remainder(values: np.ndarray, total: int) -> np.ndarray:
    """Round non-negative expected counts to integers summing exactly to ``total``."""
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    if total == 0:
        return np.zeros_like(values, dtype=int)
    if values.sum() <= 0:
        raise ValueError("cannot allocate a positive total from all-zero counts")
    scaled = values * (total / values.sum())
    result = np.floor(scaled).astype(int)
    missing = total - int(result.sum())
    if missing:
        order = np.argsort(-(scaled - result), kind="stable")
        result[order[:missing]] += 1
    return result


def _decode_one_stratum(
    probabilities: np.ndarray,
    allowed: np.ndarray,
    expected_counts: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Maximum-likelihood assignment at a blended class-count target."""
    n_rows, n_classes = probabilities.shape
    masked = np.where(allowed, probabilities, -1.0)
    baseline = masked.argmax(axis=1)
    if strength <= 0 or n_rows <= 1:
        return baseline

    baseline_counts = np.bincount(baseline, minlength=n_classes)
    target = (1.0 - strength) * baseline_counts + strength * expected_counts
    # A class disallowed for every row must never receive a slot.
    target[~allowed.any(axis=0)] = 0.0
    quotas = largest_remainder(target, n_rows)
    slots = np.repeat(np.arange(n_classes), quotas)
    if len(slots) != n_rows:
        raise AssertionError("quota rounding did not preserve the stratum size")

    # Repeated columns are label slots.  Hungarian assignment maximises the joint
    # likelihood subject to using every slot exactly once.  Incompatible assignments
    # receive a finite but overwhelming cost so scipy never sees inf/nan.
    selected = probabilities[:, slots]
    selected_allowed = allowed[:, slots]
    costs = -np.log(np.clip(selected, 1e-12, 1.0))
    costs[~selected_allowed] = 1e6
    rows, columns = linear_sum_assignment(costs)
    assigned = np.empty(n_rows, dtype=int)
    assigned[rows] = slots[columns]
    if np.any(~allowed[np.arange(n_rows), assigned]):
        # This should be impossible because targets are zeroed for globally disallowed
        # classes.  Fall back rather than emit an anatomically impossible prediction.
        assigned = baseline
    return assigned


def _project_counts_to_bounds(
    baseline_counts: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    total: int,
) -> np.ndarray:
    """Closest integer count vector to baseline subject to bounds and fixed total."""
    baseline_counts = np.asarray(baseline_counts, dtype=int)
    lower = np.asarray(lower, dtype=int)
    upper = np.asarray(upper, dtype=int)
    if lower.sum() > total or upper.sum() < total:
        raise ValueError("infeasible class-count interval")
    counts = np.clip(baseline_counts, lower, upper)

    # Greedily minimise squared displacement from the baseline.  For separable convex
    # integer costs, choosing the cheapest next unit is the exact projection.
    while counts.sum() < total:
        candidates = np.flatnonzero(counts < upper)
        delta = 2 * (counts[candidates] - baseline_counts[candidates]) + 1
        counts[candidates[np.argmin(delta)]] += 1
    while counts.sum() > total:
        candidates = np.flatnonzero(counts > lower)
        delta = -2 * (counts[candidates] - baseline_counts[candidates]) + 1
        counts[candidates[np.argmin(delta)]] -= 1
    return counts


def _decode_one_interval(
    probabilities: np.ndarray,
    allowed: np.ndarray,
    expected_counts: np.ndarray,
    train_size: int,
    z_score: float,
) -> np.ndarray:
    """Decode only when predicted totals leave an IID two-sample count interval."""
    n_rows, n_classes = probabilities.shape
    masked = np.where(allowed, probabilities, -1.0)
    baseline = masked.argmax(axis=1)
    baseline_counts = np.bincount(baseline, minlength=n_classes)

    # Var(N_eval - r*N_train) under two IID multinomials, with p estimated from train.
    p = expected_counts / max(n_rows, 1)
    ratio = n_rows / max(train_size, 1)
    variance = n_rows * p * (1.0 - p) * (1.0 + ratio)
    radius = z_score * np.sqrt(np.maximum(variance, 0.0))
    lower = np.floor(np.maximum(0.0, expected_counts - radius)).astype(int)
    upper = np.ceil(expected_counts + radius).astype(int)
    lower[~allowed.any(axis=0)] = 0
    upper[~allowed.any(axis=0)] = 0
    upper = np.minimum(upper, n_rows)

    # Numerical/very-small-stratum fallback: the observed baseline itself provides a
    # feasible relaxation, while still retaining every feasible theoretical bound.
    if lower.sum() > n_rows:
        lower = largest_remainder(lower, n_rows)
    if upper.sum() < n_rows:
        deficit = n_rows - int(upper.sum())
        order = np.argsort(-baseline_counts, kind="stable")
        for k in order:
            room = n_rows - upper[k] if allowed[:, k].any() else 0
            take = min(deficit, room)
            upper[k] += take
            deficit -= take
            if deficit == 0:
                break

    quotas = _project_counts_to_bounds(baseline_counts, lower, upper, n_rows)
    if np.array_equal(quotas, baseline_counts):
        return baseline
    slots = np.repeat(np.arange(n_classes), quotas)
    selected = probabilities[:, slots]
    selected_allowed = allowed[:, slots]
    costs = -np.log(np.clip(selected, 1e-12, 1.0))
    costs[~selected_allowed] = 1e6
    rows, columns = linear_sum_assignment(costs)
    assigned = np.empty(n_rows, dtype=int)
    assigned[rows] = slots[columns]
    if np.any(~allowed[np.arange(n_rows), assigned]):
        return baseline
    return assigned


def quota_decode(
    probabilities: np.ndarray,
    meta_train: pd.DataFrame,
    y_train: np.ndarray,
    meta_eval: pd.DataFrame,
    classes: Sequence[str],
    strength: float,
    columns: Sequence[str] = MASK_COLS,
) -> np.ndarray:
    """Jointly decode evaluation labels using train-derived within-stratum counts.

    ``strength=0`` is guaranteed to equal metadata-masked per-cell argmax.  At positive
    strength, expected evaluation counts are the labelled class proportions inside the
    same stratum multiplied by the number of evaluation cells in that stratum.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(y_train, dtype=str)
    if probabilities.shape != (len(meta_eval), len(classes)):
        raise ValueError("probability shape does not match evaluation rows/classes")

    allowed = compatibility_mask(meta_train, labels, meta_eval, classes, columns)
    train_groups = metadata_strata(meta_train, columns)
    eval_groups = metadata_strata(meta_eval, columns)
    class_index = {label: i for i, label in enumerate(classes)}
    output = np.empty(len(meta_eval), dtype=int)

    for group in np.unique(eval_groups):
        eval_rows = np.flatnonzero(eval_groups == group)
        train_rows = np.flatnonzero(train_groups == group)
        local_prob = probabilities[eval_rows]
        local_allow = allowed[eval_rows]

        if len(train_rows) == 0:
            output[eval_rows] = np.where(local_allow, local_prob, -1.0).argmax(axis=1)
            continue

        counts = np.zeros(len(classes), dtype=float)
        for label, count in zip(*np.unique(labels[train_rows], return_counts=True)):
            counts[class_index[label]] = count
        expected = counts * (len(eval_rows) / len(train_rows))
        output[eval_rows] = _decode_one_stratum(
            local_prob, local_allow, expected, strength
        )
    return output


def interval_decode(
    probabilities: np.ndarray,
    meta_train: pd.DataFrame,
    y_train: np.ndarray,
    meta_eval: pd.DataFrame,
    classes: Sequence[str],
    z_score: float = 1.96,
    columns: Sequence[str] = MASK_COLS,
) -> np.ndarray:
    """Joint decoder with theoretical IID count intervals instead of fixed quotas.

    The metadata-masked argmax is retained unless a predicted class total falls outside
    the approximate two-sided ``z_score`` sampling interval around the count inferred
    from the labelled cohort.  This makes the intervention data-adaptive but not
    label-adaptive: evaluation labels are never used.
    """
    if z_score <= 0:
        raise ValueError("z_score must be positive")
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(y_train, dtype=str)
    allowed = compatibility_mask(meta_train, labels, meta_eval, classes, columns)
    train_groups = metadata_strata(meta_train, columns)
    eval_groups = metadata_strata(meta_eval, columns)
    class_index = {label: i for i, label in enumerate(classes)}
    output = np.empty(len(meta_eval), dtype=int)

    for group in np.unique(eval_groups):
        eval_rows = np.flatnonzero(eval_groups == group)
        train_rows = np.flatnonzero(train_groups == group)
        local_prob = probabilities[eval_rows]
        local_allow = allowed[eval_rows]
        if len(train_rows) == 0:
            output[eval_rows] = np.where(local_allow, local_prob, -1.0).argmax(axis=1)
            continue

        counts = np.zeros(len(classes), dtype=float)
        for label, count in zip(*np.unique(labels[train_rows], return_counts=True)):
            counts[class_index[label]] = count
        expected = counts * (len(eval_rows) / len(train_rows))
        output[eval_rows] = _decode_one_interval(
            local_prob, local_allow, expected, len(train_rows), z_score
        )
    return output
