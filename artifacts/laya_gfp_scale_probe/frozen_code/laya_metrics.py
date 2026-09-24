#!/usr/bin/env python3
"""Small, dependency-light classification and calibration metrics.

The full Laya experiments use this module for all task-level reports.  It is
deliberately implemented with NumPy only so that evaluation does not depend on
the training stack (PyTorch, Transformers, or scikit-learn).  ``logits`` are
always expected to contain one column for each real task class; callers must
not append padding columns.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


_ECE_BINS = 15


def _validate_inputs(
    logits: np.ndarray,
    labels: np.ndarray,
    n_classes: int | None,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Validate and normalize metric inputs.

    Labels are kept as integer class IDs.  ``assert`` is intentional for the
    finite-input invariant: silently propagating NaNs makes an evaluation
    report unusable and is much harder to diagnose than a failing run.
    """

    raw_logits = np.asarray(logits)
    if raw_logits.ndim != 2:
        raise ValueError(f"logits must have shape (N, K), got {raw_logits.shape}")
    if raw_logits.shape[0] == 0 or raw_logits.shape[1] == 0:
        raise ValueError("logits must contain at least one row and one class")
    # Convert before the check so integer inputs are supported while float16/
    # float32 calculations are accumulated in float64/longdouble below.
    try:
        logits64 = np.asarray(raw_logits, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("logits must be numeric") from exc
    assert np.isfinite(logits64).all(), "logits must contain only finite values"

    raw_labels = np.asarray(labels)
    if raw_labels.ndim != 1 or raw_labels.shape[0] != logits64.shape[0]:
        raise ValueError(
            "labels must be one-dimensional with one entry per logits row"
        )
    try:
        labels64 = np.asarray(raw_labels, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("labels must be integer class IDs") from exc
    assert np.isfinite(labels64).all(), "labels must contain only finite values"
    if not np.equal(labels64, np.floor(labels64)).all():
        raise ValueError("labels must be integer class IDs")
    labels_i = labels64.astype(np.int64)

    k = int(logits64.shape[1])
    if n_classes is not None:
        if not isinstance(n_classes, (int, np.integer)) or int(n_classes) != k:
            raise ValueError(
                "n_classes must equal logits.shape[1]; logits must not contain "
                "padded class columns"
            )
    if (labels_i < 0).any() or (labels_i >= k).any():
        raise ValueError(f"labels must be in [0, {k})")

    temp = float(temperature)
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError("temperature must be a finite positive scalar")
    return logits64, labels_i, k, temp


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Compute a row-wise softmax without overflow for finite logits."""

    # longdouble prevents a finite float64 max minus min from overflowing
    # during the max-shift (e.g. logits=[-1e308, 1e308]).  Returned values are
    # float64 because all reported metrics have that precision.
    values = np.asarray(logits, dtype=np.longdouble) / np.longdouble(temperature)
    shifted = values - np.max(values, axis=1, keepdims=True)
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        exponent = np.exp(shifted)
        probabilities = exponent / np.sum(exponent, axis=1, keepdims=True)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    # A longdouble-to-float64 conversion can leave a row sum one ulp away from
    # one.  Renormalizing also makes downstream Brier/ECE behavior explicit.
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _nll_from_probabilities(probabilities: np.ndarray, labels: np.ndarray) -> float:
    # A wrong class can underflow to probability zero for a valid extreme
    # logit.  Clipping at float64 tiny reports a finite, reproducible NLL while
    # preserving ordinary probabilities unchanged.
    p_true = np.maximum(probabilities[np.arange(labels.size), labels], np.finfo(float).tiny)
    return float(np.mean(-np.log(p_true)))


def classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    n_classes: int | None = None,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Return deterministic classification and calibration metrics.

    Parameters
    ----------
    logits:
        Array of shape ``(N, K)`` with one column for every real class.
    labels:
        Integer class IDs in ``[0, K)``.
    n_classes:
        Optional explicit task class count.  It must equal ``K``; this guards
        against accidentally evaluating padded classifier outputs.
    temperature:
        Positive softmax temperature used for NLL, Brier, and ECE.  Accuracy,
        balanced accuracy, macro-F1, and MCC use argmax and therefore do not
        change for a positive temperature.

    The macro metrics average over all ``K`` classes.  A class absent from the
    labels has recall and F1 equal to zero, so the denominator remains the
    fixed task class count rather than silently dropping missing classes.
    ``brier`` is the multiclass sum-of-squared-errors per example (range
    ``[0, 2]``), averaged over examples.  ``ece15`` uses 15 equal-width
    confidence bins, with confidence exactly one assigned to the final bin.
    """

    logits64, labels_i, k, temp = _validate_inputs(
        logits, labels, n_classes, temperature
    )
    probabilities = _softmax(logits64, temp)
    predictions = np.argmax(logits64, axis=1)
    n = int(labels_i.size)

    confusion = np.zeros((k, k), dtype=np.int64)
    np.add.at(confusion, (labels_i, predictions), 1)
    true_count = confusion.sum(axis=1).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)
    true_positive = np.diag(confusion).astype(np.float64)

    recalls = np.divide(
        true_positive,
        true_count,
        out=np.zeros(k, dtype=np.float64),
        where=true_count > 0,
    )
    precision_denominator = predicted_count
    precisions = np.divide(
        true_positive,
        precision_denominator,
        out=np.zeros(k, dtype=np.float64),
        where=precision_denominator > 0,
    )
    f1_denominator = 2.0 * true_positive + (predicted_count - true_positive) + (
        true_count - true_positive
    )
    f1 = np.divide(
        2.0 * true_positive,
        f1_denominator,
        out=np.zeros(k, dtype=np.float64),
        where=f1_denominator > 0,
    )

    # Multiclass MCC from the confusion matrix.  This is equivalent to the
    # standard covariance form and gives 0 for a degenerate denominator.
    n_float = float(n)
    numerator = n_float * float(true_positive.sum()) - float(
        np.dot(true_count, predicted_count)
    )
    denominator_left = n_float * n_float - float(np.dot(true_count, true_count))
    denominator_right = n_float * n_float - float(
        np.dot(predicted_count, predicted_count)
    )
    mcc_denominator = math.sqrt(max(0.0, denominator_left * denominator_right))
    mcc = numerator / mcc_denominator if mcc_denominator > 0.0 else 0.0

    confidence = probabilities.max(axis=1)
    correct = predictions == labels_i
    bin_index = np.minimum((confidence * _ECE_BINS).astype(np.int64), _ECE_BINS - 1)
    ece = 0.0
    for index in range(_ECE_BINS):
        in_bin = bin_index == index
        count = int(in_bin.sum())
        if count:
            ece += (count / n_float) * abs(
                float(correct[in_bin].mean()) - float(confidence[in_bin].mean())
            )

    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(n), labels_i] = 1.0
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    return {
        "n": n,
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(recalls.mean()),
        "macro_f1": float(f1.mean()),
        "mcc": float(mcc),
        "nll": _nll_from_probabilities(probabilities, labels_i),
        "brier": brier,
        "ece15": float(ece),
    }


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    n_classes: int | None = None,
    bounds: tuple[float, float] = (0.05, 20.0),
    max_iter: int = 80,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Fit one positive scalar temperature on calibration data only.

    A bounded golden-section search is used on ``log(temperature)`` and has no
    SciPy dependency.  The returned ``objective``/``nll_after`` is the NLL on
    exactly the supplied calibration rows; no validation or test data is
    accessed.  Endpoints and temperature one are explicitly considered so a
    flat objective and a no-op calibration remain deterministic.
    """

    logits64, labels_i, k, _ = _validate_inputs(logits, labels, n_classes, 1.0)
    del k  # validation above also enforces the no-padding invariant
    try:
        lower, upper = float(bounds[0]), float(bounds[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError("bounds must be a pair of positive finite scalars") from exc
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower <= 0.0
        or upper <= lower
    ):
        raise ValueError("bounds must satisfy 0 < lower < upper")
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError("max_iter must be a positive integer")
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("tolerance must be a finite positive scalar")

    def objective(temperature: float) -> float:
        return _nll_from_probabilities(_softmax(logits64, temperature), labels_i)

    initial_nll = objective(1.0)
    log_lower, log_upper = math.log(lower), math.log(upper)
    golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = log_lower, log_upper
    point1 = right - golden_ratio * (right - left)
    point2 = left + golden_ratio * (right - left)
    value1, value2 = objective(math.exp(point1)), objective(math.exp(point2))
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        if right - left <= float(tolerance):
            break
        if value1 <= value2:
            right, point2, value2 = point2, point1, value1
            point1 = right - golden_ratio * (right - left)
            value1 = objective(math.exp(point1))
        else:
            left, point1, value1 = point1, point2, value2
            point2 = left + golden_ratio * (right - left)
            value2 = objective(math.exp(point2))

    candidates = [
        (initial_nll, 1.0),
        (objective(lower), lower),
        (objective(upper), upper),
        (value1, math.exp(point1)),
        (value2, math.exp(point2)),
        (objective(math.exp((left + right) / 2.0)), math.exp((left + right) / 2.0)),
    ]
    best_nll, best_temperature = min(candidates, key=lambda item: (item[0], item[1]))
    return {
        "temperature": float(best_temperature),
        "objective": float(best_nll),
        "nll_before": float(initial_nll),
        "nll_after": float(best_nll),
        "success": bool(best_nll <= initial_nll + 1e-12),
        "n_iter": int(iterations),
        "bounds": [lower, upper],
    }


__all__ = ["classification_metrics", "fit_temperature"]
