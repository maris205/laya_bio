"""Focused CPU checks for :mod:`laya_metrics`.

These tests intentionally avoid a dependency on scikit-learn.  The formulas
are small enough to check against hand-computed values and the tests exercise
the edge cases that can silently corrupt a calibration report.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from laya_metrics import classification_metrics, fit_temperature  # noqa: E402


def test_perfect_multiclass_metrics_are_perfect():
    logits = np.diag([10.0, 10.0, 10.0])
    result = classification_metrics(logits, np.arange(3), n_classes=3)

    assert result["n"] == 3
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "mcc"):
        assert result[key] == pytest.approx(1.0)
    assert result["nll"] < 1e-3
    assert result["brier"] < 1e-6
    assert result["ece15"] < 1e-3


def test_uniform_probabilities_and_fixed_class_macros():
    # Class 2 is absent from labels.  It must still contribute zero to both
    # balanced accuracy and macro-F1 rather than being dropped from the mean.
    result = classification_metrics(np.zeros((4, 3)), [0, 1, 0, 0], n_classes=3)
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["balanced_accuracy"] == pytest.approx(1.0 / 3.0)
    assert result["macro_f1"] == pytest.approx(2.0 / 7.0)
    assert result["nll"] == pytest.approx(math.log(3.0))
    assert result["brier"] == pytest.approx(2.0 / 3.0)
    # Confidence is 1/3 and accuracy is 3/4 for every row.
    assert result["ece15"] == pytest.approx(5.0 / 12.0)


def test_discrete_metrics_match_sklearn_when_available():
    """Cross-check confusion-matrix metrics against the standard reference."""
    sklearn = pytest.importorskip("sklearn.metrics")
    logits = np.array(
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [3.0, 0.0, 0.0],
         [0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [0.0, 3.0, 0.0], [3.0, 0.0, 0.0]]
    )
    labels = np.array([0, 1, 2, 1, 1, 0, 2, 2])
    prediction = np.argmax(logits, axis=1)
    result = classification_metrics(logits, labels, n_classes=3)
    assert result["accuracy"] == pytest.approx(sklearn.accuracy_score(labels, prediction))
    assert result["balanced_accuracy"] == pytest.approx(
        sklearn.balanced_accuracy_score(labels, prediction)
    )
    assert result["macro_f1"] == pytest.approx(
        sklearn.f1_score(labels, prediction, average="macro", zero_division=0)
    )
    assert result["mcc"] == pytest.approx(sklearn.matthews_corrcoef(labels, prediction))


def test_ece_confidence_one_is_in_final_bin():
    result = classification_metrics(
        np.array([[1000.0, -1000.0], [-1000.0, 1000.0]]), [0, 1], n_classes=2
    )
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["ece15"] == pytest.approx(0.0)

    wrong = classification_metrics(
        np.array([[1000.0, -1000.0], [-1000.0, 1000.0]]), [1, 0], n_classes=2
    )
    assert wrong["ece15"] == pytest.approx(1.0)


def test_extreme_finite_logits_do_not_overflow():
    result = classification_metrics(
        np.array([[1e308, -1e308], [-1e308, 1e308]]), [0, 1], n_classes=2
    )
    assert all(math.isfinite(float(result[key])) for key in result if key != "n")
    assert result["accuracy"] == pytest.approx(1.0)


def test_temperature_fit_does_not_worsen_calibration_nll():
    # Deliberately overconfident and partly mislabeled predictions.  A single
    # scalar temperature should flatten these distributions and lower NLL.
    logits = np.array(
        [[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0], [8.0, 0.0], [0.0, 8.0]]
    )
    labels = np.array([0, 1, 1, 0, 0, 1])
    fitted = fit_temperature(logits, labels, n_classes=2)
    assert 0.05 <= fitted["temperature"] <= 20.0
    assert fitted["objective"] == pytest.approx(fitted["nll_after"])
    assert fitted["nll_after"] <= fitted["nll_before"] + 1e-12
    assert fitted["success"]


def test_invalid_or_padded_inputs_fail_loudly():
    with pytest.raises(AssertionError):
        classification_metrics(np.array([[np.nan, 0.0]]), [0])
    with pytest.raises(ValueError):
        classification_metrics(np.zeros((2, 3)), [0, 3])
    with pytest.raises(ValueError, match="padded"):
        classification_metrics(np.zeros((2, 3)), [0, 1], n_classes=4)
