import numpy as np
import pytest

from sfdao.evaluator.statistical import StatisticalEvaluator


def test_ks_test_identical_distributions():
    evaluator = StatisticalEvaluator()
    rng = np.random.default_rng(42)

    data = rng.normal(0, 1, 1000)
    synthetic = data.copy()

    result = evaluator.ks_test(data, synthetic)

    assert result.statistic == 0.0
    assert result.p_value == pytest.approx(1.0)


def test_ks_test_detects_difference():
    evaluator = StatisticalEvaluator()
    rng = np.random.default_rng(123)

    data = rng.normal(0, 1, 1000)
    shifted = rng.normal(2, 1, 1000)

    result = evaluator.ks_test(data, shifted)

    assert result.statistic > 0.1
    assert result.p_value < 0.01


def test_js_divergence_with_similar_distributions():
    evaluator = StatisticalEvaluator()
    rng = np.random.default_rng(7)

    data = rng.normal(0, 1, 2000)
    synthetic = rng.normal(0, 1, 2000)

    divergence = evaluator.js_divergence(data, synthetic, bins=50)

    assert 0 <= divergence <= 1
    assert divergence < 0.05


def test_js_divergence_detects_distance_between_distributions():
    evaluator = StatisticalEvaluator()
    rng = np.random.default_rng(99)

    data = rng.normal(0, 1, 2000)
    distant = rng.normal(5, 1, 2000)

    divergence = evaluator.js_divergence(data, distant, bins=50)

    assert 0 <= divergence <= 1
    assert divergence > 0.1


@pytest.mark.parametrize(
    "real,synthetic",
    [
        (np.array([]), np.array([1, 2, 3])),
        (np.array([1, 2, 3]), np.array([])),
        (np.array([]), np.array([])),
    ],
)
def test_empty_arrays_raise_value_error(real: np.ndarray, synthetic: np.ndarray):
    evaluator = StatisticalEvaluator()

    with pytest.raises(ValueError):
        evaluator.ks_test(real, synthetic)

    with pytest.raises(ValueError):
        evaluator.js_divergence(real, synthetic)
